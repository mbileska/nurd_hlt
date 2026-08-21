"""
Roy's Linformer/Transformer contrastive encoder adapted for the NURD interface.

NURD expects:   activations, logits = model(inputs)
We return:      activations = encoder latent  [B, latent_dim]
                logits      = classifier head  [B, num_classes]

The contrastive loss (InfoNCE / SupCon) is computed outside using
model.get_embeddings(activations).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-4

# ── PF preprocessor ───────────────────────────────────────────────────────────

class PFPreProcessor(nn.Module):
    PDGIDs = [211, 11, 13, 22, 130, 1, 2]

    def __init__(self):
        super().__init__()
        self.register_buffer("avail_pdgIds", torch.tensor(self.PDGIDs, dtype=torch.long))
        self.num_features_cont = 6
        self.num_features_disc = 2 + len(self.PDGIDs)
        self.num_features      = self.num_features_cont + self.num_features_disc
        self.batch_norm        = nn.BatchNorm1d(self.num_features_cont)

    def pdgId_to_onehot(self, pid):
        return (pid.long().unsqueeze(-1).abs() == self.avail_pdgIds).float()

    def forward(self, x):
        pt_raw, eta_raw, phi_raw = x[...,0], x[...,1], x[...,2]
        dxy_raw, dxysig_raw      = x[...,3], x[...,4]
        is_pf_raw, pdgId_raw     = x[...,5], x[...,6]

        valid = pt_raw > 0
        if valid.any():
            pt = torch.where(valid, pt_raw, torch.zeros_like(pt_raw))
            pt = pt / (pt.sum(dim=-1, keepdim=True) + EPS)
            pt[valid] = torch.log(pt[valid])

            dxy = torch.where(valid, dxy_raw, torch.zeros_like(dxy_raw))
            dxy[valid] = torch.tanh(dxy_raw[valid])

            energy = torch.zeros_like(pt_raw)
            energy[valid] = (pt_raw * torch.cosh(eta_raw)).abs()[valid]
            energy = energy / (energy.sum(dim=-1, keepdim=True) + EPS)
            energy[valid] = torch.log(energy[valid])

            pos = (pdgId_raw==11)|(pdgId_raw==13)|(pdgId_raw==211)
            neg = (pdgId_raw==-11)|(pdgId_raw==-13)|(pdgId_raw==-211)
            charge = torch.zeros_like(valid, dtype=torch.float)
            charge[valid & pos] =  1.0
            charge[valid & neg] = -1.0

            x_proc = torch.cat([
                pt.unsqueeze(-1), eta_raw.unsqueeze(-1), phi_raw.unsqueeze(-1),
                dxy.unsqueeze(-1), dxysig_raw.unsqueeze(-1), energy.unsqueeze(-1),
                charge.unsqueeze(-1), is_pf_raw.unsqueeze(-1),
                self.pdgId_to_onehot(pdgId_raw)
            ], dim=-1)

            B, N, C = x_proc[..., :self.num_features_cont].shape
            cont_flat = x_proc[..., :self.num_features_cont].reshape(B*N, C)
            vf = valid.reshape(B*N)
            cont_flat[vf] = self.batch_norm(cont_flat[vf])
            x_proc[..., :self.num_features_cont] = cont_flat.reshape(B, N, C)
        else:
            x_proc = torch.zeros(*x.shape[:-1], self.num_features, device=x.device)
        return x_proc


# ── Linformer / Transformer blocks ────────────────────────────────────────────

class _LinearAttn(nn.Module):
    def __init__(self, embed_dim, num_heads, linear_dim, num_tokens):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.o = nn.Linear(embed_dim, embed_dim)
        self.e = nn.Linear(num_tokens, linear_dim, bias=False)
        self.f = nn.Linear(num_tokens, linear_dim, bias=False)

    def forward(self, x, mask=None):
        B, N, E = x.shape
        H, D = self.num_heads, self.head_dim

        Q = self.q(x).view(B,N,H,D).transpose(1,2)
        K = self.k(x).view(B,N,H,D).transpose(1,2)
        V = self.v(x).view(B,N,H,D).transpose(1,2)

        if mask is not None:
            m = mask.unsqueeze(1).unsqueeze(-1).expand(B,1,N,D)
            K = K.masked_fill(m, 0.0)
            V = V.masked_fill(m, 0.0)

        Kp = self.e(K.transpose(2,3)).transpose(2,3)
        Vp = self.f(V.transpose(2,3)).transpose(2,3)

        scores = torch.matmul(Q.float(), Kp.float().transpose(-2,-1)) / math.sqrt(D)
        attn   = torch.softmax(scores, dim=-1)
        out    = torch.matmul(attn, Vp.float()).to(Q.dtype)
        out    = out.transpose(1,2).contiguous().view(B,N,E)
        return self.o(out)


class _Block(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_ff, linear_dim, num_tokens, dropout=0.1):
        super().__init__()
        self.attn   = _LinearAttn(embed_dim, num_heads, linear_dim, num_tokens)
        self.ff1    = nn.Linear(embed_dim, dim_ff)
        self.ff2    = nn.Linear(dim_ff, embed_dim)
        self.norm1  = nn.LayerNorm(embed_dim)
        self.norm2  = nn.LayerNorm(embed_dim)
        self.drop1  = nn.Dropout(dropout)
        self.drop2  = nn.Dropout(dropout)
        self.act    = nn.ReLU()

    def forward(self, x, mask=None):
        x = self.norm1(x + self.drop1(self.attn(x, mask)))
        x = self.norm2(x + self.drop2(self.ff2(self.drop1(self.act(self.ff1(x))))))
        return x


# ── Full HLT contrastive model ────────────────────────────────────────────────

class HLTContrastiveModel(nn.Module):
    """
    TransformerEncoder (Linformer) + Projector + Classifier.

    Forward returns (latent, logits) for the NURD interface.
    Use get_embeddings(latent) for the contrastive loss.
    """
    def __init__(
        self,
        num_classes: int,
        embed_size:  int  = 128,
        latent_dim:  int  = 6,
        proj_dim:    int  = 6,
        num_heads:   int  = 8,
        num_layers:  int  = 4,
        dim_ff:      int  = 512,
        linear_dim:  int  = 16,
        num_tokens:  int  = 100,   # max PF candidates in data
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.preproc   = PFPreProcessor()
        n_feat         = self.preproc.num_features

        self.input_proj = nn.Linear(n_feat, embed_size)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, embed_size))
        self.layers     = nn.ModuleList([
            _Block(embed_size, num_heads, dim_ff, linear_dim, num_tokens+1, dropout)
            for _ in range(num_layers)
        ])
        self.norm_cls   = nn.LayerNorm(embed_size)
        self.bottleneck = nn.Linear(embed_size, latent_dim)

        self.projector  = nn.Sequential(
            nn.Linear(latent_dim, proj_dim*4), nn.BatchNorm1d(proj_dim*4), nn.GELU(),
            nn.Linear(proj_dim*4, proj_dim*4), nn.BatchNorm1d(proj_dim*4), nn.GELU(),
            nn.Linear(proj_dim*4, proj_dim),
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def _make_mask(self, x):
        """Prepend a False (unpadded) slot for the CLS token."""
        pad = (x[..., 0] == 0)                                          # [B, N]
        cls = torch.zeros(x.size(0), 1, device=x.device, dtype=torch.bool)
        return torch.cat([cls, pad], dim=1)                              # [B, N+1]

    def forward(self, x: torch.Tensor):
        """
        x: [B, N, 7]  raw PF candidates
        Returns: (latent [B, latent_dim], logits [B, num_classes])
        """
        mask = self._make_mask(x)
        x    = self.preproc(x)                                           # [B, N, n_feat]
        x    = self.input_proj(x)                                        # [B, N, E]
        cls  = self.cls_token.expand(x.size(0), -1, -1)
        x    = torch.cat([cls, x], dim=1)                                # [B, N+1, E]

        for layer in self.layers:
            x = layer(x, mask)

        latent  = self.bottleneck(self.norm_cls(x[:, 0, :]))             # [B, latent_dim]
        emb     = F.normalize(self.projector(latent), dim=1)             # [B, proj_dim]
        logits  = self.classifier(latent)                                # [B, num_classes]
        return latent, logits

    def get_embeddings(self, latent: torch.Tensor) -> torch.Tensor:
        """Normalized projector output — used for the contrastive loss."""
        return F.normalize(self.projector(latent), dim=1)


# ── Critic model ──────────────────────────────────────────────────────────────

class HLTCritic(nn.Module):
    """Continuous-nuisance density-ratio critic.

    The critic follows the engineer reference: it distinguishes real
    ``(r(x), z, y)`` tuples from tuples whose continuous nuisance ``z`` was
    shuffled. Labels are represented explicitly as one-hot values so the four
    physics classes are not assigned an artificial ordinal relationship.
    """

    def __init__(self, latent_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = int(num_classes)
        self.net = nn.Sequential(
            nn.Linear(int(latent_dim) + 1 + self.num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        latent: torch.Tensor,
        labels: torch.Tensor,
        nuisance: torch.Tensor,
    ) -> torch.Tensor:
        labels = labels.long().reshape(-1)
        nuisance = nuisance.float().reshape(-1, 1)
        label_features = F.one_hot(
            labels, num_classes=self.num_classes).to(latent.dtype)
        return self.net(torch.cat(
            [latent.float(), nuisance, label_features], dim=1))
