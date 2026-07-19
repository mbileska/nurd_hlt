"""
HLT SM Cocktail dataset for NURD training.

The nuisance variable z is the **binned AE reconstruction loss**.
NURD exact weights w(y,z) = p(y)*p(z)/p(y,z) are pre-computed on load
so that train_exact.py can look them up with dataset.weights[(y,z)].

Dataset returns (pf_features, label, nuisance_bin, ae_reco, nurd_weight)
per event.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter
from sklearn.model_selection import train_test_split


def _make_nurd_weights(labels, nuisances, max_weight_ratio=10.0):
    """
    Exact NURD weights: w(y,z) = p(y)*p(z)/p(y,z) = n_y*n_z / (N*n_yz).
    Under this weighting, y and z are marginally independent.
    Normalized so that the per-sample mean weight equals 1, then clipped at
    max_weight_ratio × mean to prevent extreme weights from destabilising training.
    """
    N = len(labels)
    labels_list    = [int(y) for y in labels.tolist()]
    nuisances_list = [int(z) for z in nuisances.tolist()]

    group_counts    = Counter(zip(labels_list, nuisances_list))
    label_counts    = Counter(labels_list)
    nuisance_counts = Counter(nuisances_list)

    weights_raw = {
        (y, z): (label_counts[y] * nuisance_counts[z]) / (N * n_yz)
        for (y, z), n_yz in group_counts.items()
    }
    # normalize so E[w] = 1 over all training samples
    mean_w = sum(weights_raw[k] * v for k, v in group_counts.items()) / N
    weights_norm = {k: v / mean_w for k, v in weights_raw.items()}
    # clip to max_weight_ratio × 1.0 (since mean is now 1) to reduce variance
    cap = max_weight_ratio
    return {k: min(v, cap) for k, v in weights_norm.items()}


def _as_float_tensor(value):
    if torch.is_tensor(value):
        return value.detach().cpu().float()
    return torch.as_tensor(value, dtype=torch.float32)


def _compute_ae_reco(obj_data, ae_model, batch_size=4096):
    """Compute AE reconstruction loss once for all events."""
    ae_model.eval()
    device = next(ae_model.parameters()).device
    mse = torch.nn.MSELoss(reduction='none')
    ae_reco_all = []
    with torch.no_grad():
        for i in range(0, obj_data.shape[0], batch_size):
            batch = obj_data[i:i + batch_size].to(device)
            recon, _ = ae_model(batch)
            ae_reco_all.append(mse(recon, batch).mean(dim=1).cpu())
    return torch.cat(ae_reco_all).float()


class HLTSmCocktailDataset(Dataset):
    """
    Args:
        pf_data:       [N, max_cands, n_feats]  PF candidate features
        labels:        [N] long
        nuisances_all: [N] binned AE reconstruction-loss nuisance
        ae_reco_all:   [N] continuous AE reconstruction loss
        idx:           selected event indices for this split
        split:      "train" | "val"
    """
    def __init__(self, pf_data, labels, nuisances_all, ae_reco_all, idx,
                 split="train", bin_edges=None, max_weight_ratio=10.0):
        super().__init__()
        self.bin_edges = bin_edges

        self.features = pf_data
        self.labels_all = labels
        self.nuisances_all = nuisances_all
        self.ae_reco_all = ae_reco_all
        self.idx = idx.long()
        self.split = split
        self.num_tokens = pf_data.size(1)

        self.labels = labels[self.idx].float()
        self.nuisances = nuisances_all[self.idx].float()
        self.ae_reco = ae_reco_all[self.idx].float()

        # ── NURD exact weights ────────────────────────────────────────────────
        labels_split = labels[self.idx]
        nuisances_split = nuisances_all[self.idx]
        self.weights = _make_nurd_weights(
            labels_split, nuisances_split, max_weight_ratio=max_weight_ratio)
        self.sample_weights = torch.tensor(
            [self.weights[(int(y.item()), int(z.item()))]
             for y, z in zip(labels_split, nuisances_split)],
            dtype=torch.float32,
        )
        _w = list(self.weights.values())
        import statistics
        _w_mean = sum(_w) / len(_w)
        _w_std = statistics.stdev(_w) if len(_w) > 1 else 0.0
        print(f"[{split}] NURD weight groups={len(_w)}  mean={_w_mean:.3f}  "
              f"std={_w_std:.3f}  min={min(_w):.3f}  max={max(_w):.3f}")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        event_idx = self.idx[idx]
        return (
            self.features[event_idx],
            self.labels[idx],
            self.nuisances[idx],
            self.ae_reco[idx],
            self.sample_weights[idx],
        )

    def get_label_prior(self):
        total = len(self.labels)
        counts = Counter(int(y) for y in self.labels.tolist())
        return {k: v / total for k, v in counts.items()}

    def get_nuisance_prior(self, label=None):
        nuisances = self.nuisances
        if label is not None:
            label_mask = self.labels.long() == int(label)
            nuisances = nuisances[label_mask]
        total = len(nuisances)
        if total == 0:
            return {}
        counts = Counter(int(z) for z in nuisances.tolist())
        return {k: v / total for k, v in counts.items()}


def build_hlt_datasets(pt_path, ae_model, n_bins=10, val_split=0.1, seed=42,
                       max_events=-1, ae_scaler=None, ae_batch_size=4096,
                       max_weight_ratio=10.0, nuisance_bin_scope="all",
                       qcd_label=1):
    """
    Load the HLT .pt file, pre-normalise obj features, and return
    (train_dataset, val_dataset).  Call once; pass the same bin_edges
    to both splits so nuisance definitions are consistent.
    """
    raw = torch.load(pt_path, map_location="cpu")
    pf     = raw["pf"]
    labels = raw["label"].long()
    obj    = raw["obj"]
    if max_events > 0:
        pf, labels, obj = pf[:max_events], labels[:max_events], obj[:max_events]
    pf = torch.nan_to_num(pf, nan=0.0, posinf=0.0, neginf=0.0)

    # flatten + z-score normalise obj features (first 4 features per cand)
    obj_flat = obj[:, :, :4].reshape(obj.shape[0], -1).float()
    if ae_scaler is None:
        mu = obj_flat.mean(dim=0)
        std = obj_flat.std(dim=0, unbiased=False)
        std = torch.where(std < 1e-8, torch.ones_like(std), std)
    else:
        mu = _as_float_tensor(ae_scaler["mu"])
        std = _as_float_tensor(ae_scaler["std"])
        if mu.numel() != obj_flat.shape[1] or std.numel() != obj_flat.shape[1]:
            raise ValueError(
                f"AE scaler has {mu.numel()} features but obj input has {obj_flat.shape[1]}"
            )
    obj_norm = (obj_flat - mu.view(1, -1)) / std.view(1, -1)
    obj_scaler = {"mu": mu.cpu(), "std": std.cpu()}
    del obj, obj_flat

    # AE reco is the nuisance definition. Compute it once, then split.
    ae_reco_all = _compute_ae_reco(obj_norm, ae_model, batch_size=ae_batch_size)
    quantiles = torch.linspace(0, 1, n_bins + 1)
    if nuisance_bin_scope == "qcd":
        bin_source = ae_reco_all[labels == int(qcd_label)]
        if bin_source.numel() == 0:
            raise ValueError(
                f"Cannot build QCD-scoped nuisance bins: no label={qcd_label} events found."
            )
    elif nuisance_bin_scope == "all":
        bin_source = ae_reco_all
    else:
        raise ValueError(f"Unsupported nuisance_bin_scope={nuisance_bin_scope!r}")
    bin_edges = torch.quantile(bin_source, quantiles)
    nuisances_all = torch.bucketize(ae_reco_all, bin_edges[1:-1]).long()
    del obj_norm

    idx_all = np.arange(len(labels))
    idx_tr, idx_val = train_test_split(
        idx_all, test_size=val_split, random_state=seed,
        stratify=labels.cpu().numpy()
    )
    idx_tr = torch.tensor(idx_tr, dtype=torch.long)
    idx_val = torch.tensor(idx_val, dtype=torch.long)

    ds_train = HLTSmCocktailDataset(
        pf, labels, nuisances_all, ae_reco_all, idx_tr,
        split="train", bin_edges=bin_edges,
        max_weight_ratio=max_weight_ratio)
    ds_val = HLTSmCocktailDataset(
        pf, labels, nuisances_all, ae_reco_all, idx_val,
        split="val", bin_edges=bin_edges,
        max_weight_ratio=max_weight_ratio)
    return ds_train, ds_val, obj_scaler
