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
    if N == 0:
        raise ValueError("Cannot fit NURD weights on an empty sample.")
    if max_weight_ratio < 1.0:
        raise ValueError("max_weight_ratio must be at least 1.0.")
    labels_list    = [int(y) for y in labels.tolist()]
    nuisances_list = [int(z) for z in nuisances.tolist()]

    group_counts    = Counter(zip(labels_list, nuisances_list))
    label_counts    = Counter(labels_list)
    nuisance_counts = Counter(nuisances_list)

    weights_raw = {
        (y, z): (label_counts[y] * nuisance_counts[z]) / (N * n_yz)
        for (y, z), n_yz in group_counts.items()
    }
    # Normalize so E[w] = 1 over all training samples.
    mean_w = sum(weights_raw[k] * v for k, v in group_counts.items()) / N
    weights_norm = {k: v / mean_w for k, v in weights_raw.items()}

    # Find a common scale for min(scale * w, cap) whose sample-weighted mean is
    # one. Clipping and then renormalizing directly can violate the requested cap.
    cap = float(max_weight_ratio)

    def clipped_mean(scale):
        return sum(
            min(scale * weights_norm[k], cap) * count
            for k, count in group_counts.items()
        ) / N

    low, high = 0.0, 1.0
    while clipped_mean(high) < 1.0:
        high *= 2.0
    for _ in range(64):
        mid = 0.5 * (low + high)
        if clipped_mean(mid) < 1.0:
            low = mid
        else:
            high = mid
    return {k: min(high * value, cap) for k, value in weights_norm.items()}


def _label_mask(labels, label_values):
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for label in label_values:
        mask |= labels == int(label)
    return mask


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
                 split="train", bin_edges=None, max_weight_ratio=10.0,
                 weight_table=None):
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
        if weight_table is None:
            self.weights = _make_nurd_weights(
                labels_split, nuisances_split,
                max_weight_ratio=max_weight_ratio)
            weight_source = split
        else:
            self.weights = dict(weight_table)
            weight_source = "train"

        split_groups = {
            (int(pair[0]), int(pair[1]))
            for pair in torch.unique(
                torch.stack([labels_split, nuisances_split], dim=1), dim=0
            ).tolist()
        }
        missing_groups = sorted(split_groups.difference(self.weights))
        if missing_groups:
            print(
                f"[{split}] WARNING: {len(missing_groups)} label/nuisance groups "
                f"were absent from the training weight fit; using the cap."
            )
        max_label = max(
            int(labels_split.max().item()),
            max(label for label, _ in self.weights),
        )
        max_nuisance = max(
            int(nuisances_split.max().item()),
            max(nuisance for _, nuisance in self.weights),
        )
        weight_lookup = torch.full(
            (max_label + 1, max_nuisance + 1),
            float(max_weight_ratio), dtype=torch.float32)
        for (label, nuisance), value in self.weights.items():
            weight_lookup[label, nuisance] = float(value)
        self.sample_weights = weight_lookup[labels_split, nuisances_split]
        print(
            f"[{split}] NURD weights fit={weight_source} "
            f"groups={len(self.weights)}  "
            f"sample_mean={self.sample_weights.mean().item():.3f}  "
            f"sample_std={self.sample_weights.std(unbiased=False).item():.3f}  "
            f"min={self.sample_weights.min().item():.3f}  "
            f"max={self.sample_weights.max().item():.3f}"
        )

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
            if isinstance(label, (list, tuple, set)):
                label_mask = _label_mask(self.labels.long(), label)
            else:
                label_mask = self.labels.long() == int(label)
            nuisances = nuisances[label_mask]
        total = len(nuisances)
        if total == 0:
            return {}
        counts = Counter(int(z) for z in nuisances.tolist())
        return {k: v / total for k, v in counts.items()}


def build_hlt_datasets(pt_path, ae_model, n_bins=20, val_split=0.1, seed=42,
                       max_events=-1, ae_scaler=None, ae_batch_size=4096,
                       max_weight_ratio=10.0, nuisance_bin_scope="all",
                       qcd_label=1, baseline_labels=None):
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

    # AE reco is the nuisance definition. Compute it once, then fit all nuisance
    # preprocessing on the training split only.
    ae_reco_all = _compute_ae_reco(obj_norm, ae_model, batch_size=ae_batch_size)
    del obj_norm

    idx_all = np.arange(len(labels))
    idx_tr, idx_val = train_test_split(
        idx_all, test_size=val_split, random_state=seed,
        stratify=labels.cpu().numpy()
    )
    idx_tr = torch.tensor(idx_tr, dtype=torch.long)
    idx_val = torch.tensor(idx_val, dtype=torch.long)

    quantiles = torch.linspace(0, 1, n_bins + 1)
    if baseline_labels is None:
        baseline_labels = sorted(int(v) for v in torch.unique(labels).tolist())

    if nuisance_bin_scope == "qcd":
        train_qcd = labels[idx_tr] == int(qcd_label)
        bin_source = ae_reco_all[idx_tr][train_qcd]
        if bin_source.numel() == 0:
            raise ValueError(
                f"Cannot build QCD-scoped nuisance bins: no training "
                f"label={qcd_label} events found."
            )
        bin_edges = torch.quantile(bin_source, quantiles)
        nuisances_all = torch.bucketize(ae_reco_all, bin_edges[1:-1]).long()
    elif nuisance_bin_scope == "all":
        bin_source = ae_reco_all[idx_tr]
        bin_edges = torch.quantile(bin_source, quantiles)
        nuisances_all = torch.bucketize(ae_reco_all, bin_edges[1:-1]).long()
    elif nuisance_bin_scope in {"per_class", "per_label", "baseline_per_class"}:
        nuisances_all = torch.zeros_like(labels, dtype=torch.long)
        bin_edges = {}
        assigned = torch.zeros_like(labels, dtype=torch.bool)
        for label in baseline_labels:
            label = int(label)
            train_mask = labels[idx_tr] == label
            if train_mask.sum() == 0:
                continue
            edges = torch.quantile(ae_reco_all[idx_tr][train_mask], quantiles)
            mask = labels == label
            nuisances_all[mask] = torch.bucketize(ae_reco_all[mask], edges[1:-1]).long()
            bin_edges[label] = edges
            assigned |= mask
        if not assigned.all():
            remaining = ~assigned
            for label in sorted(int(v) for v in torch.unique(labels[remaining]).tolist()):
                train_mask = labels[idx_tr] == label
                if train_mask.sum() == 0:
                    raise ValueError(
                        f"Cannot define nuisance bins for label={label}: "
                        "the class is absent from the training split."
                    )
                mask = labels == label
                edges = torch.quantile(ae_reco_all[idx_tr][train_mask], quantiles)
                nuisances_all[mask] = torch.bucketize(ae_reco_all[mask], edges[1:-1]).long()
                bin_edges[label] = edges
    else:
        raise ValueError(f"Unsupported nuisance_bin_scope={nuisance_bin_scope!r}")

    ds_train = HLTSmCocktailDataset(
        pf, labels, nuisances_all, ae_reco_all, idx_tr,
        split="train", bin_edges=bin_edges,
        max_weight_ratio=max_weight_ratio)
    ds_val = HLTSmCocktailDataset(
        pf, labels, nuisances_all, ae_reco_all, idx_val,
        split="val", bin_edges=bin_edges,
        max_weight_ratio=max_weight_ratio,
        weight_table=ds_train.weights)
    return ds_train, ds_val, obj_scaler
