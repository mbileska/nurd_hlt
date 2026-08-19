"""
HLT SM Cocktail dataset for NURD training.

The nuisance variable z is the **binned AE reconstruction loss**.
NURD exact weights w(y,z) = p(y)*p(z)/p(y,z) are pre-computed on load
so that train_exact.py can look them up with dataset.weights[(y,z)].

Dataset returns (pf_features, label, nuisance_bin, ae_reco, nurd_weight,
generator_weight)
per event.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict

from utils.event_weights import (
    EVENT_ID_KEYS,
    load_event_weights,
    sample_signature,
    split_diagnostics,
    weighted_mean_and_std,
    weighted_quantile,
    weighted_stratified_split,
)


def _make_nurd_weights(labels, nuisances, max_weight_ratio=10.0,
                       base_weights=None):
    """
    Exact NURD weights: w(y,z) = p(y)*p(z)/p(y,z) = n_y*n_z / (N*n_yz).
    Under this weighting, y and z are marginally independent.
    Normalized so that the per-sample mean weight equals 1, then clipped at
    max_weight_ratio × mean to prevent extreme weights from destabilising training.
    """
    n_events = len(labels)
    if n_events == 0:
        raise ValueError("Cannot fit NURD weights on an empty sample.")
    if max_weight_ratio < 1.0:
        raise ValueError("max_weight_ratio must be at least 1.0.")
    labels_list = [int(y) for y in labels.tolist()]
    nuisances_list = [int(z) for z in nuisances.tolist()]
    if base_weights is None:
        base_weights = torch.ones(n_events, dtype=torch.float64)
    base_weights = torch.as_tensor(base_weights).double().reshape(-1)
    if base_weights.numel() != n_events:
        raise ValueError("base_weights must align with labels and nuisances.")
    if (base_weights < 0).any() or not torch.isfinite(base_weights).all():
        raise ValueError("NURD base weights must be finite and non-negative.")

    group_sums = defaultdict(float)
    label_sums = defaultdict(float)
    nuisance_sums = defaultdict(float)
    for y, z, weight in zip(labels_list, nuisances_list, base_weights.tolist()):
        group_sums[(y, z)] += weight
        label_sums[y] += weight
        nuisance_sums[z] += weight
    total_weight = float(base_weights.sum().item())
    if total_weight <= 0.0:
        raise ValueError("NURD base weights have non-positive total weight.")

    weights_raw = {
        (y, z): (label_sums[y] * nuisance_sums[z])
        / (total_weight * group_weight)
        for (y, z), group_weight in group_sums.items()
        if group_weight > 0.0
    }
    # Normalize so E_gen[w_NURD] = 1 over the physical training measure.
    mean_w = sum(
        weights_raw[k] * group_sums[k] for k in weights_raw
    ) / total_weight
    weights_norm = {k: v / mean_w for k, v in weights_raw.items()}

    # Find a common scale for min(scale * w, cap) whose sample-weighted mean is
    # one. Clipping and then renormalizing directly can violate the requested cap.
    cap = float(max_weight_ratio)

    def clipped_mean(scale):
        return sum(
            min(scale * weights_norm[k], cap) * group_sums[k]
            for k in weights_norm
        ) / total_weight

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
        gen_weights:   [N] generator/event weights
        idx:           selected event indices for this split
        split:      "train" | "val"
    """
    def __init__(self, pf_data, labels, nuisances_all, ae_reco_all,
                 gen_weights, idx,
                 split="train", bin_edges=None, max_weight_ratio=10.0,
                 weight_table=None):
        super().__init__()
        self.bin_edges = bin_edges

        self.features = pf_data
        self.labels_all = labels
        self.nuisances_all = nuisances_all
        self.ae_reco_all = ae_reco_all
        self.gen_weights_all = gen_weights
        self.idx = idx.long()
        self.split = split
        self.num_tokens = pf_data.size(1)

        self.labels = labels[self.idx].float()
        self.nuisances = nuisances_all[self.idx].float()
        self.ae_reco = ae_reco_all[self.idx].float()
        self.gen_weights = gen_weights[self.idx].float()

        # ── NURD exact weights ────────────────────────────────────────────────
        labels_split = labels[self.idx]
        nuisances_split = nuisances_all[self.idx]
        if weight_table is None:
            self.weights = _make_nurd_weights(
                labels_split, nuisances_split,
                max_weight_ratio=max_weight_ratio,
                base_weights=self.gen_weights)
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
        combined = self.sample_weights * self.gen_weights
        combined_mean = combined.sum() / self.gen_weights.sum().clamp(min=1e-8)
        print(
            f"[{split}] NURD weights fit={weight_source} "
            f"groups={len(self.weights)}  "
            f"physical_mean={combined_mean.item():.3f}  "
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
            self.gen_weights[idx],
        )

    def get_label_prior(self):
        total = float(self.gen_weights.sum().item())
        counts = defaultdict(float)
        for label, weight in zip(self.labels.tolist(), self.gen_weights.tolist()):
            counts[int(label)] += float(weight)
        return {k: v / total for k, v in counts.items()}

    def get_nuisance_prior(self, label=None):
        nuisances = self.nuisances
        weights = self.gen_weights
        if label is not None:
            if isinstance(label, (list, tuple, set)):
                label_mask = _label_mask(self.labels.long(), label)
            else:
                label_mask = self.labels.long() == int(label)
            nuisances = nuisances[label_mask]
            weights = weights[label_mask]
        total = float(weights.sum().item())
        if nuisances.numel() == 0 or total <= 0.0:
            return {}
        counts = defaultdict(float)
        for nuisance, weight in zip(nuisances.tolist(), weights.tolist()):
            counts[int(nuisance)] += float(weight)
        return {k: v / total for k, v in counts.items()}


def build_hlt_datasets(pt_path, ae_model, n_bins=20, val_split=0.1, seed=42,
                       max_events=-1, ae_scaler=None, ae_batch_size=4096,
                       max_weight_ratio=10.0, nuisance_bin_scope="all",
                       qcd_label=1, baseline_labels=None,
                       gen_weight_path=None,
                       threshold_tune_fraction=0.5):
    """
    Load the HLT .pt file, pre-normalise obj features, and return
    (train_dataset, val_dataset).  Call once; pass the same bin_edges
    to both splits so nuisance definitions are consistent.
    """
    raw = torch.load(pt_path, map_location="cpu")
    gen_weights, gen_weight_metadata = load_event_weights(
        gen_weight_path, raw, max_events=max_events)
    pf     = raw["pf"]
    labels = raw["label"].long()
    obj    = raw["obj"]
    if max_events > 0:
        pf, labels, obj = pf[:max_events], labels[:max_events], obj[:max_events]
    signature_sample = {"pf": pf, "obj": obj, "label": labels}
    for key in EVENT_ID_KEYS:
        if key in raw:
            signature_sample[key] = raw[key][:len(labels)]
    data_signature = sample_signature(signature_sample)
    pf = torch.nan_to_num(pf, nan=0.0, posinf=0.0, neginf=0.0)

    if not 0.0 < float(val_split) < 1.0:
        raise ValueError("val_split must lie in (0, 1).")
    if not 0.0 < float(threshold_tune_fraction) < 1.0:
        raise ValueError("threshold_tune_fraction must lie in (0, 1).")
    tune_fraction = float(val_split) * float(threshold_tune_fraction)
    checkpoint_fraction = float(val_split) - tune_fraction
    train_fraction = 1.0 - float(val_split)
    split_values = weighted_stratified_split(
        labels, gen_weights,
        (train_fraction, checkpoint_fraction, tune_fraction), seed=seed)
    idx_tr, idx_val, idx_tune = [
        torch.as_tensor(values, dtype=torch.long) for values in split_values
    ]

    # Flatten object features and fit fallback normalization on training events
    # only. The normal path uses the scaler saved by the weighted AE checkpoint.
    obj_flat = torch.nan_to_num(
        obj[:, :, :4].reshape(obj.shape[0], -1).float(),
        nan=0.0, posinf=0.0, neginf=0.0)
    if ae_scaler is None:
        mu, std = weighted_mean_and_std(
            obj_flat[idx_tr], gen_weights[idx_tr], dim=0)
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
        bin_edges = weighted_quantile(
            bin_source, quantiles, gen_weights[idx_tr][train_qcd])
        nuisances_all = torch.bucketize(ae_reco_all, bin_edges[1:-1]).long()
    elif nuisance_bin_scope == "all":
        bin_source = ae_reco_all[idx_tr]
        bin_edges = weighted_quantile(
            bin_source, quantiles, gen_weights[idx_tr])
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
            edges = weighted_quantile(
                ae_reco_all[idx_tr][train_mask], quantiles,
                gen_weights[idx_tr][train_mask])
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
                edges = weighted_quantile(
                    ae_reco_all[idx_tr][train_mask], quantiles,
                    gen_weights[idx_tr][train_mask])
                nuisances_all[mask] = torch.bucketize(ae_reco_all[mask], edges[1:-1]).long()
                bin_edges[label] = edges
    else:
        raise ValueError(f"Unsupported nuisance_bin_scope={nuisance_bin_scope!r}")

    ds_train = HLTSmCocktailDataset(
        pf, labels, nuisances_all, ae_reco_all, gen_weights, idx_tr,
        split="train", bin_edges=bin_edges,
        max_weight_ratio=max_weight_ratio)
    ds_val = HLTSmCocktailDataset(
        pf, labels, nuisances_all, ae_reco_all, gen_weights, idx_val,
        split="val", bin_edges=bin_edges,
        max_weight_ratio=max_weight_ratio,
        weight_table=ds_train.weights)
    split_indices = {
        "reference_fit": idx_tr.cpu(),
        "checkpoint_validation": idx_val.cpu(),
        "threshold_tune": idx_tune.cpu(),
    }
    split_metadata = {
        "scheme": "generator_mass_balanced_stratified_folds_v1",
        "seed": int(seed),
        "fractions": {
            "reference_fit": train_fraction,
            "checkpoint_validation": checkpoint_fraction,
            "threshold_tune": tune_fraction,
        },
        "diagnostics": split_diagnostics(
            labels, gen_weights, split_indices),
    }
    provenance = {
        "sample": data_signature,
        "generator_weights": dict(gen_weight_metadata),
    }
    for dataset in (ds_train, ds_val):
        dataset.split_indices = split_indices
        dataset.split_metadata = split_metadata
        dataset.provenance = provenance
    return ds_train, ds_val, obj_scaler, gen_weight_metadata
