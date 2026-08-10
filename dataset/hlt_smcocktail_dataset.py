"""
HLT SM Cocktail dataset for NURD training.

The nuisance variable z is represented both by binned AE reconstruction loss
for exact NURD weights and by a continuous weighted AE-loss rank for the
adversarial critic. Both coordinates use the configured nuisance population.
NURD exact weights w(y,z) = p(y)*p(z)/p(y,z) are pre-computed on load
so that train_exact.py can look them up with dataset.weights[(y,z)].

Dataset returns (pf_features, label, nuisance_bin, nuisance_cdf, ae_reco,
nurd_weight, generator_weight)
per event.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split

from utils.event_weights import (
    load_event_weights,
    weighted_mean_and_std,
    weighted_quantile,
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

    group_counts = Counter(zip(labels_list, nuisances_list))
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


def class_balance_factors(labels, base_weights):
    """Scale physical event weights so every class has equal total mass."""
    labels = torch.as_tensor(labels).long().view(-1)
    base_weights = torch.as_tensor(base_weights).double().view(-1)
    if labels.numel() != base_weights.numel():
        raise ValueError("labels and base_weights must align.")
    classes = labels.unique(sorted=True)
    if classes.numel() < 2:
        raise ValueError("Class-balanced training requires at least two classes.")
    total = base_weights.sum()
    factors = {}
    for label in classes.tolist():
        class_mass = base_weights[labels == int(label)].sum()
        if float(class_mass.item()) <= 0.0:
            raise ValueError(f"Class {label} has non-positive physical mass.")
        factors[int(label)] = float(
            total.item() / (classes.numel() * class_mass.item()))
    return factors


def apply_class_balance(labels, base_weights, factors):
    labels = torch.as_tensor(labels).long().view(-1)
    base_weights = torch.as_tensor(base_weights).float().view(-1)
    result = torch.empty_like(base_weights)
    for label in labels.unique(sorted=True).tolist():
        if int(label) not in factors:
            raise ValueError(f"Missing class-balance factor for label={label}.")
        mask = labels == int(label)
        result[mask] = base_weights[mask] * float(factors[int(label)])
    return result


def weighted_cdf_coordinate(values, reference_values, reference_weights):
    """Map values to a stable [0, 1] weighted empirical CDF coordinate."""
    values = torch.as_tensor(values).float().view(-1)
    reference_values = torch.as_tensor(reference_values).float().view(-1)
    reference_weights = torch.as_tensor(reference_weights).double().view(-1)
    if reference_values.numel() != reference_weights.numel():
        raise ValueError("reference_values and reference_weights must align.")
    valid = (
        torch.isfinite(reference_values)
        & torch.isfinite(reference_weights)
        & (reference_weights > 0)
    )
    if not valid.any():
        raise ValueError("Weighted CDF reference has no positive finite mass.")
    order = torch.argsort(reference_values[valid])
    sorted_values = reference_values[valid][order]
    sorted_weights = reference_weights[valid][order]
    unique_values, inverse = torch.unique_consecutive(
        sorted_values, return_inverse=True)
    group_weights = torch.zeros(
        unique_values.numel(), dtype=torch.float64)
    group_weights.scatter_add_(0, inverse, sorted_weights)
    cumulative = torch.cumsum(group_weights, dim=0)
    mid_cdf = (cumulative - 0.5 * group_weights) / cumulative[-1]
    positions = torch.searchsorted(unique_values, values).clamp(
        max=unique_values.numel() - 1)
    result = mid_cdf[positions].float()
    result[values < unique_values[0]] = 0.0
    result[values > unique_values[-1]] = 1.0
    return result.clamp(0.0, 1.0)


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
        nuisance_cdf_all: [N] continuous weighted AE-loss rank
        ae_reco_all:   [N] continuous AE reconstruction loss
        gen_weights:   [N] generator/event weights
        idx:           selected event indices for this split
        split:      "train" | "val"
    """
    def __init__(self, pf_data, labels, nuisances_all, nuisance_cdf_all,
                 ae_reco_all,
                 gen_weights, idx,
                 split="train", bin_edges=None, max_weight_ratio=10.0,
                 weight_table=None, training_measure="physical",
                 balance_factors=None):
        super().__init__()
        self.bin_edges = bin_edges

        self.features = pf_data
        self.labels_all = labels
        self.nuisances_all = nuisances_all
        self.nuisance_cdf_all = nuisance_cdf_all
        self.ae_reco_all = ae_reco_all
        self.gen_weights_all = gen_weights
        self.idx = idx.long()
        self.split = split
        self.num_tokens = pf_data.size(1)

        self.labels = labels[self.idx].float()
        self.nuisances = nuisances_all[self.idx].float()
        self.nuisance_cdf = nuisance_cdf_all[self.idx].float()
        self.ae_reco = ae_reco_all[self.idx].float()
        self.gen_weights = gen_weights[self.idx].float()
        self.training_measure = str(training_measure)
        if self.training_measure not in {"physical", "class_balanced_physical"}:
            raise ValueError(
                f"Unsupported training_measure={self.training_measure!r}")
        if balance_factors is None:
            balance_factors = class_balance_factors(
                labels[self.idx], self.gen_weights)
        self.class_balance_factors = dict(balance_factors)
        self.class_balanced_weights = apply_class_balance(
            labels[self.idx], self.gen_weights, self.class_balance_factors)
        self.measure_weights = (
            self.class_balanced_weights
            if self.training_measure == "class_balanced_physical"
            else self.gen_weights
        )

        # ── NURD exact weights ────────────────────────────────────────────────
        labels_split = labels[self.idx]
        nuisances_split = nuisances_all[self.idx]
        if weight_table is None:
            self.weights = _make_nurd_weights(
                labels_split, nuisances_split,
                max_weight_ratio=max_weight_ratio,
                base_weights=self.measure_weights)
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
        combined = self.sample_weights * self.measure_weights
        combined_mean = combined.sum() / self.measure_weights.sum().clamp(min=1e-8)
        print(
            f"[{split}] NURD weights fit={weight_source} "
            f"groups={len(self.weights)}  "
            f"measure_mean={combined_mean.item():.3f}  "
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
            self.nuisance_cdf[idx],
            self.ae_reco[idx],
            self.sample_weights[idx],
            self.gen_weights[idx],
        )

    def get_label_prior(self):
        total = float(self.measure_weights.sum().item())
        counts = defaultdict(float)
        for label, weight in zip(self.labels.tolist(), self.measure_weights.tolist()):
            counts[int(label)] += float(weight)
        return {k: v / total for k, v in counts.items()}

    def get_nuisance_prior(self, label=None):
        nuisances = self.nuisances
        weights = self.measure_weights
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
                       training_measure="physical"):
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
    pf = torch.nan_to_num(pf, nan=0.0, posinf=0.0, neginf=0.0)

    idx_all = np.arange(len(labels))
    idx_tr, idx_val = train_test_split(
        idx_all, test_size=val_split, random_state=seed,
        stratify=labels.cpu().numpy()
    )
    idx_tr = torch.tensor(idx_tr, dtype=torch.long)
    idx_val = torch.tensor(idx_val, dtype=torch.long)

    # Flatten object features and fit fallback normalization on training events
    # only. The normal path uses the scaler saved by the weighted AE checkpoint.
    obj_flat = obj[:, :, :4].reshape(obj.shape[0], -1).float()
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

    # The continuous adversary removes the finite-bin resolution ceiling. Its
    # reference population must match the discrete nuisance definition.
    if nuisance_bin_scope == "qcd":
        rank_mask = labels[idx_tr] == int(qcd_label)
        if not rank_mask.any():
            raise ValueError(
                f"Cannot build continuous QCD nuisance: no label={qcd_label} events.")
        nuisance_cdf_all = weighted_cdf_coordinate(
            ae_reco_all, ae_reco_all[idx_tr][rank_mask],
            gen_weights[idx_tr][rank_mask])
    elif nuisance_bin_scope == "all":
        nuisance_cdf_all = weighted_cdf_coordinate(
            ae_reco_all, ae_reco_all[idx_tr], gen_weights[idx_tr])
    else:
        nuisance_cdf_all = torch.empty_like(ae_reco_all)
        for label in sorted(int(v) for v in torch.unique(labels).tolist()):
            train_mask = labels[idx_tr] == label
            full_mask = labels == label
            nuisance_cdf_all[full_mask] = weighted_cdf_coordinate(
                ae_reco_all[full_mask], ae_reco_all[idx_tr][train_mask],
                gen_weights[idx_tr][train_mask])
    balance_factors = class_balance_factors(
        labels[idx_tr], gen_weights[idx_tr])

    ds_train = HLTSmCocktailDataset(
        pf, labels, nuisances_all, nuisance_cdf_all, ae_reco_all,
        gen_weights, idx_tr,
        split="train", bin_edges=bin_edges,
        max_weight_ratio=max_weight_ratio,
        training_measure=training_measure,
        balance_factors=balance_factors)
    ds_val = HLTSmCocktailDataset(
        pf, labels, nuisances_all, nuisance_cdf_all, ae_reco_all,
        gen_weights, idx_val,
        split="val", bin_edges=bin_edges,
        max_weight_ratio=max_weight_ratio,
        weight_table=ds_train.weights,
        training_measure=training_measure,
        balance_factors=balance_factors)
    return ds_train, ds_val, obj_scaler, gen_weight_metadata
