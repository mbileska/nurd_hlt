"""HLT SM-cocktail datasets for continuous-nuisance NURD training.

The critic receives the continuous, standardized AE reconstruction error. A
training-only histogram of that continuous value is used solely to estimate
engineer-style class/nuisance balancing weights.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import torch
from torch.utils.data import Dataset

from utils.hlt_weights import (
    apply_class_balance_factors,
    apply_joint_balance,
    effective_mass_by_class,
    effective_sample_size_fraction,
    fit_class_balance_factors,
    fit_joint_balance,
    load_generator_weights,
    sample_signature,
    stratified_split_indices,
    weighted_mean_and_std,
)


def _as_float_tensor(value) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32).detach().cpu()


def _compute_ae_reconstruction_error(
    obj_normalized: torch.Tensor,
    ae_model,
    batch_size: int,
) -> torch.Tensor:
    ae_model.eval()
    device = next(ae_model.parameters()).device
    scores = []
    with torch.no_grad():
        for start in range(0, obj_normalized.shape[0], int(batch_size)):
            batch = obj_normalized[start:start + int(batch_size)].to(device)
            reconstruction, _ = ae_model(batch)
            scores.append((reconstruction - batch).square().mean(dim=1).cpu())
    return torch.cat(scores).float()


class HLTSmCocktailDataset(Dataset):
    """A view over shared event tensors and precomputed training quantities."""

    def __init__(
        self,
        pf_data: torch.Tensor,
        labels: torch.Tensor,
        nuisance: torch.Tensor,
        ae_reco: torch.Tensor,
        effective_weights: torch.Tensor,
        physics_weights: torch.Tensor,
        indices: torch.Tensor,
        split: str,
    ):
        self.features_all = pf_data
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        self.labels = labels[self.indices].long()
        self.nuisance = nuisance[self.indices].float()
        self.ae_reco = ae_reco[self.indices].float()
        self.effective_weights = effective_weights.float()
        self.physics_weights = physics_weights[self.indices].float()
        self.split = str(split)
        self.num_tokens = int(pf_data.shape[1])

        expected = self.indices.numel()
        for name, value in (
            ("effective_weights", self.effective_weights),
            ("physics_weights", self.physics_weights),
        ):
            if value.numel() != expected:
                raise ValueError(
                    f"{name} has {value.numel()} entries for {expected} events.")

    def __len__(self):
        return self.indices.numel()

    def __getitem__(self, item):
        event_index = self.indices[item]
        return (
            self.features_all[event_index],
            self.labels[item],
            self.nuisance[item],
            self.ae_reco[item],
            self.effective_weights[item],
            self.physics_weights[item],
        )


def build_hlt_datasets(
    pt_path: str,
    ae_model,
    val_split: float = 0.1,
    seed: int = 42,
    max_events: int = -1,
    exclude_labels: Optional[Sequence[int]] = None,
    gen_weight_path: Optional[str] = None,
    qcd_label: int = 1,
    ae_scaler: Optional[Mapping[str, torch.Tensor]] = None,
    balance_strata: int = 20,
    ae_batch_size: int = 4096,
):
    """Build leakage-free train/validation datasets.

    Returns ``(train, validation, preprocessing)``. All fitted quantities in
    ``preprocessing`` come from the training split and are saved in the model
    checkpoint for exact evaluation reuse.
    """
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    for key in ("pf", "obj", "label"):
        if key not in raw:
            raise KeyError(f"Training file is missing required key {key!r}.")

    data_signature = sample_signature(raw, max_events=max_events)
    pf = raw["pf"]
    obj = raw["obj"]
    labels_original = raw["label"].long().reshape(-1)
    if not (pf.shape[0] == obj.shape[0] == labels_original.numel()):
        raise ValueError("PF, object, and label arrays do not have equal lengths.")

    if max_events > 0:
        pf = pf[:max_events]
        obj = obj[:max_events]
        labels_original = labels_original[:max_events]
    physics_weights, generator_metadata = load_generator_weights(
        gen_weight_path,
        labels_original,
        qcd_label=qcd_label,
        max_events=max_events,
        sample=raw,
    )
    del raw

    if exclude_labels:
        keep = torch.ones(labels_original.numel(), dtype=torch.bool)
        for label in exclude_labels:
            keep &= labels_original != int(label)
        pf, obj = pf[keep], obj[keep]
        physics_weights = physics_weights[keep]
        labels_original = labels_original[keep]

    unique_labels = sorted(int(value) for value in labels_original.unique().tolist())
    label_map = {old: new for new, old in enumerate(unique_labels)}
    if int(qcd_label) not in label_map:
        raise ValueError(f"QCD label {qcd_label} was excluded or is absent.")
    labels = torch.empty_like(labels_original)
    for old_label, new_label in label_map.items():
        labels[labels_original == old_label] = new_label
    remapped_qcd_label = label_map[int(qcd_label)]

    pf = torch.nan_to_num(pf.float(), nan=0.0, posinf=0.0, neginf=0.0)
    obj_flat = torch.nan_to_num(
        obj[:, :, :4].reshape(obj.shape[0], -1).float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    del obj

    train_indices, val_indices = stratified_split_indices(
        labels, val_fraction=val_split, seed=seed,
        weights=physics_weights)

    class_factors = fit_class_balance_factors(
        labels[train_indices], physics_weights[train_indices])
    scaler_weights = apply_class_balance_factors(
        labels[train_indices],
        physics_weights[train_indices],
        class_factors,
        normalize_mean=False,
    )
    if ae_scaler is None:
        obj_mean, obj_std = weighted_mean_and_std(
            obj_flat[train_indices], scaler_weights)
        obj_std = torch.where(obj_std < 1e-8, torch.ones_like(obj_std), obj_std)
        scaler_source = "training_split_class_balanced_physics"
    else:
        obj_mean = _as_float_tensor(ae_scaler["mu"]).reshape(-1)
        obj_std = _as_float_tensor(ae_scaler["std"]).reshape(-1)
        if obj_mean.numel() != obj_flat.shape[1] or obj_std.numel() != obj_flat.shape[1]:
            raise ValueError(
                "AE checkpoint scaler is incompatible with the object feature shape.")
        obj_std = torch.where(obj_std < 1e-8, torch.ones_like(obj_std), obj_std)
        scaler_source = "ae_checkpoint"

    obj_normalized = (obj_flat - obj_mean.view(1, -1)) / obj_std.view(1, -1)
    del obj_flat
    ae_reco = _compute_ae_reconstruction_error(
        obj_normalized, ae_model, batch_size=ae_batch_size)
    del obj_normalized

    nuisance_mean, nuisance_std = weighted_mean_and_std(
        ae_reco[train_indices].view(-1, 1), scaler_weights)
    nuisance_mean = nuisance_mean.reshape(())
    nuisance_std = nuisance_std.reshape(()).clamp(min=1e-8)
    nuisance = (ae_reco - nuisance_mean) / nuisance_std

    train_effective_weights, balance_spec = fit_joint_balance(
        labels[train_indices],
        ae_reco[train_indices],
        physics_weights[train_indices],
        n_strata=balance_strata,
    )
    val_effective_weights = apply_joint_balance(
        labels[val_indices],
        ae_reco[val_indices],
        physics_weights[val_indices],
        balance_spec,
    )

    train_dataset = HLTSmCocktailDataset(
        pf, labels, nuisance, ae_reco,
        train_effective_weights, physics_weights,
        train_indices, split="train")
    val_dataset = HLTSmCocktailDataset(
        pf, labels, nuisance, ae_reco,
        val_effective_weights, physics_weights,
        val_indices, split="validation")

    train_mass = effective_mass_by_class(
        train_dataset.labels, train_dataset.effective_weights)
    val_mass = effective_mass_by_class(
        val_dataset.labels, val_dataset.effective_weights)
    train_ess = effective_sample_size_fraction(train_effective_weights)
    val_ess = effective_sample_size_fraction(val_effective_weights)
    print(
        "Unified training weights: "
        f"class_mass={train_mass} ESS/N={train_ess:.4f}",
        flush=True,
    )
    print(
        "Unified validation weights (training fit): "
        f"class_mass={val_mass} ESS/N={val_ess:.4f}",
        flush=True,
    )

    preprocessing: Dict[str, object] = {
        "ae_scaler": {"mu": obj_mean.cpu(), "std": obj_std.cpu()},
        "ae_scaler_source": scaler_source,
        "nuisance_transform": {
            "kind": "standardize_continuous_ae_reconstruction_error",
            "mean": nuisance_mean.cpu(),
            "std": nuisance_std.cpu(),
        },
        "weighting": {
            "method": "generator_weighted_uniform_class_and_nuisance_strata",
            "balance_spec": balance_spec,
            "generator": generator_metadata,
            "train_class_mass": train_mass,
            "validation_class_mass": val_mass,
            "train_ess_fraction": train_ess,
            "validation_ess_fraction": val_ess,
        },
        "split": {
            "seed": int(seed),
            "validation_fraction": float(val_split),
            "train_indices": train_indices.cpu(),
            "validation_indices": val_indices.cpu(),
        },
        "label_map": label_map,
        "qcd_label": int(remapped_qcd_label),
        "data_signature": data_signature,
    }
    return train_dataset, val_dataset, preprocessing
