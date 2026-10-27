"""Train the HLT classifier/representation with continuous-nuisance NURD.

This implementation follows the engineer density-ratio method:

* the critic sees continuous AE reconstruction error, never a bin index;
* balancing strata are used only to estimate event weights;
* the critic distinguishes real tuples from globally shuffled nuisance tuples
  by default, with weighted within-class resampling retained as an option;
* the encoder minimizes the critic log density ratio on real tuples; and
* CE, critic, encoder-information, and optional SupCon losses use one common
  class-balanced, generator-aware event weight.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from collections import defaultdict
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.hlt_smcocktail_dataset import build_hlt_datasets
from models.hlt_autoencoder import HLTAutoencoder
from models.hlt_con import HLTContrastiveModel, HLTCritic
from utils.hlt_density_ratio import (
    critic_context_only_accuracy,
    density_ratio_critic_loss,
    engineer_information_penalty,
    frozen_parameters,
    sample_nuisance_donor_indices,
    weighted_mean,
)
from utils.hlt_weights import file_sha256
from utils.hlt_targets import TARGET_SCHEMES


class SupConLoss(nn.Module):
    """Supervised contrastive loss with effective anchor weights."""

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = float(temperature)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        features = F.normalize(features.float(), dim=1)
        labels = labels.reshape(-1, 1)
        batch_size = features.shape[0]
        positive_mask = (labels == labels.T).float()
        diagonal_mask = 1.0 - torch.eye(
            batch_size, device=features.device, dtype=features.dtype)
        positive_mask = positive_mask * diagonal_mask

        logits = features @ features.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * diagonal_mask
        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-12))
        positive_count = positive_mask.sum(dim=1)
        valid = positive_count > 0
        if not valid.any():
            return features.sum() * 0.0
        per_anchor = torch.zeros(batch_size, device=features.device)
        per_anchor[valid] = -(
            positive_mask[valid] * log_prob[valid]).sum(dim=1) / positive_count[valid]
        # Effective dataset weights have global mean one. Preserve that fixed
        # normalizer instead of dividing by a noisy random-batch weight sum.
        return (per_anchor * weights.to(per_anchor.dtype)).mean()


class EpochMetrics:
    def __init__(self, num_classes: int):
        self.num_classes = int(num_classes)
        self.events = 0
        self.weight_sum = 0.0
        self.weighted_ce_sum = 0.0
        self.weighted_correct_sum = 0.0
        self.correct_by_class = defaultdict(int)
        self.count_by_class = defaultdict(int)
        self.total_sum = 0.0
        self.info_sum = 0.0
        self.contrast_sum = 0.0
        self.critic_loss_sum = 0.0
        self.critic_accuracy_sum = 0.0
        self.critic_context_accuracy_sum = 0.0
        self.critic_updates = 0

    def update_main(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
        ce: torch.Tensor,
        total: torch.Tensor,
        info: torch.Tensor,
        contrast: torch.Tensor,
    ):
        predictions = logits.argmax(dim=1)
        batch_size = labels.numel()
        self.events += batch_size
        self.weight_sum += float(weights.sum())
        self.weighted_ce_sum += float((ce * weights).sum())
        self.weighted_correct_sum += float(
            ((predictions == labels).float() * weights).sum())
        self.total_sum += float(total.detach()) * batch_size
        self.info_sum += float(info.detach()) * batch_size
        self.contrast_sum += float(contrast.detach()) * batch_size
        for label in labels.unique().tolist():
            label = int(label)
            mask = labels == label
            self.correct_by_class[label] += int((predictions[mask] == labels[mask]).sum())
            self.count_by_class[label] += int(mask.sum())

    def update_critic(
        self,
        loss: torch.Tensor,
        accuracy: torch.Tensor,
        context_accuracy: torch.Tensor,
    ):
        self.critic_loss_sum += float(loss.detach())
        self.critic_accuracy_sum += float(accuracy.detach())
        self.critic_context_accuracy_sum += float(context_accuracy.detach())
        self.critic_updates += 1

    def summary(self) -> Dict[str, object]:
        per_class = {
            label: self.correct_by_class[label] / max(self.count_by_class[label], 1)
            for label in range(self.num_classes)
        }
        return {
            "weighted_ce": self.weighted_ce_sum / max(self.weight_sum, 1e-12),
            "weighted_accuracy": self.weighted_correct_sum / max(self.weight_sum, 1e-12),
            "balanced_accuracy": sum(per_class.values()) / max(len(per_class), 1),
            "per_class_accuracy": per_class,
            "total_loss": self.total_sum / max(self.events, 1),
            "information_penalty": self.info_sum / max(self.events, 1),
            "contrastive_loss": self.contrast_sum / max(self.events, 1),
            "critic_loss": self.critic_loss_sum / max(self.critic_updates, 1),
            "critic_accuracy": self.critic_accuracy_sum / max(self.critic_updates, 1),
            "critic_context_only_accuracy": (
                self.critic_context_accuracy_sum / max(self.critic_updates, 1)),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HLT continuous-nuisance density-ratio NURD training")
    parser.add_argument("--data", required=True)
    parser.add_argument("--ae_ckpt", required=True)
    parser.add_argument("--gen_weight_path", "--gen_weights", dest="gen_weight_path")
    parser.add_argument(
        "--generator_weight_scope",
        choices=("auto", "all_events", "single_label"),
        default="auto",
        help=("auto honors explicit payload scope and treats bare legacy tensors "
              "as single-label weights."),
    )
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--balance_strata", type=int, default=20,
                        help="Training-only histogram strata used to estimate weights; "
                             "never passed to the critic.")
    parser.add_argument(
        "--balance_binning",
        choices=("log_fixed", "weighted_quantile"),
        default="log_fixed",
        help=("Nuisance-weight binning. log_fixed uses fixed-width bins in a "
              "training-fitted scaled log1p(AE loss) coordinate."),
    )
    parser.add_argument(
        "--balance_clip_quantile", type=float, default=0.995,
        help=("Per-class quantile cap for effective NURD event weights. "
              "Use 1.0 to disable clipping."),
    )
    parser.add_argument("--qcd_label", type=int, default=1)
    parser.add_argument(
        "--target_scheme", choices=TARGET_SCHEMES, default="parent4",
        help=("Classifier targets: parent4 preserves the historical four-parent "
              "baseline; decay8 splits W and Z by generator decay channel."),
    )
    parser.add_argument("--exclude_labels", type=int, nargs="+", default=None)
    parser.add_argument("--max_events", type=int, default=-1)
    parser.add_argument("--ae_batch_size", type=int, default=4096)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", "-b", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--critic_weight_decay", type=float, default=0.0)
    parser.add_argument("--critic_steps", type=int, default=1,
                        help="Independent critic batches trained before each encoder batch.")
    parser.add_argument(
        "--nurd_enabled", type=int, choices=(0, 1), default=0,
        help=("Master NURD switch. Zero disables both critic optimization and "
              "the encoder information penalty."),
    )
    parser.add_argument(
        "--critic_start_epoch", type=int, default=0,
        help=("One-based epoch for sharp critic/NURD activation. Zero starts "
              "immediately in epoch 1. Ignored when --nurd_enabled=0."),
    )
    parser.add_argument(
        "--critic_warmup_epochs", type=int, default=0,
        help=("Number of critic-active epochs with zero encoder information "
              "penalty. This lets a fresh critic learn before it supplies a "
              "gradient to a warm-started classifier."),
    )
    parser.add_argument(
        "--lambda_info_ramp_epochs", type=int, default=0,
        help=("Linearly ramp the encoder information coefficient from zero to "
              "--lambda_info over this many epochs after critic warm-up. Zero "
              "preserves the historical sharp switch."),
    )
    parser.add_argument(
        "--selection_after_ramp", type=int, choices=(0, 1), default=0,
        help=("When enabled, start best-checkpoint selection only after the "
              "critic warm-up and lambda ramp reach their final value."),
    )
    parser.add_argument(
        "--critic_shuffle_mode",
        choices=("weighted_within_class", "global"),
        default="global",
        help=("Fake-sample construction. global is the engineer-reference "
              "shuffle; weighted_within_class targets conditional independence "
              "under the effective event measure."))
    parser.add_argument("--lambda_info", "--_lambda", dest="lambda_info",
                        type=float, default=1.0)
    parser.add_argument(
        "--warm_start_ckpt", default="",
        help=("Optional classifier checkpoint used to initialize only the HLT "
              "model. Optimizers and the critic are deliberately initialized "
              "fresh. The source checkpoint must be a NURD-inactive baseline "
              "with the same data, AE, targets, split, and weighting setup."),
    )
    parser.add_argument("--contrast_weight", type=float, default=0.0)
    parser.add_argument("--contrast_temp", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--checkpoint_every", type=int, default=0,
        help=("Save a loadable epoch snapshot every N epochs; zero disables "
              "periodic snapshots. A final-state snapshot is also saved when enabled."))
    parser.add_argument("--cosine", type=int, default=1)
    parser.add_argument(
        "--lr_schedule_epochs", type=int, default=0,
        help=("Cosine-decay horizon. Zero uses --epochs. When shorter than "
              "--epochs, the learning rate remains at eta_min afterward."))

    parser.add_argument("--embed_size", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=6)
    parser.add_argument("--proj_dim", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_ff", type=int, default=512)
    parser.add_argument("--linear_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--exp_name", default="hlt_nurd_engineer_continuous")
    parser.add_argument("--project_name", default="hlt")
    parser.add_argument("--local_testing", type=int, default=0)
    parser.add_argument("--manualSeed", type=int, default=42)
    parser.add_argument("--code_commit", default="")
    return parser


def nurd_active_for_epoch(
    epoch: int,
    nurd_enabled: int,
    critic_start_epoch: int,
) -> bool:
    """Return whether critic training and NURD pressure are active."""
    if epoch < 1:
        raise ValueError("Epochs are one-based and must be positive.")
    if int(nurd_enabled) not in (0, 1):
        raise ValueError("nurd_enabled must be zero or one.")
    if critic_start_epoch < 0:
        raise ValueError("critic_start_epoch must be non-negative.")
    if not int(nurd_enabled):
        return False
    return critic_start_epoch == 0 or epoch >= critic_start_epoch


def checkpoint_selection_start_epoch(
    nurd_enabled: int,
    critic_start_epoch: int,
    critic_warmup_epochs: int = 0,
    lambda_info_ramp_epochs: int = 0,
    selection_after_ramp: int = 0,
) -> int:
    """Return the first continuation epoch eligible for best-model selection."""
    if int(nurd_enabled) not in (0, 1):
        raise ValueError("nurd_enabled must be zero or one.")
    if critic_start_epoch < 0:
        raise ValueError("critic_start_epoch must be non-negative.")
    if critic_warmup_epochs < 0 or lambda_info_ramp_epochs < 0:
        raise ValueError("NURD warm-up and ramp epochs must be non-negative.")
    if int(selection_after_ramp) not in (0, 1):
        raise ValueError("selection_after_ramp must be zero or one.")
    start = 1 if critic_start_epoch == 0 else critic_start_epoch
    if not int(nurd_enabled) or not int(selection_after_ramp):
        return 1 if not int(nurd_enabled) else start
    if lambda_info_ramp_epochs:
        return start + critic_warmup_epochs + lambda_info_ramp_epochs - 1
    return start + critic_warmup_epochs


def information_weight_for_epoch(
    epoch: int,
    nurd_enabled: int,
    critic_start_epoch: int,
    critic_warmup_epochs: int,
    lambda_info_ramp_epochs: int,
    maximum_weight: float,
) -> float:
    """Return the scheduled encoder information coefficient for one epoch."""
    if maximum_weight < 0.0:
        raise ValueError("maximum information weight must be non-negative.")
    if critic_warmup_epochs < 0 or lambda_info_ramp_epochs < 0:
        raise ValueError("NURD warm-up and ramp epochs must be non-negative.")
    if not nurd_active_for_epoch(epoch, nurd_enabled, critic_start_epoch):
        return 0.0
    start = critic_start_epoch or 1
    active_index = epoch - start
    if active_index < critic_warmup_epochs:
        return 0.0
    if lambda_info_ramp_epochs == 0:
        return float(maximum_weight)
    ramp_step = active_index - critic_warmup_epochs + 1
    fraction = min(max(ramp_step / float(lambda_info_ramp_epochs), 0.0), 1.0)
    return float(maximum_weight) * fraction


def cosine_lr_multiplier(
    completed_epochs: int,
    schedule_epochs: int,
    eta_min_ratio: float = 1e-3,
) -> float:
    """Cosine decay that stays at its minimum after the chosen horizon."""
    if schedule_epochs < 1:
        raise ValueError("The LR schedule horizon must be positive.")
    progress = min(max(completed_epochs / float(schedule_epochs), 0.0), 1.0)
    return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * progress))


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_ae_checkpoint(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "ae" not in checkpoint:
        raise KeyError("AE checkpoint is missing the 'ae' state dictionary.")
    if "ae_scaler" not in checkpoint:
        raise KeyError(
            "AE checkpoint is missing ae_scaler; retrain it with the corrected train_ae.py.")
    config = checkpoint.get("ae_config")
    if config is None:
        raise KeyError("AE checkpoint is missing ae_config.")
    model = HLTAutoencoder(config).to(device)
    model.load_state_dict(checkpoint["ae"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def validate_ae_training_contract(ae_checkpoint, preprocessing):
    """Require identical rows and generator weights across AE and NURD."""
    expected_signature = ae_checkpoint.get("data_signature")
    actual_signature = preprocessing.get("data_signature")
    if expected_signature is None or expected_signature != actual_signature:
        raise ValueError(
            "AE checkpoint data signature does not match --data. Retrain the "
            "AE with this corrected workflow and the same sample.")
    expected_generator = ae_checkpoint.get("weighting", {}).get("generator", {})
    actual_generator = preprocessing.get("weighting", {}).get("generator", {})
    expected_checksum = expected_generator.get("effective_physics_weight_sha256")
    actual_checksum = actual_generator.get("effective_physics_weight_sha256")
    if not expected_checksum or expected_checksum != actual_checksum:
        raise ValueError(
            "AE checkpoint generator weights do not match --gen_weight_path. "
            "The AE and NURD stages must use the same aligned physics weights.")


def _tensor_equal(left, right) -> bool:
    return torch.equal(torch.as_tensor(left).cpu(), torch.as_tensor(right).cpu())


def load_warm_start_model(
    model: nn.Module,
    path: str,
    args,
    preprocessing,
    ae_checkpoint_sha256: str,
):
    """Load a compatible NURD-inactive classifier and return its provenance."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = ("state_dict_model", "preprocessing", "config", "epoch")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(f"Warm-start checkpoint is missing keys: {missing}")
    if bool(checkpoint.get("nurd_active", False)):
        raise ValueError(
            "Warm-start checkpoint has NURD active. Use a classifier-only "
            "baseline so the continuation starts from an auditable reference.")

    source = checkpoint["preprocessing"]
    if source.get("target_scheme", "parent4") != args.target_scheme:
        raise ValueError("Warm-start target scheme does not match --target_scheme.")
    if source.get("data_signature") != preprocessing.get("data_signature"):
        raise ValueError("Warm-start data signature does not match --data.")
    if source.get("label_map") != preprocessing.get("label_map"):
        raise ValueError("Warm-start label map does not match this run.")
    if checkpoint.get("ae_checkpoint_sha256") != ae_checkpoint_sha256:
        raise ValueError("Warm-start checkpoint was trained with a different AE.")

    source_split = source.get("split", {})
    current_split = preprocessing.get("split", {})
    for key in ("train_indices", "validation_indices"):
        if key not in source_split or key not in current_split:
            raise ValueError(f"Warm-start preprocessing is missing split field {key}.")
        if not _tensor_equal(source_split[key], current_split[key]):
            raise ValueError(f"Warm-start {key} do not match this run.")

    source_generator = source.get("weighting", {}).get("generator", {})
    current_generator = preprocessing.get("weighting", {}).get("generator", {})
    checksum_key = "effective_physics_weight_sha256"
    if (not source_generator.get(checksum_key)
            or source_generator.get(checksum_key)
            != current_generator.get(checksum_key)):
        raise ValueError("Warm-start generator weights do not match this run.")

    source_config = checkpoint["config"]
    for key in (
        "embed_size", "latent_dim", "proj_dim", "num_heads", "num_layers",
        "dim_ff", "linear_dim", "dropout", "contrast_weight", "contrast_temp",
        "weight_decay", "target_scheme", "qcd_label",
        "balance_strata", "balance_binning", "balance_clip_quantile",
    ):
        if source_config.get(key) != getattr(args, key):
            raise ValueError(
                f"Warm-start configuration mismatch for {key}: "
                f"source={source_config.get(key)!r}, "
                f"current={getattr(args, key)!r}.")

    model.load_state_dict(checkpoint["state_dict_model"], strict=True)
    return {
        "path": os.path.abspath(path),
        "sha256": file_sha256(path),
        "source_epoch": int(checkpoint["epoch"]),
        "source_checkpoint_role": checkpoint.get("checkpoint_role", "unknown"),
        "source_nurd_active": bool(checkpoint.get("nurd_active", False)),
        "source_validation_metrics": checkpoint.get("validation_metrics", {}),
    }


def _move_batch(batch: Iterable[torch.Tensor], device: torch.device):
    inputs, labels, nuisance, ae_reco, weights, physics_weights = batch
    return (
        inputs.to(device, non_blocking=True),
        labels.long().to(device, non_blocking=True),
        nuisance.float().to(device, non_blocking=True),
        ae_reco.float().to(device, non_blocking=True),
        weights.float().to(device, non_blocking=True),
        physics_weights.float().to(device, non_blocking=True),
    )


def critic_update(
    model,
    critic,
    optimizer,
    batch,
    device: torch.device,
    shuffle_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs, labels, nuisance, _ae_reco, weights, _physics = _move_batch(batch, device)
    model.eval()
    critic.train()
    for parameter in critic.parameters():
        parameter.requires_grad_(True)
    with torch.no_grad():
        latent, _ = model(inputs)
    donor_indices = sample_nuisance_donor_indices(
        labels, weights, shuffle_mode=shuffle_mode)
    loss, accuracy, _ = density_ratio_critic_loss(
        critic, latent.detach(), labels, nuisance, weights,
        permutation=donor_indices, shuffle_mode=shuffle_mode)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    context_accuracy = critic_context_only_accuracy(
        critic, latent.detach(), labels, nuisance, weights,
        permutation=donor_indices, shuffle_mode=shuffle_mode)
    return loss.detach(), accuracy.detach(), context_accuracy.detach()


def train_epoch(
    model,
    critic,
    train_loader,
    model_optimizer,
    critic_optimizer,
    device: torch.device,
    args,
    contrastive_loss,
    nurd_active: bool,
    information_weight: float,
) -> Dict[str, object]:
    metrics = EpochMetrics(model.classifier.out_features)
    critic_iterator = iter(train_loader) if nurd_active else None

    for batch in train_loader:
        if nurd_active:
            for _ in range(args.critic_steps):
                try:
                    critic_batch = next(critic_iterator)
                except StopIteration:
                    critic_iterator = iter(train_loader)
                    critic_batch = next(critic_iterator)
                critic_loss, critic_accuracy, context_accuracy = critic_update(
                    model, critic, critic_optimizer, critic_batch, device,
                    args.critic_shuffle_mode)
                metrics.update_critic(
                    critic_loss, critic_accuracy, context_accuracy)

        inputs, labels, nuisance, _ae_reco, weights, _physics = _move_batch(
            batch, device)
        model.train()
        critic.eval()
        latent, logits = model(inputs)
        ce = F.cross_entropy(logits, labels, reduction="none")

        if nurd_active:
            with frozen_parameters(critic):
                information_per_event = engineer_information_penalty(
                    critic, latent, labels, nuisance)
        else:
            information_per_event = torch.zeros_like(ce)
        # Weights are normalized to mean one over the complete training split.
        # A fixed denominator is unbiased even for the very broad Mequinna
        # generator weights; per-batch self-normalization is not.
        ce_loss = (ce * weights).mean()
        information_loss = (information_per_event * weights).mean()

        if args.contrast_weight > 0.0:
            embeddings = model.get_embeddings(latent)
            contrast_loss = contrastive_loss(embeddings, labels, weights)
        else:
            contrast_loss = latent.sum() * 0.0
        total_loss = (
            ce_loss
            + information_weight * information_loss
            + args.contrast_weight * contrast_loss
        )

        model_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        model_optimizer.step()
        metrics.update_main(
            logits.detach(), labels, weights, ce.detach(), total_loss,
            information_loss, contrast_loss)

    summary = metrics.summary()
    summary["nurd_active"] = bool(nurd_active)
    summary["effective_lambda_info"] = float(information_weight)
    return summary


@torch.no_grad()
def validate(
    model,
    critic,
    loader,
    device: torch.device,
    critic_shuffle_mode: str,
    nurd_active: bool,
) -> Dict[str, object]:
    model.eval()
    critic.eval()
    metrics = EpochMetrics(model.classifier.out_features)
    for batch in loader:
        inputs, labels, nuisance, _ae_reco, weights, _physics = _move_batch(
            batch, device)
        latent, logits = model(inputs)
        ce = F.cross_entropy(logits, labels, reduction="none")
        if nurd_active:
            information = weighted_mean(
                engineer_information_penalty(critic, latent, labels, nuisance),
                weights,
            )
            donor_indices = sample_nuisance_donor_indices(
                labels, weights, shuffle_mode=critic_shuffle_mode)
            critic_loss, critic_accuracy, _ = density_ratio_critic_loss(
                critic, latent, labels, nuisance, weights,
                permutation=donor_indices,
                shuffle_mode=critic_shuffle_mode)
            context_accuracy = critic_context_only_accuracy(
                critic, latent, labels, nuisance, weights,
                permutation=donor_indices,
                shuffle_mode=critic_shuffle_mode)
            metrics.update_critic(
                critic_loss, critic_accuracy, context_accuracy)
        else:
            information = latent.sum() * 0.0
        metrics.update_main(
            logits, labels, weights, ce,
            weighted_mean(ce, weights), information,
            latent.sum() * 0.0)
    summary = metrics.summary()
    summary["nurd_active"] = bool(nurd_active)
    return summary


def format_metrics(split: str, epoch: int, metrics: Dict[str, object]) -> str:
    return (
        f"{split} epoch={epoch} nurd_active={metrics['nurd_active']} "
        f"weighted_ce={metrics['weighted_ce']:.6f} "
        f"weighted_acc={metrics['weighted_accuracy']:.4f} "
        f"balanced_acc={metrics['balanced_accuracy']:.4f} "
        f"critic_acc={metrics['critic_accuracy']:.4f} "
        f"critic_context_acc={metrics['critic_context_only_accuracy']:.4f} "
        f"info={metrics['information_penalty']:.6f} "
        f"per_class={metrics['per_class_accuracy']}"
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.critic_steps < 1:
        raise ValueError("critic_steps must be at least one.")
    if args.lambda_info < 0.0 or args.contrast_weight < 0.0:
        raise ValueError("Loss weights must be non-negative.")
    if args.critic_start_epoch < 0:
        raise ValueError("critic_start_epoch must be non-negative.")
    if args.critic_warmup_epochs < 0 or args.lambda_info_ramp_epochs < 0:
        raise ValueError("critic warm-up and lambda ramp must be non-negative.")
    if args.checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative.")
    if args.lr_schedule_epochs < 0:
        raise ValueError("lr_schedule_epochs must be non-negative.")
    if not 0.5 <= args.balance_clip_quantile <= 1.0:
        raise ValueError("balance_clip_quantile must lie in [0.5, 1.0].")
    selection_start_epoch = checkpoint_selection_start_epoch(
        args.nurd_enabled, args.critic_start_epoch,
        args.critic_warmup_epochs, args.lambda_info_ramp_epochs,
        args.selection_after_ramp)
    if selection_start_epoch > args.epochs:
        raise ValueError(
            "The critic must start within the requested epochs: "
            f"start={selection_start_epoch}, epochs={args.epochs}.")

    set_random_seed(args.manualSeed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = bool(torch.cuda.is_available())

    directory = os.path.join(
        "checkpoints", "hlt", args.project_name, args.exp_name)
    os.makedirs(directory, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s : %(message)s")
    logger = logging.getLogger("train_hlt")
    file_handler = logging.FileHandler(
        os.path.join(directory, "train.log"), mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s : %(message)s"))
    logger.addHandler(file_handler)

    wandb = None
    if not args.local_testing:
        import wandb as wandb_module
        wandb = wandb_module
        wandb.init(
            name=args.exp_name,
            project="nurd-ood-" + args.project_name,
            reinit=True,
            config=vars(args),
        )

    ae_model, ae_checkpoint = load_ae_checkpoint(args.ae_ckpt, device)
    ae_checkpoint_sha256 = file_sha256(args.ae_ckpt)
    train_dataset, val_dataset, preprocessing = build_hlt_datasets(
        args.data,
        ae_model,
        val_split=args.val_split,
        seed=args.manualSeed,
        max_events=args.max_events,
        exclude_labels=args.exclude_labels,
        gen_weight_path=args.gen_weight_path,
        qcd_label=args.qcd_label,
        target_scheme=args.target_scheme,
        generator_weight_scope=args.generator_weight_scope,
        ae_scaler=ae_checkpoint["ae_scaler"],
        balance_strata=args.balance_strata,
        balance_binning=args.balance_binning,
        balance_clip_quantile=args.balance_clip_quantile,
        ae_batch_size=args.ae_batch_size,
    )
    validate_ae_training_contract(ae_checkpoint, preprocessing)
    num_classes = int(train_dataset.labels.max()) + 1
    pin_memory = bool(torch.cuda.is_available())
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = HLTContrastiveModel(
        num_classes=num_classes,
        embed_size=args.embed_size,
        latent_dim=args.latent_dim,
        proj_dim=args.proj_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        linear_dim=args.linear_dim,
        num_tokens=train_dataset.num_tokens,
        dropout=args.dropout,
    ).to(device)
    warm_start = None
    if args.warm_start_ckpt:
        if not os.path.isfile(args.warm_start_ckpt):
            raise FileNotFoundError(
                f"Warm-start checkpoint does not exist: {args.warm_start_ckpt}")
        warm_start = load_warm_start_model(
            model, args.warm_start_ckpt, args, preprocessing,
            ae_checkpoint_sha256)
    critic = HLTCritic(args.latent_dim, num_classes).to(device)
    model_optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=args.critic_lr,
        weight_decay=args.critic_weight_decay)
    lr_schedule_epochs = args.lr_schedule_epochs or args.epochs
    if not args.cosine:
        model_scheduler = None
    elif lr_schedule_epochs == args.epochs:
        # Preserve the original campaign behavior exactly when no independent
        # LR horizon is requested.
        model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optimizer,
            T_max=max(args.epochs, 1),
            eta_min=args.lr * 1e-3,
        )
    else:
        model_scheduler = torch.optim.lr_scheduler.LambdaLR(
            model_optimizer,
            lr_lambda=lambda completed: cosine_lr_multiplier(
                completed, lr_schedule_epochs),
        )
    contrastive_loss = SupConLoss(args.contrast_temp)

    logger.info(
        "Continuous density-ratio training: events=%d/%d classes=%d "
        "target_scheme=%s "
        "balance_strata=%d balance_binning=%s balance_clip_quantile=%.4f "
        "lambda_info=%.3f critic_steps=%d shuffle_mode=%s",
        len(train_dataset), len(val_dataset), num_classes,
        args.target_scheme,
        args.balance_strata, args.balance_binning,
        args.balance_clip_quantile, args.lambda_info, args.critic_steps,
        args.critic_shuffle_mode)
    logger.info(
        "NURD schedule: enabled=%d critic_start_epoch=%d "
        "critic_warmup_epochs=%d lambda_ramp_epochs=%d lambda_max=%.6f "
        "selection_start_epoch=%d selection_after_ramp=%d",
        args.nurd_enabled, args.critic_start_epoch,
        args.critic_warmup_epochs, args.lambda_info_ramp_epochs,
        args.lambda_info, selection_start_epoch, args.selection_after_ramp)
    if warm_start is not None:
        logger.info("Warm-start provenance: %s", warm_start)
    logger.info(
        "Optimization schedule: maximum_epochs=%d lr_schedule_epochs=%d "
        "patience=%d checkpoint_every=%d",
        args.epochs, lr_schedule_epochs, args.patience,
        args.checkpoint_every)
    logger.info("Weighting metadata: %s", preprocessing["weighting"])

    best_val = math.inf
    bad_epochs = 0
    main_checkpoint_path = os.path.join(directory, "checkpoint_main.pth.tar")
    critic_checkpoint_path = os.path.join(directory, "checkpoint_critic.pth.tar")
    history = []
    last_payload = None

    for epoch in range(1, args.epochs + 1):
        nurd_active = nurd_active_for_epoch(
            epoch, args.nurd_enabled, args.critic_start_epoch)
        information_weight = information_weight_for_epoch(
            epoch, args.nurd_enabled, args.critic_start_epoch,
            args.critic_warmup_epochs, args.lambda_info_ramp_epochs,
            args.lambda_info)
        train_metrics = train_epoch(
            model, critic, train_loader,
            model_optimizer, critic_optimizer,
            device, args, contrastive_loss, nurd_active,
            information_weight)
        val_metrics = validate(
            model, critic, val_loader, device, args.critic_shuffle_mode,
            nurd_active)
        if model_scheduler is not None:
            model_scheduler.step()
        logger.info(format_metrics("train", epoch, train_metrics))
        logger.info(format_metrics("validation", epoch, val_metrics))
        logger.info(
            "NURD state epoch=%d active=%s effective_lambda_info=%.6f",
            epoch, nurd_active, information_weight)
        history.append({
            "epoch": epoch,
            "nurd_active": nurd_active,
            "effective_lambda_info": information_weight,
            "train": train_metrics,
            "validation": val_metrics,
        })

        if wandb is not None:
            flat = {
                "epoch": epoch,
                "lr": model_optimizer.param_groups[0]["lr"],
                "nurd_active": int(nurd_active),
                "effective_lambda_info": information_weight,
            }
            for split, values in (("train", train_metrics), ("validation", val_metrics)):
                for key, value in values.items():
                    if key == "per_class_accuracy":
                        for label, accuracy in value.items():
                            flat[f"{split}/class_{label}_accuracy"] = accuracy
                    else:
                        flat[f"{split}/{key}"] = value
            wandb.log(flat, step=epoch)

        current_payload = {
            "epoch": epoch,
            "state_dict_model": model.state_dict(),
            "state_dict_critic": critic.state_dict(),
            "config": vars(args),
            "ae_scaler": preprocessing["ae_scaler"],
            "preprocessing": preprocessing,
            "ae_checkpoint_sha256": ae_checkpoint_sha256,
            "selection_metric": "training-fit-weighted_validation_ce",
            "selection_value": float(val_metrics["weighted_ce"]),
            "nurd_active": nurd_active,
            "effective_lambda_info": information_weight,
            "train_metrics": train_metrics,
            "validation_metrics": val_metrics,
            "warm_start": warm_start,
        }
        last_payload = current_payload
        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            snapshot = dict(current_payload)
            snapshot["checkpoint_role"] = "periodic_snapshot"
            snapshot_path = os.path.join(
                directory, f"checkpoint_epoch_{epoch:03d}.pth.tar")
            torch.save(snapshot, snapshot_path)
            logger.info("Saved periodic checkpoint: %s", snapshot_path)

        if epoch < selection_start_epoch:
            logger.info(
                "Checkpoint selection and early stopping disabled until epoch %d.",
                selection_start_epoch)
            continue

        if val_metrics["weighted_ce"] < best_val:
            best_val = float(val_metrics["weighted_ce"])
            bad_epochs = 0
            payload = dict(current_payload)
            payload["checkpoint_role"] = "best_weighted_validation_ce"
            torch.save(payload, main_checkpoint_path)
            torch.save({
                "epoch": epoch,
                "state_dict_critic": critic.state_dict(),
                "config": vars(args),
            }, critic_checkpoint_path)
            if wandb is not None:
                wandb.run.summary["selected_epoch"] = epoch
                wandb.run.summary[
                    "selected_validation_balanced_accuracy"
                ] = val_metrics["balanced_accuracy"]
                wandb.run.summary[
                    "selected_validation_weighted_accuracy"
                ] = val_metrics["weighted_accuracy"]
                wandb.run.summary[
                    "selected_weighted_validation_ce"
                ] = val_metrics["weighted_ce"]
                wandb.run.summary[
                    "selected_effective_lambda_info"
                ] = information_weight
            logger.info("Saved best checkpoint: %s", main_checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                logger.info("Early stopping after %d non-improving epochs.", bad_epochs)
                break

    if args.checkpoint_every and last_payload is not None:
        final_payload = dict(last_payload)
        final_payload["checkpoint_role"] = "final_state"
        final_checkpoint_path = os.path.join(
            directory, "checkpoint_final.pth.tar")
        torch.save(final_payload, final_checkpoint_path)
        logger.info("Saved final-state checkpoint: %s", final_checkpoint_path)

    history_path = os.path.join(directory, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as output:
        json.dump(history, output, indent=2)
    logger.info("Done. best weighted validation CE=%.6f", best_val)
    if wandb is not None:
        wandb.run.summary["best_weighted_validation_ce"] = best_val
        wandb.finish()


if __name__ == "__main__":
    main()
