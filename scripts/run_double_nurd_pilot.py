"""Matched global-QCD versus global+tail-QCD density-ratio NURD pilot.

Standalone experiment: no changes to the ordinary trainer, AE, or source files.
The checking sample is deliberately not an input to this training program.
"""
import argparse
from contextlib import contextmanager
import csv
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from models.hlt_autoencoder import HLTAutoencoder
from models.hlt_con import HLTCritic
from scripts.audit_qcd_abcd_composition import infer_ae_scores
from scripts.audit_tail_pilot_split import quantile
from scripts.prepare_tail_pilot import report_json, validate_split, weight_stats
from scripts.run_tail_nurd_pilot import (
    FixedBatches, SafeDataset, finite_report, model_from_config, plan_populations,
)
from train_hlt import SupConLoss, build_parser, set_random_seed
from utils.hlt_data import load_hlt_sample
from utils.hlt_density_ratio import (
    critic_context_only_accuracy, density_ratio_critic_loss,
    engineer_information_penalty, frozen_parameters,
)
from utils.hlt_targets import checkpoint_target_labels
from utils.hlt_weights import (
    file_sha256, fit_joint_balance, load_generator_weights, sample_signature,
    stratified_split_indices,
)

ARMS = ("global", "double")
PROTOCOL = "double_nurd_global_and_conditional_tail_QCD_v1"


@contextmanager
def inference_representation(model):
    """Inference-mode representation WITH gradients; restore all module modes."""
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


def tail_lambda(update, ramp_updates, maximum):
    if update < 1 or ramp_updates < 0 or maximum < 0:
        raise ValueError("Invalid one-based update or tail ramp")
    return maximum * min(1., update / ramp_updates) if ramp_updates else maximum


class PhysicalQCDStream:
    """IID physics-distributed QCD; unit loss weights avoid double weighting.

    Draw a bounded chunk at a time (not millions of torch.multinomial calls).
    Both anchors and independent shuffled donors then have the same physical
    population. A tail stream contains ONLY above-threshold training QCD.
    """
    def __init__(self, dataset, physics_weights, batch_size, seed):
        weights = torch.as_tensor(physics_weights, dtype=torch.float64)
        if (len(weights) != len(dataset) or not len(weights)
                or not torch.isfinite(weights).all() or (weights < 0).any()
                or weights.sum() <= 0 or not torch.all(dataset.labels == 1)
                or not torch.all(dataset.effective_weights == 1)):
            raise ValueError("Expected positive physical QCD population and unit loss weights")
        self.dataset, self.weights, self.batch_size = dataset, weights, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.iterator = None

    def __next__(self):
        if self.iterator is None:
            sampler = WeightedRandomSampler(
                self.weights, self.batch_size * 128, replacement=True,
                generator=self.generator,
            )
            self.iterator = iter(DataLoader(
                self.dataset, batch_size=self.batch_size, sampler=sampler,
                num_workers=0, pin_memory=torch.cuda.is_available(),
                generator=self.generator,
            ))
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = None
            return next(self)


def critic_step(model, critic, optimizer, batch, device, steps=1):
    x, labels, nuisance, _, unit_weights, _ = (v.to(device) for v in batch)
    # Exactly the representation used by the encoder penalty and final MD.
    with inference_representation(model), torch.no_grad():
        latent, _ = model(x)
    critic.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, accuracy, _ = density_ratio_critic_loss(
            critic, latent.detach(), labels, nuisance, unit_weights,
            shuffle_mode="weighted_within_class",
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite critic loss")
        loss.backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    critic.eval()
    with torch.no_grad():
        context = critic_context_only_accuracy(
            critic, latent, labels, nuisance, unit_weights,
            shuffle_mode="weighted_within_class",
        )
    return {"loss": float(loss.detach()), "accuracy": float(accuracy),
            "context_only_accuracy": float(context)}


def information(model, critic, batch, device):
    x, labels, nuisance, _, unit_weights, _ = (v.to(device) for v in batch)
    critic.eval()
    with inference_representation(model), frozen_parameters(critic):
        latent, _ = model(x)
        values = engineer_information_penalty(critic, latent, labels, nuisance)
    return (values * unit_weights).mean()


def encoder_step(model, critics, optimizer, supervised, adversarial, device,
                 contrastive, global_coefficient, tail_coefficient):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    x, labels, _, _, weights, _ = (v.to(device) for v in supervised)
    latent, logits = model(x)
    ce = (F.cross_entropy(logits, labels, reduction="none") * weights).mean()
    embeddings = model.get_embeddings(latent)
    supcon = contrastive(embeddings, labels, weights)
    # Backward sequentially: do not retain three 400-token encoder graphs.
    supervised_loss = ce + .3 * supcon
    supervised_loss.backward()
    info_values = {}
    for name, coefficient in (("global", global_coefficient), ("tail", tail_coefficient)):
        if name not in critics:
            continue
        penalty = information(model, critics[name], adversarial[name], device)
        (coefficient * penalty).backward()
        info_values[name] = float(penalty.detach())
    # Same guard in both arms, explicitly recorded in the experiment contract.
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
    optimizer.step()
    correct = logits.detach().argmax(1) == labels
    return {"weighted_ce": float(ce.detach()), "supcon": float(supcon.detach()),
            "batch_weighted_accuracy": float((correct * weights).sum() / weights.sum()),
            "batch_balanced_accuracy": float(torch.stack([
                correct[labels == k].float().mean() for k in labels.unique()]).mean()),
            "information": info_values, "gradient_norm_before_clip": float(gradient_norm),
            "global_lambda": global_coefficient, "tail_lambda": tail_coefficient}


def prepare(args):
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ae = torch.load(args.ae_checkpoint, map_location="cpu", weights_only=False)
    pre, cfg = source["preprocessing"], source["config"]
    ae_hash = file_sha256(args.ae_checkpoint)
    if (source["ae_checkpoint_sha256"] != ae_hash or pre["target_scheme"] != "decay8"
            or not source["nurd_active"] or source["effective_lambda_info"] != 1
            or source.get("classification_head_trained") is False
            or cfg["contrast_weight"] != .3 or cfg["balance_strata"] != 1
            or cfg["balance_clip_quantile"] != .995 or ae["weighting"]["scheme"] != "physics"):
        raise ValueError("Expected original full lambda=1, decay8, SupCon=.3, class-only model and its physics AE")
    raw = load_hlt_sample(args.sample)
    if len(np.unique(raw["event_id"].numpy(), axis=0)) != len(raw["label"]):
        raise ValueError("Duplicate training event IDs could leak across calibration roles")
    signature = sample_signature(raw)
    if signature != pre["data_signature"] or signature != ae["data_signature"]:
        raise ValueError("Training manifest changed")
    weight_tensor, metadata = load_generator_weights(
        str(args.weights), raw["label"], sample=raw, qcd_label=1, weight_scope="all_events")
    if metadata["effective_physics_weight_sha256"] != pre["weighting"]["generator"]["effective_physics_weight_sha256"]:
        raise ValueError("Training physics weights changed")
    weights = weight_tensor.numpy()
    labels = checkpoint_target_labels(raw, pre).numpy()
    train, val = validate_split(pre["split"]["train_indices"].numpy(),
                                pre["split"]["validation_indices"].numpy(), len(labels))
    control = json.loads((args.ae_cache_dir / "AE_control_summary.json").read_text())
    if control["baseline_sha256"] != ae_hash:
        raise ValueError("AE cache is from a different checkpoint")
    with np.load(args.ae_cache_dir / "frozen_AE_scores.npz", allow_pickle=False) as cache:
        if not np.array_equal(cache["training_event_id"], raw["event_id"].numpy()):
            raise ValueError("AE cache event IDs changed")
        scores = cache["baseline_training_manifest_scores"].copy()
    if len(scores) != len(labels) or not np.isfinite(scores).all():
        raise ValueError("Invalid AE scores")
    ae_model = HLTAutoencoder(ae["ae_config"])
    ae_model.load_state_dict(ae["ae"]); ae_model.eval()
    probe = np.random.default_rng(args.seed).choice(len(scores), min(1024, len(scores)), replace=False)
    rescored = infer_ae_scores(ae_model, source["ae_scaler"], raw, torch.as_tensor(probe), "cpu", 512)
    if not np.allclose(rescored, scores[probe], rtol=2e-4, atol=1e-7):
        raise ValueError("Frozen AE scoring parity failed")
    if "split" in ae:
        ae_train = ae["split"]["train_indices"].numpy()
        calibration = ae["split"]["validation_indices"].numpy()
    else:
        m = ae["weighting"]
        ae_train, calibration = (r.numpy() for r in stratified_split_indices(
            raw["label"], m["validation_fraction"], m["split_seed"], weights=weight_tensor))
    validate_split(ae_train, calibration, len(labels))
    cut, eligible, tail = plan_populations(scores, weights, train, calibration)
    if not np.isclose(cut, control["results"]["baseline"]["ae_threshold"], rtol=1e-7, atol=1e-9):
        raise ValueError("AE-unseen reference threshold changed")
    for rows in (eligible, tail):
        if set(labels[rows]) != set(range(8)) or any(weights[rows[labels[rows] == k]].sum() <= 0 for k in range(8)):
            raise ValueError("All eight positive-weight classes required in full and tail training populations")
    defaults = vars(build_parser().parse_args(["--data", str(args.sample), "--ae_ckpt", str(args.ae_checkpoint)]))
    config = {**defaults, **cfg, "lr": args.lr, "contrast_weight": .3,
              "batch_size": args.batch_size, "lambda_info": args.global_lambda,
              "critic_shuffle_mode": "weighted_within_class"}
    nuisance = (scores - float(pre["nuisance_transform"]["mean"])) / float(pre["nuisance_transform"]["std"])
    if not np.isfinite(nuisance).all():
        raise ValueError("Nonfinite source-standardized nuisance")
    effective, balance = fit_joint_balance(torch.from_numpy(labels[eligible]),
        torch.from_numpy(scores[eligible]), torch.from_numpy(weights[eligible]), 1, "log_fixed", .995)
    roles = {"full_train": eligible, "tail_train": tail, "calibration": calibration,
             "ae_gradient_train": ae_train, "source_validation": val,
             "global_QCD": eligible[labels[eligible] == 1], "tail_QCD": tail[labels[tail] == 1]}
    def dataset(rows, optimization_weights):
        return SafeDataset(raw["pf"], torch.from_numpy(labels), torch.from_numpy(nuisance),
            torch.from_numpy(scores), optimization_weights, weight_tensor, torch.as_tensor(rows), "double_nurd_train")
    datasets = {"supervised": dataset(eligible, effective)}
    for name in ("global", "tail"):
        rows = roles[f"{name}_QCD"]
        datasets[name] = dataset(rows, torch.ones(len(rows)))
    manifest = {
        "protocol": PROTOCOL, "code_commit": args.code_commit,
        "source_checkpoint": str(args.checkpoint), "source_sha256": file_sha256(args.checkpoint),
        "AE_sha256": ae_hash, "source_epoch": source["epoch"], "source_config": cfg,
        "data_signature": signature, "physics_weight_sha256": metadata["effective_physics_weight_sha256"],
        "AE_cache_sha256": file_sha256(args.ae_cache_dir / "frozen_AE_scores.npz"),
        "ae_threshold": cut, "updates_per_arm": args.updates, "batch_size": args.batch_size,
        "critic_steps_per_update": args.critic_steps, "critic_warmup_updates": args.warmup_updates,
        "tail_ramp_updates": args.ramp_updates, "minimum_post_ramp_updates": args.min_post_ramp_updates,
        "actual_post_ramp_updates": args.updates - args.ramp_updates,
        "global_lambda": args.global_lambda, "maximum_tail_lambda": args.tail_lambda,
        "SupCon": .3, "learning_rate": args.lr, "gradient_norm_clip": 10.,
        "checkpoint_selection": "Final fixed budget only; no CE early stopping or checking-based epoch selection",
        "supervised_weights": "Same class-only physics weights with per-class .995 clipping, fit once on common training rows",
        "critic_weights": "Original UNCLIPPED physical QCD distribution via replacement sampling; unit losses; no second physics weighting",
        "tail_definition": "Fixed physical all-background AE-unseen calibration 90th percentile; QCD critics condition on this cut",
        "representation": "eval-mode gradient-enabled encoder for information, eval/no-grad for critic; train-mode CE/SupCon",
        "optimizer_state": "Fresh Adam for both arms; source encoder/global critic cloned; independent fresh tail critic",
        "source_classifier_training_rows_in_calibration": int(np.isin(calibration, train).sum()),
        "limitations": ["Exploratory previously inspected MC, not a blind test",
            "AE calibration rows selected the original AE epoch but were gradient-unseen",
            "Some calibration rows were seen by the historical source classifier, never by either new arm",
            "Both arms differ from historical NURD in QCD-only physics measure and representation mode; only their matched difference isolates the extra tail critic",
            "The double arm adds regularization strength as well as tail specialization; a later global-lambda=2 control would isolate specialization from total pressure",
            "No signal sensitivity conclusion from closure alone"],
    }
    return source, config, roles, datasets, weights, labels, balance, manifest


def validate_options(args):
    for key in ("updates", "batch_size", "report_every", "save_every", "warmup_updates", "critic_steps", "min_post_ramp_updates"):
        if getattr(args, key) < 1:
            raise ValueError(f"Positive {key} required")
    if args.batch_size < 2 or args.ramp_updates < 0 or args.updates < args.ramp_updates + args.min_post_ramp_updates:
        raise ValueError("Budget must include the complete ramp plus minimum post-ramp updates")
    if not all(np.isfinite(v) and v > 0 for v in (args.lr, args.global_lambda, args.tail_lambda)):
        raise ValueError("Positive finite LR and information coefficients required")


def run(args):
    validate_options(args)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source, config, roles, datasets, weights, labels, balance, manifest = prepare(args)
    args.output_dir.mkdir(parents=True)
    for path, name in ((args.checkpoint, "source_lambda1_checkpoint.pth.tar"),
                       (args.ae_checkpoint, "original_checkpoint_ae.pth")):
        shutil.copy2(path, args.output_dir / name)
        if file_sha256(path) != file_sha256(args.output_dir / name):
            raise ValueError("Source snapshot verification failed")
    np.savez_compressed(args.output_dir / "training_roles.npz", **roles)
    (args.output_dir / "experiment_manifest.json").write_text(report_json(finite_report(manifest)))
    statistics = []
    for role, rows in roles.items():
        for label in range(8):
            statistics.append({"role": role, "label": label,
                               **weight_stats(weights[rows[labels[rows] == label]])})
    with (args.output_dir / "class_statistics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(statistics[0]))
        writer.writeheader(); writer.writerows(statistics)
    print(f"Fixed AE cut={manifest['ae_threshold']:.8g}; global/tail QCD physical ESS="
          f"{weight_stats(weights[roles['global_QCD']])['effective_events']:.1f}/"
          f"{weight_stats(weights[roles['tail_QCD']])['effective_events']:.1f}", flush=True)
    device = torch.device(args.device)
    for arm in ARMS:
        started = time.monotonic()
        set_random_seed(args.seed)
        model = model_from_config(config, datasets["supervised"].num_tokens, device)
        model.load_state_dict(source["state_dict_model"])
        critics = {"global": HLTCritic(config["latent_dim"], 8).to(device)}
        critics["global"].load_state_dict(source["state_dict_critic"])
        if arm == "double":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(args.seed + 9000)
                critics["tail"] = HLTCritic(config["latent_dim"], 8).to(device)
        critic_opts = {name: torch.optim.Adam(c.parameters(), lr=config["critic_lr"],
                      weight_decay=config["critic_weight_decay"]) for name, c in critics.items()}
        streams = {name: PhysicalQCDStream(datasets[name], weights[roles[f"{name}_QCD"]],
                   args.batch_size, args.seed + (10 if name == "global" else 20)) for name in critics}
        print(f"{arm}: critic-only warm-up {args.warmup_updates}; encoder frozen", flush=True)
        for step in range(args.warmup_updates):
            for name, critic in critics.items():
                set_random_seed(args.seed + 100000 + step + (0 if name == "global" else 500000))
                critic_step(model, critic, critic_opts[name], next(streams[name]), device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=config["weight_decay"])
        supervised = iter(FixedBatches(datasets["supervised"], args.batch_size, args.updates, args.seed))
        contrastive = SupConLoss(config["contrast_temp"])
        folder = args.output_dir / arm; folder.mkdir()
        history = []; recent = []
        for update in range(1, args.updates + 1):
            # Separate physics draws for critic fitting and encoder gradients.
            for name, critic in critics.items():
                set_random_seed(args.seed + 1000000 + update + (0 if name == "global" else 500000))
                diagnostic = critic_step(model, critic, critic_opts[name], next(streams[name]),
                                         device, args.critic_steps)
                if name == "global": global_diagnostic = diagnostic
                else: tail_diagnostic = diagnostic
            batches = {name: next(stream) for name, stream in streams.items()}
            # Match supervised dropout RNG across arms despite the extra critic.
            set_random_seed(args.seed + 3000000 + update)
            metrics = encoder_step(model, critics, optimizer, next(supervised), batches, device,
                contrastive, args.global_lambda,
                tail_lambda(update, args.ramp_updates, args.tail_lambda) if arm == "double" else 0.)
            metrics["global_critic"] = global_diagnostic
            if arm == "double": metrics["tail_critic"] = tail_diagnostic
            recent.append(metrics)
            if update % args.report_every == 0 or update == args.updates:
                def average(values):
                    return {k: average([v[k] for v in values]) if isinstance(values[0][k], dict)
                            else float(np.mean([v[k] for v in values])) for k in values[0]}
                entry = {"encoder_updates": update, "elapsed_seconds": time.monotonic()-started,
                         "last_update": metrics, "window_mean": average(recent)}
                history.append(entry); recent = []
                (folder / "training_history.json").write_text(report_json(finite_report(history)))
                elapsed = time.monotonic()-started
                print(f"{arm} update={update}/{args.updates} CE={entry['window_mean']['weighted_ce']:.4f} "
                      f"tail_lambda={metrics['tail_lambda']:.3f} elapsed={elapsed/60:.1f}min "
                      f"rough_remaining={elapsed/update*(args.updates-update)/60:.1f}min", flush=True)
            if update % args.save_every == 0 or update == args.updates:
                payload = {"state_dict_model": model.state_dict(),
                    "state_dict_critic": critics["global"].state_dict(),
                    "state_dict_tail_critic": critics["tail"].state_dict() if "tail" in critics else None,
                    "optimizer": optimizer.state_dict(),
                    "critic_optimizers": {k: v.state_dict() for k, v in critic_opts.items()},
                    "config": config, "preprocessing": source["preprocessing"],
                    "ae_scaler": source["ae_scaler"], "ae_checkpoint_sha256": manifest["AE_sha256"],
                    "pilot_protocol": PROTOCOL, "arm": arm, "encoder_updates": update,
                    "source_sha256": manifest["source_sha256"], "balance_spec": balance,
                    "actual_train_indices": torch.as_tensor(roles["full_train"]),
                    "nurd_active": True, "classification_head_trained": True,
                    "effective_lambda_info": args.global_lambda, "effective_tail_lambda": metrics["tail_lambda"],
                    "checkpoint_role": "fixed_budget_final" if update == args.updates else "periodic_no_selection"}
                path = folder / ("checkpoint_final.pth.tar" if update == args.updates else f"checkpoint_update_{update:06d}.pth.tar")
                torch.save(payload, path)
        del model, critics, optimizer, critic_opts, streams, supervised, payload
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
    if (file_sha256(args.checkpoint) != manifest["source_sha256"]
            or file_sha256(args.ae_checkpoint) != manifest["AE_sha256"]):
        raise ValueError("Original checkpoint changed during the experiment")
    manifest["final_checkpoint_sha256"] = {arm: file_sha256(args.output_dir / arm / "checkpoint_final.pth.tar") for arm in ARMS}
    manifest["source_checkpoints_unchanged"] = True
    (args.output_dir / "training_complete.json").write_text(report_json(finite_report(manifest)))
    print("BOTH MATCHED ARMS FINISHED. Ready for separate full + frozen-tail checking evaluation.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "ae-checkpoint", "sample", "weights", "ae-cache-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--warmup-updates", type=int, default=300)
    parser.add_argument("--ramp-updates", type=int, default=1000)
    parser.add_argument("--min-post-ramp-updates", type=int, default=3000)
    parser.add_argument("--critic-steps", type=int, default=3)
    parser.add_argument("--global-lambda", type=float, default=1.)
    parser.add_argument("--tail-lambda", type=float, default=1.)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--code-commit", default="")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run(args)


if __name__ == "__main__":
    main()
