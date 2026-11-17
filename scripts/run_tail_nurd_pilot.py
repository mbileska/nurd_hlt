"""Fixed-budget matched full/tail NURD continuations. No AE training or calibration transform."""
import argparse
import csv
import gc
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.hlt_smcocktail_dataset import HLTSmCocktailDataset
from models.hlt_con import HLTContrastiveModel, HLTCritic
from scripts.prepare_tail_pilot import report_json, validate_split, weight_stats
from scripts.audit_tail_pilot_split import quantile, retention
from scripts.audit_qcd_abcd_composition import infer_latents
from scripts.audit_qcd_abcd_composition import infer_ae_scores
from scripts.scan_cached_tail_cuts import scan
from scripts.compare_qcd_score_audit import fixed_point, paired_bootstrap
from train_hlt import SupConLoss, build_parser, set_random_seed, train_epoch, validate
from utils.hlt_data import load_hlt_sample
from utils.hlt_targets import checkpoint_target_labels
from utils.hlt_weights import (
    file_sha256,
    fit_joint_balance,
    apply_joint_balance,
    sample_signature,
    load_generator_weights,
    stratified_split_indices,
)


class SafeDataset(HLTSmCocktailDataset):
    """Sanitize one PF row at a time rather than copying the entire 23 GB tensor."""

    def __getitem__(self, item):
        values = super().__getitem__(item)
        return (
            torch.nan_to_num(values[0].float(), nan=0.0, posinf=0.0, neginf=0.0),
            *values[1:],
        )


class FixedBatches:
    def __init__(self, dataset, batch_size, updates, seed):
        if len(dataset) < batch_size:
            raise ValueError("Training population smaller than a full batch")
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            generator=torch.Generator().manual_seed(seed),
        )
        self.updates = updates

    def __iter__(self):
        iterator = iter(self.loader)
        for _ in range(self.updates):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self.loader)
                batch = next(iterator)
            yield batch

    def __len__(self):
        return self.updates


def plan_populations(scores, weights, original_train, calibration):
    # Both continuations exclude every calibration row, including any that
    # occurred in the source classifier's historical training population.
    eligible = np.setdiff1d(original_train, calibration)
    cut = float(quantile(scores[calibration], weights[calibration], 0.9))
    tail = eligible[scores[eligible] >= cut]
    return cut, eligible, tail


def model_from_config(config, tokens, device):
    return HLTContrastiveModel(
        num_classes=8,
        num_tokens=tokens,
        **{
            k: config[k]
            for k in (
                "embed_size",
                "latent_dim",
                "proj_dim",
                "num_heads",
                "num_layers",
                "dim_ff",
                "linear_dim",
                "dropout",
            )
        },
    ).to(device)


def finite_report(value):
    if isinstance(value, dict):
        return {k: finite_report(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_report(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not np.isfinite(value) else value


def md_reference(model, raw, rows, weights, device, batch_size):
    latent = infer_latents(model, raw, torch.as_tensor(rows), device, batch_size)
    w = weights[rows] / weights[rows].sum()
    mean = np.sum(latent * w[:, None], axis=0)
    centered = latent - mean
    covariance = (centered * w[:, None]).T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return mean, eigenvectors / np.sqrt(np.maximum(eigenvalues, 1e-6))


def evaluate(
    model,
    critic,
    raw,
    check,
    rows,
    check_rows,
    scores,
    check_scores,
    labels,
    check_labels,
    weights,
    check_weights,
    reference_rows,
    metric_spec,
    config,
    device,
    batch_size,
):
    mean, whitening = md_reference(
        model, raw, reference_rows, weights, device, batch_size
    )
    cal_latent = infer_latents(model, raw, torch.as_tensor(rows), device, batch_size)
    test_latent = infer_latents(
        model, check, torch.as_tensor(check_rows), device, batch_size
    )
    cal_md = np.square((cal_latent - mean) @ whitening).sum(1)
    test_md = np.square((test_latent - mean) @ whitening).sum(1)
    cal_qcd = labels[rows] == 1
    test_qcd = check_labels[check_rows] == 1
    calibration = {
        "ae": scores[rows][cal_qcd],
        "md": cal_md[cal_qcd],
        "weight": weights[rows][cal_qcd],
    }
    checking = {
        "ae": check_scores[check_rows][test_qcd],
        "md": test_md[test_qcd],
        "weight": check_weights[check_rows][test_qcd],
    }
    point, codes = fixed_point(checking, calibration, 0.5, 0.1)
    point["fixed_cut_ratio_uncertainty"] = float(
        point["predicted_over_observed"]
        * np.sqrt(
            np.sum(
                np.divide(
                    1.0,
                    point["region_effective_counts"],
                    out=np.full(4, np.inf),
                    where=np.asarray(point["region_effective_counts"]) > 0,
                )
            )
        )
    )
    point["calibration_A_efficiency_target_met"] = bool(
        abs(point["validation_A_weighted_efficiency"] - 0.1) <= 0.02
    )
    percentiles = np.linspace(0.1, 0.95, 35)
    tx = quantile(calibration["ae"], calibration["weight"], percentiles)
    ty = quantile(calibration["md"], calibration["weight"], percentiles)
    grid = scan(
        checking["ae"], checking["md"], checking["weight"], tx, ty, [50, 50, 50, 500]
    )
    # Fixed, COMMON checking weights (fit on full eligible training) make
    # classification metrics comparable despite each branch's train weights.
    effective = apply_joint_balance(
        torch.from_numpy(check_labels[check_rows]),
        torch.from_numpy(check_scores[check_rows]),
        torch.from_numpy(check_weights[check_rows]),
        metric_spec,
    )
    ncheck = (check_scores - config["nuisance_mean"]) / config["nuisance_std"]
    dataset = SafeDataset(
        check["pf"],
        torch.from_numpy(check_labels),
        torch.from_numpy(ncheck),
        torch.from_numpy(check_scores),
        effective,
        torch.from_numpy(check_weights),
        torch.as_tensor(check_rows),
        "pilot_check",
    )
    metrics = validate(
        model,
        critic,
        DataLoader(dataset, batch_size=batch_size, num_workers=0),
        device,
        config["critic_shuffle_mode"],
        True,
    )
    return (
        {
            "classification_common_weights": metrics,
            "matched_point": point,
            "checking_grid_valid_points": int(grid["valid"].sum()),
            "MD_reference_events": len(reference_rows),
            "MD_reference_mean": mean.tolist(),
            "MD_reference_whitening": whitening.tolist(),
        },
        checking,
        codes,
        grid,
        percentiles,
    )


def run(args):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source_hash = file_sha256(args.checkpoint)
    ae_hash = file_sha256(args.ae_checkpoint)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ae = torch.load(args.ae_checkpoint, map_location="cpu", weights_only=False)
    cfg = source["config"]
    pre = source["preprocessing"]
    if (
        source["ae_checkpoint_sha256"] != ae_hash
        or pre["target_scheme"] != "decay8"
        or source.get("classification_head_trained") is False
        or not source["nurd_active"]
        or source["effective_lambda_info"] != 1
        or cfg["contrast_weight"] != 0.3
        or cfg["balance_strata"] != 1
        or cfg["balance_clip_quantile"] != 0.995
        or ae["weighting"]["scheme"] != "physics"
    ):
        raise ValueError(
            "Expected the original full-model lambda=1/class-only/SupCon=.3 checkpoint and exact AE"
        )
    raw = load_hlt_sample(args.sample)
    signature = sample_signature(raw)
    if signature != pre["data_signature"] or signature != ae["data_signature"]:
        raise ValueError("Original full training manifest differs")
    weights, metadata = load_generator_weights(
        str(args.weights),
        raw["label"],
        sample=raw,
        qcd_label=1,
        weight_scope="all_events",
    )
    if (
        metadata["effective_physics_weight_sha256"]
        != pre["weighting"]["generator"]["effective_physics_weight_sha256"]
    ):
        raise ValueError("Original physics weights differ")
    weights = weights.numpy()
    labels = checkpoint_target_labels(raw, pre).numpy()
    train, val = validate_split(
        pre["split"]["train_indices"].cpu().numpy(),
        pre["split"]["validation_indices"].cpu().numpy(),
        len(labels),
    )
    control = json.loads((args.ae_cache_dir / "AE_control_summary.json").read_text())
    if control["baseline_sha256"] != ae_hash:
        raise ValueError("AE score cache belongs to a different AE")
    cache_path = args.ae_cache_dir / "frozen_AE_scores.npz"
    cache_hash = file_sha256(cache_path)
    with np.load(cache_path, allow_pickle=False) as cache:
        if not np.array_equal(cache["training_event_id"], raw["event_id"].numpy()):
            raise ValueError("Cached AE training IDs mismatch")
        scores = cache["baseline_training_manifest_scores"].copy()
    if len(scores) != len(labels) or not np.isfinite(scores).all():
        raise ValueError("Invalid cached AE scores")
    # Small object-only parity check, never a full PF/AE inference rerun.
    from models.hlt_autoencoder import HLTAutoencoder

    ae_model = HLTAutoencoder(ae["ae_config"])
    ae_model.load_state_dict(ae["ae"])
    ae_model.eval()
    probe = np.random.default_rng(args.seed).choice(
        len(scores), min(1024, len(scores)), replace=False
    )
    rescored = infer_ae_scores(
        ae_model, source["ae_scaler"], raw, torch.as_tensor(probe), "cpu", 512
    )
    if not np.allclose(rescored, scores[probe], rtol=2e-4, atol=1e-7):
        raise ValueError("Original frozen AE cache fails scoring parity check")
    del ae_model
    # AE calibration pool is unseen to AE gradient training. Legacy classifier
    # exposure is recorded below; both new arms exclude it from optimization.
    if "split" in ae:
        ae_train = ae["split"]["train_indices"].cpu().numpy()
        calibration = ae["split"]["validation_indices"].cpu().numpy()
        membership = "archived AE indices"
    else:
        m = ae["weighting"]
        a, rows = stratified_split_indices(
            raw["label"],
            m["validation_fraction"],
            m["split_seed"],
            weights=torch.from_numpy(weights),
        )
        ae_train = a.numpy()
        calibration = rows.numpy()
        membership = (
            "reconstructed AE split; mass/loss agreement checked in prior split audit"
        )
    validate_split(ae_train, calibration, len(labels))
    cut, eligible, tail = plan_populations(scores, weights, train, calibration)
    if not np.isclose(
        cut, control["results"]["baseline"]["ae_threshold"], rtol=1e-7, atol=1e-9
    ):
        raise ValueError(
            "Reconstructed AE calibration threshold differs from the completed original-AE control"
        )
    selected_cal = calibration[scores[calibration] >= cut]
    for name, rows in (
        ("eligible_full_train", eligible),
        ("tail_train", tail),
        ("selected_calibration", selected_cal),
    ):
        if set(labels[rows]) != set(range(8)) or any(
            weights[rows[labels[rows] == k]].sum() <= 0 for k in range(8)
        ):
            raise ValueError(f"All eight positive-weight classes required in {name}")
    source_defaults = vars(
        build_parser().parse_args(
            ["--data", str(args.sample), "--ae_ckpt", str(args.ae_checkpoint)]
        )
    )
    config = {
        **source_defaults,
        **cfg,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "nurd_enabled": 1,
        "lambda_info": 1.0,
        "contrast_weight": 0.3,
        "nuisance_mean": float(pre["nuisance_transform"]["mean"]),
        "nuisance_std": float(pre["nuisance_transform"]["std"]),
    }
    nuisance = (scores - config["nuisance_mean"]) / config["nuisance_std"]
    _, metric_spec = fit_joint_balance(
        torch.from_numpy(labels[eligible]),
        torch.from_numpy(scores[eligible]),
        torch.from_numpy(weights[eligible]),
        1,
        "log_fixed",
        0.995,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.checkpoint, args.output_dir / "source_lambda1_checkpoint.pth.tar")
    shutil.copy2(args.ae_checkpoint, args.output_dir / "original_checkpoint_ae.pth")
    if (
        file_sha256(args.output_dir / "source_lambda1_checkpoint.pth.tar")
        != source_hash
        or file_sha256(args.output_dir / "original_checkpoint_ae.pth") != ae_hash
    ):
        raise ValueError("Source snapshot verification failed")
    manifest = {
        "source_checkpoint": str(args.checkpoint),
        "source_sha256": source_hash,
        "AE_sha256": ae_hash,
        "source_selected_epoch": source["epoch"],
        "AE_selected_epoch": ae.get("epoch"),
        "AE_cache_sha256": cache_hash,
        "data_signature": signature,
        "physics_weight_sha256": metadata["effective_physics_weight_sha256"],
        "ae_threshold": cut,
        "calibration_membership": membership,
        "updates_per_branch": args.updates,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "constant_lambda": 1.0,
        "SupCon": 0.3,
        "optimizer_state": "Fresh identical Adam states; model AND critic weights cloned from source, no new ramp.",
        "source_training_rows_in_calibration": int(np.isin(calibration, train).sum()),
        "train_populations": {
            name: {
                "events": len(rows),
                "physical_statistics": weight_stats(weights[rows]),
            }
            for name, rows in (("full", eligible), ("tail", tail))
        },
        "selected_calibration_QCD": weight_stats(
            weights[selected_cal[labels[selected_cal] == 1]]
        ),
        "selection_rule": "Fixed AE-unseen all-background physical 90th percentile; calibration excluded from both new training arms.",
        "checking_protocol": "Common outer held-out selected events, once after fixed budget; no epoch/cut selection using checking.",
        "code_commit": args.code_commit,
        "limitations": [
            "Previously inspected validation/held-out MC: exploratory pilot, not a new blind test.",
            "Original AE validation selected its epoch: gradient-unseen calibration is not independent of AE model selection.",
            "The source classifier historically saw some AE-calibration rows; AE did not. Future arms exclude all calibration rows.",
            "Batch size may differ from historical 4096; it is identical across arms. SupCon is batch-dependent.",
            "Tail statistics may fail nominal region ESS even if raw row counts are large. No silent lowering of requirements.",
        ],
    }
    (args.output_dir / "pilot_manifest.json").write_text(report_json(manifest))
    statistics = []
    for role, rows in (
        ("full_train", eligible),
        ("tail_train", tail),
        ("AE_calibration", calibration),
        ("selected_calibration", selected_cal),
    ):
        for label in range(8):
            subset = rows[labels[rows] == label]
            statistics.append(
                {"role": role, "label": label, **weight_stats(weights[subset])}
            )
    with (args.output_dir / "class_statistics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(statistics[0]))
        writer.writeheader()
        writer.writerows(statistics)
    print(
        f"Frozen AE threshold: {cut:.8g}; calibration rows: {len(calibration):,}; selected: {len(selected_cal):,}",
        flush=True,
    )
    print(
        "All eight positive-weight classes present; class_statistics.csv records physical ESS before training.",
        flush=True,
    )
    np.savez_compressed(
        args.output_dir / "pilot_roles.npz",
        full_train=eligible,
        tail_train=tail,
        calibration=calibration,
        ae_gradient_train=ae_train,
        selected_calibration=selected_cal,
        source_validation=val,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, indices in (("full", eligible), ("tail", tail)):
        print(
            f"=== Pilot {name}: {len(indices):,} events; exactly {args.updates} encoder updates ===",
            flush=True,
        )
        set_random_seed(args.seed)
        model = model_from_config(config, raw["pf"].shape[1], device)
        model.load_state_dict(source["state_dict_model"])
        critic = HLTCritic(config["latent_dim"], 8).to(device)
        critic.load_state_dict(source["state_dict_critic"])
        effective, balance = fit_joint_balance(
            torch.from_numpy(labels[indices]),
            torch.from_numpy(scores[indices]),
            torch.from_numpy(weights[indices]),
            1,
            "log_fixed",
            0.995,
        )
        dataset = SafeDataset(
            raw["pf"],
            torch.from_numpy(labels),
            torch.from_numpy(nuisance),
            torch.from_numpy(scores),
            effective,
            torch.from_numpy(weights),
            torch.as_tensor(indices),
            "pilot_train",
        )
        model_opt = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=config["weight_decay"]
        )
        critic_opt = torch.optim.Adam(
            critic.parameters(),
            lr=config["critic_lr"],
            weight_decay=config["critic_weight_decay"],
        )
        branch = args.output_dir / name
        branch.mkdir()
        (branch / "training_weights.json").write_text(
            report_json(
                {
                    "method": "class-only physics weights, 0.995 per-class clipping; fitted separately on each training population",
                    "effective_statistics": weight_stats(effective.numpy()),
                    "class_mass": {
                        i: float(
                            effective[labels[indices] == i].sum() / effective.sum()
                        )
                        for i in range(8)
                    },
                }
            )
        )
        history = []
        completed = 0
        while completed < args.updates:
            count = min(args.report_every, args.updates - completed)
            loader = FixedBatches(
                dataset, args.batch_size, count, args.seed + completed
            )
            metrics = train_epoch(
                model,
                critic,
                loader,
                model_opt,
                critic_opt,
                device,
                argparse.Namespace(**config),
                SupConLoss(config["contrast_temp"]),
                True,
                1.0,
            )
            if not all(
                np.isfinite(metrics[k])
                for k in (
                    "weighted_ce",
                    "total_loss",
                    "information_penalty",
                    "contrastive_loss",
                )
            ):
                raise RuntimeError(
                    f"{name} training diverged; see partial history, source checkpoints are preserved"
                )
            completed += count
            history.append(
                {
                    "encoder_updates": completed,
                    "critic_updates": completed * config["critic_steps"],
                    "train": metrics,
                }
            )
            print(
                f'{name} update {completed}/{args.updates}: CE={metrics["weighted_ce"]:.4f} BA={metrics["balanced_accuracy"]:.4f}',
                flush=True,
            )
            (branch / "training_history.json").write_text(report_json(history))
            if completed == args.updates or completed % args.save_every == 0:
                torch.save(
                    {
                        "state_dict_model": model.state_dict(),
                        "state_dict_critic": critic.state_dict(),
                        "config": config,
                        "preprocessing": pre,
                        "ae_scaler": source["ae_scaler"],
                        "ae_checkpoint_sha256": ae_hash,
                        "pilot_protocol": "fixed_budget_full_vs_tail",
                        "pilot_manifest": manifest,
                        "branch": name,
                        "encoder_updates": completed,
                        "train_indices": torch.as_tensor(indices),
                        "train_balance_spec": balance,
                        "nurd_active": True,
                        "effective_lambda_info": 1.0,
                        "classification_head_trained": True,
                        "checkpoint_role": "fixed_budget_final"
                        if completed == args.updates
                        else "periodic_update",
                        "source_selected_epoch": source["epoch"],
                    },
                    branch
                    / (
                        "checkpoint_final.pth.tar"
                        if completed == args.updates
                        else f"checkpoint_update_{completed:06d}.pth.tar"
                    ),
                )
        del model, critic, model_opt, critic_opt, dataset
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(
        "Both branches finished. Opening common checking sample for fixed-budget report only.",
        flush=True,
    )
    check = load_hlt_sample(args.heldout)
    check_weights, check_meta = load_generator_weights(
        str(args.heldout_weights),
        check["label"],
        sample=check,
        qcd_label=1,
        weight_scope="all_events",
    )
    check_weights = check_weights.numpy()
    check_labels = checkpoint_target_labels(check, pre).numpy()
    all_ids = np.concatenate((raw["event_id"].numpy(), check["event_id"].numpy()))
    if len(np.unique(all_ids, axis=0)) != len(all_ids):
        raise ValueError(
            "Training/calibration and checking event IDs overlap or duplicate"
        )
    with np.load(cache_path, allow_pickle=False) as cache:
        if not np.array_equal(cache["heldout_event_id"], check["event_id"].numpy()):
            raise ValueError("Checking AE cache IDs differ")
        check_scores = cache["baseline_heldout_scores"].copy()
    if len(check_scores) != len(check_weights) or not np.isfinite(check_scores).all():
        raise ValueError("Invalid checking AE cache")
    check_rows = np.flatnonzero(check_scores >= cut)
    print(
        f"Checking selection: {len(check_rows):,} rows; physical retention {check_weights[check_rows].sum()/check_weights.sum():.3%}",
        flush=True,
    )
    reference_rows = eligible[labels[eligible] == 1]
    results = {}
    checking_arrays = {}
    codes = {}
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir()
    for name, path in (
        ("source", args.checkpoint),
        ("full", args.output_dir / "full/checkpoint_final.pth.tar"),
        ("tail", args.output_dir / "tail/checkpoint_final.pth.tar"),
    ):
        print(f"Frozen pilot evaluation: {name}", flush=True)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = model_from_config(config, raw["pf"].shape[1], device)
        model.load_state_dict(payload["state_dict_model"])
        model.eval()
        critic = HLTCritic(config["latent_dim"], 8).to(device)
        critic.load_state_dict(payload["state_dict_critic"])
        critic.eval()
        result, arrays, c, grid, percentiles = evaluate(
            model,
            critic,
            raw,
            check,
            selected_cal,
            check_rows,
            scores,
            check_scores,
            labels,
            check_labels,
            weights,
            check_weights,
            reference_rows,
            metric_spec,
            config,
            device,
            args.batch_size,
        )
        results[name] = result
        checking_arrays[name] = arrays
        codes[name] = c
        qrows = check_rows[check_labels[check_rows] == 1]
        np.savez_compressed(
            args.output_dir / f"{name}_checking_QCD.npz",
            event_id=check["event_id"][qrows].numpy(),
            ae_score=arrays["ae"],
            md_score=arrays["md"],
            physics_weight=arrays["weight"],
            region_code=c,
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist2d(
            np.log10(arrays["ae"]),
            np.log10(np.maximum(arrays["md"], 1e-12)),
            weights=arrays["weight"],
            bins=80,
        )
        point = result["matched_point"]
        ax.axvline(np.log10(point["ae_cut"]), c="red", ls="--")
        ax.axhline(np.log10(max(point["md_cut"], 1e-12)), c="red", ls="--")
        ax.set(
            xlabel="log10 AE loss",
            ylabel="log10 QCD MD²",
            title=f"{name}: fixed calibration cuts, checking QCD",
        )
        fig.tight_layout()
        fig.savefig(plot_dir / f"{name}_checking_cuts.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 6))
        values = np.where(grid["valid"], np.abs(grid["ratio"] - 1) * 100, np.nan)
        image = ax.pcolormesh(
            percentiles, percentiles, values.T, shading="auto", vmin=0, vmax=100
        )
        fig.colorbar(image, ax=ax, label="|Predicted/observed - 1| (%)")
        ax.set(
            xlabel="Calibration tail AE percentile",
            ylabel="Calibration tail MD percentile",
            title=f"{name}: checking grid (not used for selection)",
        )
        if not grid["valid"].any():
            ax.text(
                0.5,
                0.5,
                "No cells pass nominal region ESS",
                ha="center",
                transform=ax.transAxes,
            )
        fig.tight_layout()
        fig.savefig(plot_dir / f"{name}_checking_grid.png", dpi=160)
        plt.close(fig)
        del model, critic
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    try:
        bootstrap = paired_bootstrap(
            checking_arrays["full"],
            checking_arrays["tail"],
            codes["full"],
            codes["tail"],
            repeats=500,
            seed=args.seed,
        )
        bootstrap["available"] = True
    except ValueError as error:
        bootstrap = {
            "available": False,
            "reason": str(error),
            "warning": "Do not interpret a collapsed or undefined ABCD point as improved closure.",
        }
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, result) in enumerate(results.items()):
        point = result["matched_point"]
        if np.isfinite(point["predicted_over_observed"]) and np.isfinite(
            point["fixed_cut_ratio_uncertainty"]
        ):
            ax.errorbar(
                i,
                point["predicted_over_observed"],
                yerr=point["fixed_cut_ratio_uncertainty"],
                fmt="o",
            )
    ax.axhline(1, c="black", ls="--")
    ax.set_xticks(range(3), list(results))
    ax.set_ylabel("Predicted / observed QCD (fixed calibration cuts)")
    fig.tight_layout()
    fig.savefig(plot_dir / "matched_point_comparison.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, (name, result) in zip((-0.25, 0, 0.25), results.items()):
        perclass = result["classification_common_weights"]["per_class_accuracy"]
        ax.bar(
            np.arange(8) + offset,
            [perclass.get(i, perclass.get(str(i), 0)) for i in range(8)],
            width=0.25,
            label=name,
        )
    ax.set_xticks(
        range(8),
        [
            pre["target_class_names"].get(i, pre["target_class_names"].get(str(i)))
            for i in range(8)
        ],
        rotation=25,
    )
    ax.set_ylabel("Checking per-class accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "classification_comparison.png", dpi=160)
    plt.close(fig)
    if (
        file_sha256(args.checkpoint) != source_hash
        or file_sha256(args.ae_checkpoint) != ae_hash
        or file_sha256(cache_path) != cache_hash
    ):
        raise ValueError("An original input changed during the pilot")
    report = {
        **manifest,
        "results": results,
        "tail_minus_full_paired_bootstrap": bootstrap,
        "source_checkpoints_unchanged": True,
        "checking_selection": retention(check_scores, check_weights, cut),
        "checking_QCD_selection": retention(
            check_scores[check_labels == 1], check_weights[check_labels == 1], cut
        ),
        "checking_signature": sample_signature(check),
        "checking_physics_weight_sha256": check_meta["effective_physics_weight_sha256"],
        "note": "Primary point uses identical AE cut and matched 10% QCD A-efficiency within the AE tail on calibration. "
        "PCA/MD reference population is the same full eligible training QCD for all models, refit in each latent space. "
        "Bootstrap fixes calibration cuts; it omits calibration uncertainty and retraining/systematic variation. "
        "Checking results never choose an epoch or threshold. No certified closure if region ESS fails.",
    }
    (args.output_dir / "pilot_comparison.json").write_text(
        report_json(finite_report(report))
    )
    print(
        report_json(finite_report({"results": results, "paired_bootstrap": bootstrap})),
        flush=True,
    )
    print(f"PAIRED PILOT COMPLETE: {args.output_dir}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "checkpoint",
        "ae-checkpoint",
        "sample",
        "weights",
        "ae-cache-dir",
        "heldout",
        "heldout-weights",
        "output-dir",
    ):
        p.add_argument(f"--{name}", required=True, type=Path)
    p.add_argument("--updates", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--report-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--code-commit", default="unspecified")
    args = p.parse_args()
    if (
        min(args.updates, args.report_every, args.save_every) < 1
        or args.batch_size < 2
        or not np.isfinite(args.lr)
        or args.lr <= 0
    ):
        p.error(
            "Positive budgets, batch size >= 2 and finite positive learning rate required"
        )
    run(args)


if __name__ == "__main__":
    main()
