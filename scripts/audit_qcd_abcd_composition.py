"""Describe the *fixed* held-out ABCD working point by QCD hard-pT bin.

This is a report-only audit: thresholds come from an existing evaluation, and
the QCD MD reference comes from the checkpoint's saved training split. It does
not fit thresholds or select a new checkpoint on the held-out sample.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.hlt_data import load_hlt_sample  # noqa: E402
from utils.hlt_targets import checkpoint_target_labels  # noqa: E402
from utils.hlt_weights import file_sha256  # noqa: E402


CATEGORY_NAMES = {
    0: "15-50", 1: "50-100", 2: "100-200", 3: "200-400",
    4: "400-800", 5: ">800",
}
REGIONS = ("A", "B", "C", "D")


def region_codes(ae_score, md_score, ae_threshold, md_threshold):
    """A/B/C/D are high-high/high-low/low-high/low-low, respectively."""
    high_ae = np.asarray(ae_score) > ae_threshold
    high_md = np.asarray(md_score) > md_threshold
    return np.where(high_ae, np.where(high_md, 0, 1),
                    np.where(high_md, 2, 3)).astype(np.uint8)


def region_statistics(codes, weights, categories):
    """Independent weighted counts, sumw2, and composition by category."""
    codes = np.asarray(codes)
    weights = np.asarray(weights, dtype=np.float64)
    categories = np.asarray(categories)
    if not (len(codes) == len(weights) == len(categories)):
        raise ValueError("Region, weight, and category arrays differ in length")
    if not np.isin(codes, range(4)).all():
        raise ValueError("Invalid region code")
    if not np.isin(categories, tuple(CATEGORY_NAMES)).all():
        raise ValueError("Unexpected QCD hard-pT category")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Physics weights must be finite and nonnegative")

    rows = []
    for code, region in enumerate(REGIONS):
        region_mask = codes == code
        region_yield = float(weights[region_mask].sum())
        region_raw = int(region_mask.sum())
        for category, name in CATEGORY_NAMES.items():
            mask = region_mask & (categories == category)
            selected = weights[mask]
            yield_ = float(selected.sum())
            sumw2 = float(np.square(selected).sum())
            category_mask = categories == category
            category_yield = float(weights[category_mask].sum())
            rows.append({
                "region": region,
                "category": category,
                "hard_pt_gev": name,
                "raw_count": int(mask.sum()),
                "weighted_yield": yield_,
                "sumw2": sumw2,
                "effective_count": yield_**2 / sumw2 if sumw2 else 0.0,
                "fraction_of_region_weight": yield_ / region_yield if region_yield else 0.0,
                "fraction_of_region_raw": int(mask.sum()) / region_raw if region_raw else 0.0,
                "fraction_of_category_weight": yield_ / category_yield if category_yield else 0.0,
            })
    return rows


def infer_latents(model, sample, indices, device, batch_size):
    latents = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_ids = indices[start:start + batch_size]
            pf = torch.nan_to_num(sample["pf"][batch_ids].float(),
                                  nan=0.0, posinf=0.0, neginf=0.0).to(device)
            latent, _ = model(pf)
            latents.append(latent.cpu().numpy())
    if not latents:
        raise ValueError("No QCD events available for latent inference")
    return np.concatenate(latents)


def infer_ae_scores(ae, scaler, sample, indices, device, batch_size):
    # Match eval_abcd_nurd.compute_ae_scores batch arithmetic.
    mu = scaler["mu"].cpu().numpy()
    std = scaler["std"].cpu().numpy()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_ids = indices[start:start + batch_size]
            obj = torch.nan_to_num(
                sample["obj"][batch_ids, :, :4].reshape(len(batch_ids), -1).float(),
                nan=0.0, posinf=0.0, neginf=0.0).numpy()
            normalized = torch.from_numpy(
                ((obj - mu) / (std + 1e-8)).astype(np.float32)).to(device)
            reconstruction, _ = ae(normalized)
            scores.append(((reconstruction - normalized)**2).mean(dim=1).cpu().numpy())
    return np.concatenate(scores).astype(np.float32)


def verify_report(rows, report_regions, *, tolerance=1e-5):
    """Refuse to interpret categories if total region yields do not close."""
    for region in REGIONS:
        subset = [row for row in rows if row["region"] == region]
        got_count = sum(row["raw_count"] for row in subset)
        got_yield = sum(row["weighted_yield"] for row in subset)
        got_sumw2 = sum(row["sumw2"] for row in subset)
        expected = report_regions[region]
        if got_count != int(expected["raw_count"]):
            raise ValueError(
                f"{region} raw count mismatch: recomputed {got_count}, "
                f"saved {expected['raw_count']}; check scores/thresholds")
        if not np.isclose(got_yield, expected["yield"], rtol=tolerance, atol=1e-5):
            raise ValueError(
                f"{region} weighted yield mismatch: recomputed {got_yield}, "
                f"saved {expected['yield']}; check scores/weights")
        if "sumw2" in expected and not np.isclose(
                got_sumw2, expected["sumw2"], rtol=tolerance, atol=1e-5):
            raise ValueError(
                f"{region} sumw2 mismatch: recomputed {got_sumw2}, "
                f"saved {expected['sumw2']}; check scores/weights")


def plot_composition(rows, path):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, key, title in (
        (axes[0], "fraction_of_region_weight", "Physics-weighted yield"),
        (axes[1], "fraction_of_region_raw", "Raw event count"),
    ):
        bottom = np.zeros(4)
        for category, name in CATEGORY_NAMES.items():
            fractions = np.array([
                next(row[key] for row in rows
                     if row["region"] == region and row["category"] == category)
                for region in REGIONS
            ])
            ax.bar(REGIONS, fractions, bottom=bottom, label=name)
            bottom += fractions
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.set_xlabel("ABCD region")
    axes[0].set_ylabel("QCD hard-$p_T$ bin fraction")
    axes[1].legend(title="Hard $p_T$ [GeV]", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    # The full evaluator imports W&B; keep the independent region arithmetic
    # importable in lighter local test environments.
    from eval_abcd_nurd import (
        _fit_class_transform,
        checkpoint_reference_indices,
        load_ae,
        load_nurd_model,
        load_physics_weights,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ae-checkpoint", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-weights", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--heldout-weights", required=True)
    parser.add_argument("--evaluation-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--latent-batch-size", type=int, default=512)
    parser.add_argument("--ae-batch-size", type=int, default=4096)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        parser.error(f"Refusing to overwrite existing output directory: {output_dir}")
    if Path(args.reference).resolve() == Path(args.heldout).resolve():
        parser.error("MD-fit reference and held-out report sample must differ")

    saved = json.loads(Path(args.evaluation_summary).read_text())[
        "evaluations"]["held-out"]
    if saved["diagnostics"]["evaluation_protocol"] != "saved_training_validation_split":
        raise ValueError("Expected saved-training-split held-out evaluation")
    if not saved["diagnostics"]["weighted_qcd"]:
        raise ValueError("Expected physics-weighted QCD evaluation")
    thresholds = saved["thresholds"]
    if int(thresholds["n_pca"]) != 6:
        raise ValueError("This audit expects the six-component QCD MD axis")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_nurd_model(args.checkpoint, device)
    if file_sha256(args.ae_checkpoint) != checkpoint.get("ae_checkpoint_sha256"):
        raise ValueError("AE checkpoint SHA differs from the classifier checkpoint")
    preprocessing = checkpoint["preprocessing"]
    if int(preprocessing.get("qcd_label", 1)) != 1:
        raise ValueError("Expected QCD target label 1")
    source_qcd = int(preprocessing.get("source_parent_qcd_label", 1))

    print("Loading saved training split and QCD physics weights...", flush=True)
    reference = load_hlt_sample(args.reference)
    reference_weights, metadata, signature = load_physics_weights(
        args.reference_weights, args.reference, source_qcd)
    fit_indices, _ = checkpoint_reference_indices(checkpoint, signature, metadata)
    reference_labels = checkpoint_target_labels(reference, preprocessing).numpy()
    fit_qcd = np.sort(fit_indices[reference_labels[fit_indices] == 1])
    print(f"MD fit: {len(fit_qcd)} training QCD events", flush=True)
    reference_latents = infer_latents(
        model, reference, fit_qcd, device, args.latent_batch_size)
    mu, whitening = _fit_class_transform(
        reference_latents, np.ones(len(fit_qcd), dtype=bool), 6, "QCD",
        weights=reference_weights[fit_qcd])
    del reference, reference_latents, reference_weights

    heldout = load_hlt_sample(args.heldout)
    weights, _, _ = load_physics_weights(
        args.heldout_weights, args.heldout, source_qcd)
    heldout_labels = checkpoint_target_labels(heldout, preprocessing).numpy()
    indices = np.flatnonzero(heldout_labels == 1)
    print(f"Held-out QCD: {len(indices)} events", flush=True)
    ae = load_ae(args.ae_checkpoint, checkpoint["ae_scaler"], device)
    ae_score = infer_ae_scores(
        ae, checkpoint["ae_scaler"], heldout, indices, device,
        args.ae_batch_size)
    del ae
    if device == "cuda":
        torch.cuda.empty_cache()
    latents = infer_latents(model, heldout, indices, device,
                            args.latent_batch_size)
    transformed = (latents - mu) @ whitening
    md_score = np.sum(transformed * transformed, axis=1).astype(np.float32)
    keep = np.isfinite(ae_score) & np.isfinite(md_score) & (ae_score > 0)
    if int(keep.sum()) != int(saved["diagnostics"]["qcd_events"]):
        raise ValueError("QCD event count after score mask differs from saved report")
    indices = indices[keep]
    ae_score, md_score = ae_score[keep], md_score[keep]
    categories = heldout["background_category"][indices].numpy()
    event_ids = heldout["event_id"][indices].numpy()
    if np.unique(event_ids, axis=0).shape[0] != len(event_ids):
        raise ValueError("Held-out QCD event IDs are not unique")
    qcd_weights = weights[indices]
    codes = region_codes(ae_score, md_score, thresholds["t1"], thresholds["t2"])
    rows = region_statistics(codes, qcd_weights, categories)
    verify_report(rows, saved["diagnostics"]["working_point"]["regions"])

    output_dir.mkdir(parents=True)
    np.savez_compressed(
        output_dir / "qcd_events.npz", event_id=event_ids,
        background_category=categories, ae_score=ae_score, md_score=md_score,
        physics_weight=qcd_weights, region=np.asarray(REGIONS)[codes])
    with (output_dir / "region_composition.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "checkpoint": args.checkpoint,
        "evaluation_summary": args.evaluation_summary,
        "thresholds": {"ae": thresholds["t1"], "md": thresholds["t2"]},
        "qcd_events": len(indices),
        "global_report_match": True,
        "categories": CATEGORY_NAMES,
        "regions": rows,
    }
    (output_dir / "region_composition.json").write_text(
        json.dumps(report, indent=2))
    plot_composition(rows, output_dir / "region_composition.png")
    print(f"PASS: global A/B/C/D counts and weighted yields match saved evaluation")
    print(f"Output: {output_dir}")
    for region in REGIONS:
        values = [r for r in rows if r["region"] == region]
        print(f"{region}: raw={sum(r['raw_count'] for r in values):,} "
              f"weighted={sum(r['weighted_yield'] for r in values):.6g} "
              f"hardest-bin weighted share={values[-1]['fraction_of_region_weight']:.4%}")


if __name__ == "__main__":
    main()
