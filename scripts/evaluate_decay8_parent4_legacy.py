"""Opt-in decay8-checkpoint domain transfer to legacy Mequinna parent4 samples.

The network still has eight logits. Only the report view aggregates W/Z decay
probabilities into their parents; the QCD-only MD reference is refit on legacy
training QCD, with a disjoint legacy-validation threshold selection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def aggregate_parent4(logits):
    if logits.ndim != 2 or logits.shape[1] != 8:
        raise ValueError("Expected eight decay8 logits")
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    return np.stack((probabilities[:, [0, 6, 7]].sum(axis=1),
                     probabilities[:, 1], probabilities[:, 2],
                     probabilities[:, [3, 4, 5]].sum(axis=1)), axis=1)


def main():
    from eval_abcd_nurd import (  # Imported only for actual inference; unit tests need no W&B.
        ABCD, embed_pf, load_nurd_model, load_physics_weights)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ae-checkpoint", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-weights", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--test-weights", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--signal", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"Refusing to overwrite {args.output_dir}")
    for path in (args.checkpoint, args.ae_checkpoint, args.reference,
                 args.reference_weights, args.test, args.test_weights):
        if not Path(path).is_file():
            parser.error(f"Missing input: {path}")
    os.environ.setdefault("WANDB_MODE", "offline")
    config = {
        "ckpt": args.checkpoint, "ae_ckpt": args.ae_checkpoint,
        "reference_pt": args.reference,
        "reference_weight_path": args.reference_weights,
        "test_pt": args.test, "gen_weight_path": args.test_weights,
        "signal_pt": args.signal, "outdir": str(args.output_dir),
        "qcd_label": 1, "n_pca": 6,
        "external_parent4_view": True,
        "external_reference_seed": args.seed,
        "external_val_fraction": .2,
        "statistically_valid_closure": True,
        "wandb_run_name": args.output_dir.name,
    }
    ABCD(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_nurd_model(args.checkpoint, device)
    if checkpoint.get("preprocessing", {}).get("target_scheme") != "decay8":
        raise ValueError("Expected decay8 checkpoint")
    _, logits, labels = embed_pf(model, args.test, device, return_logits=True)
    if not set(np.unique(labels)).issubset({0, 1, 2, 3}):
        raise ValueError("Legacy test labels are not parent4")
    probabilities = aggregate_parent4(logits)
    prediction = probabilities.argmax(axis=1)
    weights, metadata, _ = load_physics_weights(args.test_weights, args.test, 1)
    confusion = np.zeros((4, 4), dtype=np.int64)
    for truth, pred in zip(labels, prediction):
        confusion[int(truth), int(pred)] += 1
    per_class = np.divide(np.diag(confusion), confusion.sum(axis=1),
                          out=np.zeros(4), where=confusion.sum(axis=1) > 0)
    report = {
        "source_checkpoint_scheme": "decay8",
        "report_scheme": "parent4",
        "aggregation": {"0_Z": [0, 6, 7], "1_QCD": [1],
                        "2_ttbar": [2], "3_W": [3, 4, 5]},
        "n_events": len(labels), "balanced_accuracy": float(per_class.mean()),
        "weighted_accuracy": float(weights[prediction == labels].sum() / weights.sum()),
        "per_class_accuracy": per_class.tolist(),
        "confusion_true_by_predicted": confusion.tolist(),
        "weight_metadata": metadata,
        "caveat": "Legacy Mequinna tensors have no composite event IDs or decay categories; this is a domain-transfer diagnostic, not a matched train3M held-out comparison.",
    }
    (args.output_dir / "parent4_view.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(args.output_dir / "parent4_predictions.npz",
                        row_index=np.arange(len(labels)), label=labels,
                        parent4_probabilities=probabilities.astype(np.float32),
                        physics_weight=weights)
    figure, ax = plt.subplots(figsize=(5.2, 4.5))
    image = ax.imshow(confusion, norm=matplotlib.colors.LogNorm(vmin=1))
    ax.set(xticks=range(4), yticks=range(4), xlabel="Predicted parent",
           ylabel="True parent", title="Decay8 logits summed to parent4")
    ax.set_xticklabels(["Z", "QCD", "tt", "W"])
    ax.set_yticklabels(["Z", "QCD", "tt", "W"])
    figure.colorbar(image, ax=ax, label="Raw events")
    figure.tight_layout()
    figure.savefig(args.output_dir / "plots/parent4_confusion.png", dpi=180)
    plt.close(figure)
    print("LEGACY PARENT4 VIEW DONE:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
