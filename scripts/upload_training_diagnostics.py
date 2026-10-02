#!/usr/bin/env python3
"""Upload an already-completed local diagnostic report to Weights & Biases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def flatten_scalars(value, prefix=""):
    output = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}/{key}" if prefix else str(key)
            output.update(flatten_scalars(child, name))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = value
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Upload a completed diagnostic folder without retraining.")
    parser.add_argument("diagnostic_dir")
    parser.add_argument("--project", default="nurd-ood-hlt-diagnostics")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args(argv)

    root = Path(args.diagnostic_dir).resolve()
    summary_path = root / "diagnostic_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    import wandb

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name or root.name,
        config={
            "diagnostic_protocol": summary.get("protocol"),
            "heldout_sample_accessed": summary.get("heldout_sample_accessed"),
            "source_directory": str(root),
        },
    )
    scalar_metrics = flatten_scalars({
        "weights": summary.get("weights", {}),
        "stages": summary.get("stages", {}),
        "critic_probe_before_nurd": summary.get("critic_probe_before_nurd", {}),
        "critic_probe_after_nurd": summary.get("critic_probe_after_nurd", {}),
        "coupled_critic": summary.get("coupled_critic", {}),
        "comparison": summary.get("comparison", {}),
    })
    run.log(scalar_metrics)
    for path in sorted((root / "plots").rglob("*.png")):
        relative = path.relative_to(root).with_suffix("").as_posix()
        run.log({f"plots/{relative}": wandb.Image(str(path))})
    artifact = wandb.Artifact(f"{root.name}-report", type="diagnostic-report")
    artifact.add_file(str(summary_path))
    arrays_path = root / "diagnostic_arrays.npz"
    if arrays_path.is_file():
        artifact.add_file(str(arrays_path))
    run.log_artifact(artifact)
    run.finish()
    print(f"Uploaded {root} to project {args.project}", flush=True)


if __name__ == "__main__":
    main()
