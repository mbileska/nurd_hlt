"""Write a compact manifest for one held-out plus legacy QCD evaluation job."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def protocol_summary(directory):
    diagnostics = load_json(directory / "diagnostics.json")
    thresholds = load_json(directory / "abcd_thresholds.json")
    selection = diagnostics["abcd_selection"]
    correlations = diagnostics.get("correlations", {}).get("qcd", {})
    return {
        "evaluation_protocol": thresholds["evaluation_protocol"],
        "closure_mode": thresholds["closure_mode"],
        "nonclosure": thresholds["nonclosure"],
        "predicted_over_true": thresholds["ratio"],
        "p1": thresholds["p1"],
        "p2": thresholds["p2"],
        "A": thresholds["report_A"],
        "B": thresholds["report_B"],
        "C": thresholds["report_C"],
        "D": thresholds["report_D"],
        "grid": diagnostics.get("abcd_grid"),
        "closure_curve": diagnostics.get("closure_curve"),
        "qcd_correlations": correlations,
        "tune_rows": selection["tune_n"],
        "report_rows": selection["report_n"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--ae-ckpt", required=True, type=Path)
    parser.add_argument("--nurd-exp", required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()

    payload = {
        "contract_version": 1,
        "nurd_experiment": args.nurd_exp,
        "code_commit": args.code_commit,
        "nurd_checkpoint": str(args.ckpt.resolve()),
        "nurd_checkpoint_sha256": sha256(args.ckpt),
        "ae_checkpoint": str(args.ae_ckpt.resolve()),
        "ae_checkpoint_sha256": sha256(args.ae_ckpt),
        "primary_result": "held-out",
        "held-out": protocol_summary(args.root / "held-out"),
        "legacy": protocol_summary(args.root / "legacy"),
    }
    output = args.root / "evaluation_summary.json"
    with output.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    print("=== Dual QCD evaluation summary ===")
    for name in ("held-out", "legacy"):
        result = payload[name]
        print(
            f"{name:8s}: nonclosure={100.0 * result['nonclosure']:.2f}% "
            f"pred/true={result['predicted_over_true']:.4f} "
            f"p=({result['p1']:.3f},{result['p2']:.3f})"
        )
    print(f"Summary: {output}")


if __name__ == "__main__":
    main()
