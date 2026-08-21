"""Write and print a compact held-out/legacy closure comparison."""

import argparse
import json
from pathlib import Path


def _load(path):
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def _percent(value):
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--ae-ckpt", required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    evaluations = {}
    for name in ("held-out", "legacy"):
        diagnostics = root / name / "diagnostics.json"
        thresholds = root / name / "abcd_thresholds.json"
        if not diagnostics.is_file() or not thresholds.is_file():
            raise FileNotFoundError(
                f"Incomplete {name} evaluation under {root}.")
        evaluations[name] = {
            "diagnostics": _load(diagnostics),
            "thresholds": _load(thresholds),
        }

    summary = {
        "code_commit": args.code_commit,
        "nurd_checkpoint": args.ckpt,
        "ae_checkpoint": args.ae_ckpt,
        "evaluations": evaluations,
    }
    output_path = root / "evaluation_summary.json"
    with output_path.open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)

    print("=== Closure summary ===")
    for name in ("held-out", "legacy"):
        values = evaluations[name]["diagnostics"]
        wp = values["working_point"]["absolute_nonclosure"]
        grid = values["heldout_grid"]
        curve = values["heldout_diagonal_curve"]
        print(
            f"{name:8s} WP={_percent(wp)}  "
            f"grid med/p90={_percent(grid['median_absolute_nonclosure'])}/"
            f"{_percent(grid['p90_absolute_nonclosure'])}  "
            f"curve med/p90={_percent(curve['median_absolute_nonclosure'])}/"
            f"{_percent(curve['p90_absolute_nonclosure'])}")
    print(f"Combined summary: {output_path}")


if __name__ == "__main__":
    main()
