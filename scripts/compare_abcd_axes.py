"""Run the ordinary full-sample evaluator with QCD MD and 1-P(QCD).

No training, AE-tail selection, conditional calibration, or changes to the
standard validation-selected working point / held-out reporting protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


MODES = {"qcd_md": "qcd_mahalanobis",
         "one_minus_pqcd": "one_minus_qcd_probability"}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluator_command(args, mode, repo):
    if mode not in MODES:
        raise ValueError(f"Unknown axis mode: {mode}")
    command = [
        sys.executable, str(repo / "eval_abcd_nurd.py"),
        "--ckpt", str(args.checkpoint), "--ae_ckpt", str(args.ae_checkpoint),
        "--test_pt", str(args.heldout),
        "--gen_weight_path", str(args.heldout_weights),
        "--reference_pt", str(args.reference),
        "--reference_weight_path", str(args.reference_weights),
        "--target_scheme", "decay8", "--qcd_label", "1", "--n_pca", "6",
        "--statistically_valid_closure",
        "--outdir", str(args.output_dir / mode / "held-out"),
        "--wandb_run_name", f"{args.output_dir.name}_{mode}",
    ]
    if mode == "one_minus_pqcd":
        command.append("--axis2_logit")
    if args.signal is not None:
        command.extend(["--signal_pt", str(args.signal)])
    return command


def verify_same_population(evaluations):
    """Fail rather than silently compare different masks or physical measures."""
    left = evaluations["qcd_md"]["diagnostics"]
    right = evaluations["one_minus_pqcd"]["diagnostics"]
    for mode, evaluation in evaluations.items():
        diagnostic = evaluation["diagnostics"]
        if diagnostic.get("axis2_score") != MODES[mode]:
            raise ValueError(f"Incorrect axis definition for {mode}")
        if diagnostic.get("evaluation_protocol") != "saved_training_validation_split":
            raise ValueError("Comparison requires validation-selected, held-out evaluation")
        if not diagnostic.get("weighted_qcd"):
            raise ValueError("Original physics weights are required in both evaluations")
        statistics = diagnostic["statistical_filter"]
        if not statistics["enabled"] or statistics["count_basis"] != "effective_count":
            raise ValueError("Both axes require effective-count statistical filtering")
        audit = diagnostic["population_audit"]
        if audit["report_weight_scope"] != "all_events":
            raise ValueError("Comparison requires the original all-event physics payload")
        if any(value is None for value in audit.values()):
            raise ValueError("Incomplete event/weight/split provenance")
    for field in ("reference_sample", "report_sample", "qcd_events",
                  "source_target_scheme", "report_label_scheme", "population_audit"):
        if left[field] != right[field]:
            raise ValueError(f"Axis evaluations differ in {field}; not an identical-event comparison")
    if left["statistical_filter"]["minimums"] != right["statistical_filter"]["minimums"]:
        raise ValueError("Statistical minimums differ")
    # Independent conservation check across each mode's four disjoint regions.
    totals = []
    for diagnostic in (left, right):
        regions = diagnostic["working_point"]["regions"]
        totals.append(tuple(sum(regions[r][key] for r in "ABCD")
                            for key in ("raw_count", "yield", "sumw2")))
        if totals[-1][0] != diagnostic["qcd_events"]:
            raise ValueError("ABCD regions do not conserve the QCD event count")
    if any(not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-8)
           for a, b in zip(*totals)):
        raise ValueError("Total QCD counts/physics yields/sumw2 differ between axes")


def comparison_row(mode, evaluation):
    diagnostic = evaluation["diagnostics"]
    point = diagnostic["working_point"]
    a, b, c, d = (float(point[r]) for r in "ABCD")
    prediction = b * c / d
    ratio = prediction / a
    relative_variance = sum(
        point["regions"][r]["sumw2"] / point[r] ** 2 for r in "ABCD")
    return {
        "axis2": mode, "qcd_events": diagnostic["qcd_events"],
        "ae_cut": evaluation["thresholds"]["t1"],
        "axis2_cut": evaluation["thresholds"]["t2"],
        "validation_absolute_nonclosure": abs(point["selection_nonclosure"]),
        "heldout_predicted_over_observed": ratio,
        "heldout_ratio_uncertainty": abs(ratio) * math.sqrt(relative_variance),
        "heldout_absolute_nonclosure": point["absolute_nonclosure"],
        "heldout_A_physics_efficiency": a / (a + b + c + d),
        "working_point_valid": point["statistically_valid"],
        "grid_points": diagnostic["heldout_grid"]["points"],
        "grid_median_absolute_nonclosure": diagnostic["heldout_grid"]["median_absolute_nonclosure"],
        "grid_p90_absolute_nonclosure": diagnostic["heldout_grid"]["p90_absolute_nonclosure"],
        "curve_median_absolute_nonclosure": diagnostic["heldout_diagonal_curve"]["median_absolute_nonclosure"],
    }


def run_comparison(args):
    repo = Path(__file__).resolve().parents[1]
    inputs = {name: getattr(args, name) for name in (
        "checkpoint", "ae_checkpoint", "reference", "reference_weights",
        "heldout", "heldout_weights")}
    if args.signal is not None:
        inputs["signal"] = args.signal
    for name, path in inputs.items():
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Missing {name}: {path}")
    if args.reference.resolve() == args.heldout.resolve():
        raise ValueError("Reference and held-out samples must be distinct")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    commands = {mode: evaluator_command(args, mode, repo) for mode in MODES}
    manifest = {
        "code_commit": args.code_commit,
        "training": False, "ae_tail_selection": False,
        "conditional_calibration": False,
        "protocol": "ordinary_full_sample_validation_selected_heldout_report",
        "inputs": {name: {"path": str(path.resolve()), "sha256": file_hash(path)}
                   for name, path in inputs.items()},
        "commands": commands,
        "comparison_note": (
            "Each axis uses its own usual minimum-nonclosure validation-selected cuts. "
            "This is not a matched-efficiency comparison. Held-out grid and diagonal "
            "summaries are descriptive, not used to select the working point. "
            "Standard plot styling/count-vs-weight conventions are unchanged."),
    }
    (args.output_dir / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2))
    environment = os.environ.copy()
    environment.setdefault("WANDB_MODE", "offline")
    environment["MPLBACKEND"] = "Agg"
    for mode, command in commands.items():
        print(f"\n=== Ordinary evaluation: {mode} ===", flush=True)
        print(shlex.join(command), flush=True)
        subprocess.run(command, check=True, cwd=repo, env=environment)
    if any(file_hash(path) != manifest["inputs"][name]["sha256"]
           for name, path in inputs.items()):
        raise RuntimeError("An input changed during the comparison")
    evaluations = {
        mode: {name: json.loads((args.output_dir / mode / "held-out" / filename).read_text())
               for name, filename in (("diagnostics", "diagnostics.json"),
                                      ("thresholds", "abcd_thresholds.json"))}
        for mode in MODES}
    verify_same_population(evaluations)
    rows = [comparison_row(mode, evaluation) for mode, evaluation in evaluations.items()]
    summary = {**manifest, "identical_population_verified": True,
               "results": rows, "evaluations": evaluations}
    (args.output_dir / "axis_comparison.json").write_text(json.dumps(summary, indent=2))
    with (args.output_dir / "axis_comparison.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nAxis                 pred/obs    |NC|       grid median   QCD A efficiency")
    for row in rows:
        median = row["grid_median_absolute_nonclosure"]
        median_text = "n/a" if median is None else f"{median:.2%}"
        print(f"{row['axis2']:20s} {row['heldout_predicted_over_observed']:.4f}      "
              f"{row['heldout_absolute_nonclosure']:.2%}     {median_text:12s}  "
              f"{row['heldout_A_physics_efficiency']:.2%}")
    print(f"Summary: {args.output_dir / 'axis_comparison.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "ae-checkpoint", "reference", "reference-weights",
                 "heldout", "heldout-weights", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--signal", type=Path)
    parser.add_argument("--code-commit", default="unspecified")
    run_comparison(parser.parse_args())


if __name__ == "__main__":
    main()
