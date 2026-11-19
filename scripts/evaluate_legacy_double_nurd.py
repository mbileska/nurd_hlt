"""Golden / fresh-single / fresh-double Mequinna evaluation, full and AE tail.

GPU stage fits one training-QCD MD transform per model and freezes calibration
cuts before opening held-out. CPU stage makes the complete ordinary plot suites.
"""
import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from models.hlt_autoencoder import HLTAutoencoder
from models.hlt_con import HLTContrastiveModel
from scripts.audit_qcd_abcd_composition import infer_latents, infer_ae_scores
from scripts.compare_qcd_score_audit import paired_bootstrap
from scripts.plot_ae_tail_eval_suite import render_suite
from scripts.prepare_tail_pilot import report_json
from scripts.run_legacy_double_nurd import PROTOCOL
from scripts.run_tail_nurd_pilot import finite_report
from scripts.standard_double_nurd_eval import (
    PARENT_NAMES, class_metrics, confusion_plots, selected, standard_plan, standard_check,
)
from scripts.study_tail_pilot_cuts import qcd, recount
from utils.hlt_weights import file_sha256, load_generator_weights, sample_signature

ARMS = ("golden", "single", "double")


def paths(root):
    return {"golden": root/"golden_checkpoint.pth.tar",
            **{arm: root/arm/"checkpoint_main.pth.tar" for arm in ("single", "double")}}


def load_model(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state, cfg = payload["state_dict_model"], payload["config"]
    if state["classifier.weight"].shape[0] != 4: raise ValueError("Native parent4 checkpoint required")
    token_key = next(k for k in state if k.endswith(".attn.e.weight"))
    keys = ("embed_size", "latent_dim", "proj_dim", "num_heads", "num_layers", "dim_ff", "linear_dim", "dropout")
    model = HLTContrastiveModel(num_classes=4, num_tokens=state[token_key].shape[1]-1,
        **{k: cfg[k] for k in keys}).to(device)
    model.load_state_dict(state); model.eval()
    return model, payload


def numpy_data(scores, md, physics, labels, rows):
    return {"ae": scores[rows].numpy(), "md": md, "weight": physics[rows].numpy(),
            "label": labels[rows].numpy(), "row_index": rows.numpy()}


def score(args):
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    prep_path = args.experiment_dir/"preparation_complete.json"
    prep = json.loads(prep_path.read_text())
    if prep["protocol"] != PROTOCOL: raise ValueError("Incorrect preparation protocol")
    if file_sha256(args.experiment_dir/"prepared_recipe.pth.tar") != prep["prepared_sha256"]:
        raise ValueError("Original prepared scores/roles changed")
    prepared = torch.load(args.experiment_dir/"prepared_recipe.pth.tar", map_location="cpu", weights_only=False)
    model_paths = paths(args.experiment_dir)
    hashes = {k: file_sha256(v) for k,v in model_paths.items()}
    for arm in ("single", "double"):
        completion = json.loads((args.experiment_dir/arm/"training_complete.json").read_text())
        if completion["checkpoint_sha256"] != hashes[arm] or completion["initial_sha256"] != prep["initial_sha256"]:
            raise ValueError("Models are not the matched completed trainings")
    if hashes["golden"] != prep["golden_sha256"]: raise ValueError("Golden reference changed")
    raw = torch.load(args.sample, map_location="cpu", weights_only=False)
    if sample_signature(raw) != prepared["data_signature"]: raise ValueError("Mequinna training manifest changed")
    labels, physics = raw["label"].long().reshape(-1), prepared["physics"]
    _, meta = load_generator_weights(str(args.weights), labels, sample=raw, qcd_label=1)
    if meta["effective_physics_weight_sha256"] != prepared["weight_metadata"]["effective_physics_weight_sha256"]:
        raise ValueError("Legacy training weights changed")
    pre = prepared["preprocessing"]
    fit = pre["split"]["train_indices"]; fit = fit[labels[fit] == 1]
    cal = pre["split"]["validation_indices"]; cal = cal[labels[cal] == 1]
    cuts = {"f100": None, "f010": prepared["ae_threshold"]}
    minimums = {"f100": [50,50,50,args.full_d_min], "f010": [50,50,50,args.tail_d_min]}
    args.output_dir.mkdir(parents=True)
    ae_path = args.experiment_dir/"original_checkpoint_ae.pth"
    if file_sha256(ae_path) != prep["AE_sha256"]: raise ValueError("Original AE changed")
    ae = torch.load(ae_path, map_location="cpu", weights_only=False)
    ae_model = HLTAutoencoder(ae["ae_config"]).to(args.device); ae_model.load_state_dict(ae["ae"]); ae_model.eval()
    # Ordinary evaluation arithmetic (including its 1e-8 scaler epsilon),
    # separately from the exact original training-nuisance arithmetic.
    cal_ae = infer_ae_scores(ae_model, prepared["ae_scaler"], raw, cal, args.device, 4096)
    plans = {}; references = {}; training_provenance = {}
    for arm, path in model_paths.items():
        print(f"{arm}: fixed training-QCD reference {len(fit):,}; calibration QCD {len(cal):,}", flush=True)
        model, payload = load_model(path, args.device)
        if payload["ae_checkpoint_sha256"] != prep["AE_sha256"]: raise ValueError("Models use different AE axes")
        training_provenance[arm] = {"selected_epoch":payload["epoch"],
            "checkpoint_role":payload.get("checkpoint_role","original_best_weighted_validation_ce"),
            "epochs_trained":None if arm=="golden" else json.loads((args.experiment_dir/arm/"training_complete.json").read_text())["epochs_trained"]}
        reference = infer_latents(model, raw, fit, args.device, args.batch_size)
        w = physics[fit].numpy(); w = w/w.sum()
        mean = (reference*w[:,None]).sum(0); centered = reference-mean
        eigenvalues, vectors = np.linalg.eigh((centered*w[:,None]).T @ centered)
        whitening = vectors / np.sqrt(np.maximum(eigenvalues, 1e-6))
        references[arm] = {"mean": mean.tolist(), "whitening": whitening.tolist(), "fit_QCD_rows": len(fit)}
        latent = infer_latents(model, raw, cal, args.device, args.batch_size)
        data = numpy_data(prepared["scores"], np.square((latent-mean)@whitening).sum(1), physics, labels, cal)
        data["ae"] = cal_ae
        np.savez_compressed(args.output_dir/f"{arm}_calibration_QCD.npz", **data)
        plans[arm] = {key: standard_plan(selected(data, cut), minimums[key]) for key,cut in cuts.items()}
        del model, payload, reference, latent; gc.collect()
        if args.device == "cuda": torch.cuda.empty_cache()
    frozen = {"protocol": PROTOCOL, "checkpoint_sha256": hashes, "AE_sha256": prep["AE_sha256"],
        "AE_thresholds": cuts, "minimums": minimums, "references": references, "plans": plans,
        "training_provenance": training_provenance,
        "networks_retrained_in_evaluation": False, "checking_used_for_cut_selection": False,
        "selection_policy": "Usual calibration-QCD 50--98% percentile scan, minimum |1/R-1|, then fixed held-out cuts",
        "tail_policy": "Original AE-unseen all-background LEGACY weighted 90th percentile; frozen before new training",
        "weight_policy": "Original legacy generator weights once; QCD weighted, non-QCD unit base weights",
        "identity_limit": "Mequinna has no composite event IDs; row indices are file-local, not event IDs",
        "source_preparation_sha256": file_sha256(prep_path), "code_commit": args.code_commit}
    frozen_path = args.output_dir/"frozen_evaluation_plan.json"
    frozen_path.write_text(report_json(finite_report(frozen)))
    frozen_hash = file_sha256(frozen_path)
    del raw, labels, physics; gc.collect()
    print("All six working-point plans frozen. Opening independent held-out FILE now.", flush=True)
    check = torch.load(args.heldout, map_location="cpu", weights_only=False)
    labels = check["label"].long().reshape(-1)
    if set(labels.tolist()) != set(range(4)): raise ValueError("Held-out must use original four parents")
    if sample_signature(check) == prepared["data_signature"]: raise ValueError("Checking file equals training manifest")
    physics, weight_meta = load_generator_weights(str(args.heldout_weights), labels, sample=check, qcd_label=1)
    scores = torch.from_numpy(infer_ae_scores(ae_model, prepared["ae_scaler"], check, torch.arange(len(labels)), args.device, 4096))
    del ae_model
    np.savez_compressed(args.output_dir/"heldout_common_scores.npz", ae=scores.numpy(), weight=physics.numpy(),
                        label=labels.numpy(), row_index=np.arange(len(labels)))
    for arm, path in model_paths.items():
        model, payload = load_model(path, args.device)
        print(f"{arm}: one full held-out encoder pass, reused for full and top-10% plots", flush=True)
        latent = infer_latents(model, check, torch.arange(len(labels)), args.device, args.batch_size)
        mean = np.asarray(references[arm]["mean"]); whitening = np.asarray(references[arm]["whitening"])
        md = np.square((latent-mean)@whitening).sum(1)
        logits = latent @ payload["state_dict_model"]["classifier.weight"].numpy().T + payload["state_dict_model"]["classifier.bias"].numpy()
        probabilities = np.exp(logits-logits.max(1,keepdims=True)); probabilities /= probabilities.sum(1,keepdims=True)
        np.savez_compressed(args.output_dir/f"{arm}_heldout_predictions.npz", latent=latent, md=md,
                            probabilities=probabilities, row_index=np.arange(len(labels)))
        del model, payload; gc.collect()
        if args.device == "cuda": torch.cuda.empty_cache()
    if file_sha256(frozen_path) != frozen_hash: raise ValueError("Frozen plan changed")
    if {k:file_sha256(v) for k,v in model_paths.items()} != hashes: raise ValueError("Models changed during evaluation")
    files = [args.output_dir/"heldout_common_scores.npz", frozen_path,
             *[args.output_dir/f"{arm}_{suffix}.npz" for arm in ARMS for suffix in ("calibration_QCD", "heldout_predictions")]]
    manifest = {"protocol": PROTOCOL, "files": {p.name:file_sha256(p) for p in files},
        "checkpoint_sha256": hashes, "heldout_signature": sample_signature(check), "weights": weight_meta,
        "legacy_limitations": prep["caveats"]}
    (args.output_dir/"inference_complete.json").write_text(report_json(finite_report(manifest)))
    print("LEGACY SCORING COMPLETE; GPU can be released before standard plot rendering.", flush=True)


def plots(args):
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    manifest = json.loads((args.score_dir/"inference_complete.json").read_text())
    for name,digest in manifest["files"].items():
        if file_sha256(args.score_dir/name) != digest: raise ValueError(f"Score cache changed: {name}")
    frozen = json.loads((args.score_dir/"frozen_evaluation_plan.json").read_text())
    if frozen["protocol"] != PROTOCOL: raise ValueError("Wrong legacy protocol")
    with np.load(args.score_dir/"heldout_common_scores.npz", allow_pickle=False) as archive:
        common = {k:archive[k].copy() for k in archive.files}
    args.output_dir.mkdir(parents=True)
    results = {}; checking = {}; table = []
    qcd_mass = float(common["weight"][common["label"]==1].sum())
    for arm in ARMS:
        with np.load(args.score_dir/f"{arm}_heldout_predictions.npz", allow_pickle=False) as archive:
            pred = {k:archive[k].copy() for k in archive.files}
        if not np.array_equal(pred["row_index"], common["row_index"]): raise ValueError("Prediction rows differ")
        probabilities = pred.pop("probabilities")
        data = {**common, **pred}; checking[arm] = data; results[arm] = {}
        reference = {k:np.asarray(frozen["references"][arm][k]) for k in ("mean", "whitening")}
        for key,cut in frozen["AE_thresholds"].items():
            selected_data = selected(data, cut)
            mask = np.ones(len(probabilities),bool) if cut is None else data["ae"] >= cut
            closure, grid = standard_check(qcd(selected_data), frozen["plans"][arm][key], frozen["minimums"][key], qcd_mass)
            closure = finite_report(closure)
            metrics = class_metrics(selected_data["label"], probabilities[mask], selected_data["weight"], 4)
            folder = args.output_dir/arm/key; folder.mkdir(parents=True)
            report = {"scans":{"tail":closure}, "plot_context": {
                "population": f"{arm}, {'full' if cut is None else 'top-10%'} Mequinna held-out (legacy weights)",
                "point_label":"Calibration-selected fixed cuts", "scan_point_label":"Calibration-selected fixed cuts",
                "scan_title":"Standard ABCD scan — held-out (descriptive)",
                "scan_xlabel":"Held-out QCD AE percentile", "scan_ylabel":"Held-out QCD MD percentile",
                "efficiency_xlabel":"Selection Efficiency (QCD A / selected QCD)"}}
            print(f"Render {arm}/{key}: all standard parent4 plots",flush=True)
            style = render_suite(selected_data, PARENT_NAMES, report, grid, reference, folder/"plots", kde_grid=args.kde_grid)
            confusion_plots(metrics, PARENT_NAMES, folder/"plots")
            point = closure["working_point"]
            np.savez_compressed(folder/"closure_grid.npz", **grid)
            value = {"classification":metrics, "closure":closure, "minimums":frozen["minimums"][key], **style,
                "retained_legacy_weight_fraction":float(selected_data["weight"].sum()/common["weight"].sum()),
                "retained_raw_fraction":float(mask.mean())}
            (folder/"evaluation_summary.json").write_text(report_json(finite_report(value)))
            results[arm][key] = value
            table.append({"model":arm,"selection":key,"selected_epoch":frozen["training_provenance"][arm]["selected_epoch"],
                "balanced_accuracy":metrics["balanced_accuracy"],"physics_accuracy":metrics["physics_weighted_accuracy"],
                "ratio":None if point is None else point["predicted_over_observed"],
                "absolute_nonclosure":None if point is None else point["absolute_nonclosure"],
                "working_point_valid":None if point is None else point["statistically_valid"],
                "grid_median_nonclosure":closure["grid_median_absolute_nonclosure"]})
            print(arm,key,"balanced accuracy",metrics["balanced_accuracy"],"R",None if point is None else point["predicted_over_observed"],flush=True)
    paired = {}
    for key,cut in frozen["AE_thresholds"].items():
        arrays = [qcd(selected(checking[arm],cut)) for arm in ("single","double")]
        points = [results[arm][key]["closure"]["working_point"] for arm in ("single","double")]
        if any(p is None for p in points): paired[key] = None; continue
        codes = [recount(a,p["ae_cut"],p["md_cut"],frozen["minimums"][key])[1] for a,p in zip(arrays,points)]
        if not np.array_equal(arrays[0]["row_index"],arrays[1]["row_index"]): raise ValueError("Paired held-out rows differ")
        try: paired[key] = paired_bootstrap(*arrays,*codes,repeats=500,seed=42)
        except ValueError as error: paired[key] = {"available":False,"reason":str(error)}
    old = json.loads(args.golden_summary.read_text())["evaluations"]["held-out"]
    original = old["diagnostics"]; cut = old["thresholds"]
    archived_recount = recount(qcd(checking["golden"]),cut["t1"],cut["t2"],frozen["minimums"]["f100"])[0]
    delta = archived_recount["absolute_nonclosure"]-original["working_point"]["absolute_nonclosure"]
    reproduction = {"archived_absolute_nonclosure":original["working_point"]["absolute_nonclosure"],
        "recount_at_archived_numeric_cuts":archived_recount,
        "delta_absolute_nonclosure":delta, "agrees_within_0p1_percentage_point":abs(delta)<=.001,
        "archived_grid_median":original["heldout_grid"]["median_absolute_nonclosure"],
        "current_grid_median":results["golden"]["f100"]["closure"]["grid_median_absolute_nonclosure"]}
    final = {**frozen,"results":results,"golden_heldout_reproduction":reproduction,
        "paired_double_minus_single_nonclosure":paired,
        "comparison_warning":"Standard selected working points may have different A rates; no signal-sensitivity optimum. Bootstrap omits calibration-selection, retraining and systematics.",
        "source_limitations":manifest["legacy_limitations"]}
    (args.output_dir/"legacy_comparison.json").write_text(report_json(finite_report(final)))
    with (args.output_dir/"legacy_comparison.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    if not reproduction["agrees_within_0p1_percentage_point"]:
        print("WARNING: frozen golden does not reproduce archived numeric cuts; inspect reproduction report before interpreting the new trainings.",flush=True)
    print("LEGACY EVALUATION DONE:",args.output_dir,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("score","plots"))
    for name in ("experiment-dir","output-dir"): parser.add_argument(f"--{name}",type=Path,required=True)
    for name in ("sample","weights","heldout","heldout-weights","score-dir","golden-summary"):
        parser.add_argument(f"--{name}",type=Path)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--batch-size",type=int,default=1024)
    parser.add_argument("--full-d-min",type=int,default=500)
    parser.add_argument("--tail-d-min",type=int,default=100)
    parser.add_argument("--kde-grid",type=int,default=64)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--code-commit",default="")
    args=parser.parse_args()
    if min(args.batch_size,args.full_d_min,args.tail_d_min,args.threads)<1 or args.kde_grid<8: parser.error("Invalid evaluation resources/statistics")
    if args.phase=="score" and any(getattr(args,k) is None for k in ("sample","weights","heldout","heldout_weights")): parser.error("Scoring needs train/test data and weights")
    if args.phase=="plots" and (args.score_dir is None or args.golden_summary is None): parser.error("Plotting needs score-dir and golden-summary")
    torch.set_num_threads(args.threads)
    with threadpool_limits(args.threads):
        if args.phase=="score": score(args)
        else: plots(args)


if __name__=="__main__": main()
