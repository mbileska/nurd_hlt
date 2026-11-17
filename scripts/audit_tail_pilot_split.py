"""Audit frozen pilot preparation: data integrity, split shift, AE membership, uncertainty.

Only the outer training manifest is opened. No PF-encoder inference, training,
held-out scanning, or modification of any split/threshold is performed.
"""
import argparse
import csv
import inspect
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.hlt_autoencoder import HLTAutoencoder
from scripts.prepare_tail_pilot import freeze_selection, infer_scores, validate_split, weight_stats
from utils.hlt_data import load_hlt_sample
from utils.hlt_targets import checkpoint_target_labels
from utils.hlt_weights import file_sha256, load_generator_weights, sample_signature, stratified_split_indices


def quantile(x, w, q):
    positive = w > 0
    x, w = np.asarray(x)[positive], np.asarray(w)[positive]
    if not len(w):
        return np.full(np.asarray(q).shape, np.nan)
    order = np.argsort(x, kind="stable")
    cumulative = np.cumsum(w[order])
    return x[order[np.minimum(np.searchsorted(cumulative, np.asarray(q)*cumulative[-1]), len(x)-1)]]


def retention(scores, weights, cut):
    stats = weight_stats(weights)
    if not stats["sum_weights"]:
        return {**stats, "retention": None, "standard_error_fixed_cut": None, "selected_effective_events": 0.}
    passed = np.asarray(scores) >= cut
    fraction = float(weights[passed].sum()/weights.sum())
    error = np.sqrt(np.sum(weights**2*(passed-fraction)**2))/weights.sum()
    return {**stats, "retention": fraction, "raw_retention": float(passed.mean()), "standard_error_fixed_cut": float(error),
            "selected_effective_events": weight_stats(weights[passed])["effective_events"]}


def weighted_ks(x, w, is_validation):
    """Descriptive weighted KS distance, with correct treatment of ties; no p-value."""
    order = np.argsort(x, kind="stable")
    values, weights, valid = x[order], w[order], is_validation[order]
    train_mass, val_mass = weights[~valid].sum(), weights[valid].sum()
    if min(train_mass, val_mass) <= 0:
        return None
    train_cdf = np.cumsum(np.where(valid, 0., weights))/train_mass
    val_cdf = np.cumsum(np.where(valid, weights, 0.))/val_mass
    ends = np.r_[values[1:] != values[:-1], True]
    return float(np.max(np.abs(train_cdf[ends]-val_cdf[ends])))


def distribution_stats(x, w, is_validation, name):
    result = {"feature": name, "weighted_KS": weighted_ks(x, w, is_validation),
              "raw_KS": weighted_ks(x, np.ones(len(w)), is_validation)}
    for label, mask in (("train", ~is_validation), ("validation", is_validation)):
        values, weights = x[mask], w[mask]
        if weights.sum() <= 0:
            continue
        mean = float(np.dot(values, weights)/weights.sum())
        qs = quantile(values, weights, [.1,.5,.9,.99])
        result.update({f"{label}_mean": mean, f"{label}_raw_mean": float(values.mean()),
                       f"{label}_nonzero_fraction": float(weights[values != 0].sum()/weights.sum()),
                       **{f"{label}_q{q}": float(v) for q,v in zip((10,50,90,99), qs)}})
    return result


def standardized_validation(scores, weights, is_validation, groups, cut):
    """Descriptive common-composition adjustment; report lack of common support."""
    train_mass, val_mass = weights[~is_validation].sum(), weights[is_validation].sum()
    numerator = 0.; unsupported = []
    for group in np.unique(groups):
        train = (~is_validation) & (groups==group)
        val = is_validation & (groups==group)
        fraction = weights[train].sum()/train_mass
        if fraction > 0 and weights[val].sum() <= 0:
            unsupported.append(int(group))
        elif weights[val].sum() > 0:
            numerator += fraction * weights[val & (scores>=cut)].sum()/weights[val].sum()
    return {"validation_retention_at_train_group_composition": None if unsupported else float(numerator),
            "unsupported_train_groups": unsupported,
            "note": "Descriptive composition adjustment only; not a causal estimate or a new event weight for physics evaluation."}


def bootstrap_gap(scores, weights, labels, train, validation, repeats=200, seed=42):
    """Event-level Poisson MC uncertainty, frozen AE; refit training cut each replicate.

    Not a null-hypothesis test, not uncertainty from retraining, and conditional
    on the observed (weight-balanced, not IID random) saved split.
    """
    rng = np.random.default_rng(seed)
    order = train[np.argsort(scores[train], kind="stable")]
    x, w, qcd = scores[order], weights[order], labels[order] == 1
    val_qcd = validation[labels[validation] == 1]
    vx, vw = scores[val_qcd], weights[val_qcd]
    rows = []
    for repeat in range(repeats):
        bw = w*rng.poisson(1, len(w))
        bvw = vw*rng.poisson(1, len(vw))
        if min(bw[qcd].sum(), bvw.sum()) <= 0:
            continue
        cumulative = np.cumsum(bw)
        cut = x[min(np.searchsorted(cumulative, .9*cumulative[-1]), len(x)-1)]
        a = bw[qcd & (x >= cut)].sum()/bw[qcd].sum()
        b = bvw[vx >= cut].sum()/bvw.sum()
        rows.append((cut, a, b, b-a))
        if (repeat+1) % 50 == 0:
            print(f"Bootstrap {repeat+1}/{repeats}", flush=True)
    values = np.asarray(rows)
    if not len(values):
        raise ValueError("No usable bootstrap replicates")
    return values, {"requested_replicates": repeats, "usable_replicates": len(values),
                    "ae_cut_95pct_interval": np.quantile(values[:,0], [.025,.975]).tolist(),
                    "validation_minus_train_retention_95pct_interval": np.quantile(values[:,3], [.025,.975]).tolist(),
                    "delta_median": float(np.median(values[:,3])),
                    "interpretation": "Conditional MC precision with training-threshold refitting; not a significance or causal test."}


def random_resplits(scores, weights, labels, categories, validation, repeats=30, seed=142):
    """Random fine-category-stratified assignments of FROZEN scores, not new validation."""
    rng = np.random.default_rng(seed)
    original_val = np.zeros(len(scores), bool); original_val[validation] = True
    groups = [np.flatnonzero(categories == c) for c in np.unique(categories)]
    sizes = [int(original_val[g].sum()) for g in groups]
    sorted_rows = np.argsort(scores, kind="stable")
    rows = []
    for repeat in range(repeats):
        is_val = np.zeros(len(scores), bool)
        for group, size in zip(groups, sizes):
            is_val[rng.choice(group, size=size, replace=False)] = True
        sw = np.where(is_val[sorted_rows], 0., weights[sorted_rows])
        cumulative = np.cumsum(sw)
        cut = scores[sorted_rows[min(np.searchsorted(cumulative, .9*cumulative[-1]), len(scores)-1)]]
        qcd = labels == 1
        a = retention(scores[qcd & ~is_val], weights[qcd & ~is_val], cut)["retention"]
        b = retention(scores[qcd & is_val], weights[qcd & is_val], cut)["retention"]
        rows.append((cut, a, b, b-a))
    values = np.asarray(rows)
    return values, {"replicates": repeats,
                    "delta_95pct_reference_interval": np.quantile(values[:,3], [.025,.975]).tolist(),
                    "warning": "Repartitions scores from an already-trained AE. Both sides mix AE-seen and unseen rows; "
                               "not a new independent generalization test or a formal null for the original splitter."}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader(); writer.writerows(rows)


def cdf_plot(ax, x, w, label):
    if len(w) and w.sum() > 0:
        # Quantile knots keep figure files small, without modifying any analysis counts.
        qs = np.linspace(0, 1, 501)
        ax.plot(quantile(x, w, qs), qs, label=label)


def audit(args):
    started = time.monotonic()
    prep = json.loads((args.preparation_dir/"preparation_report.json").read_text())
    checkpoint_path = Path(prep["classifier_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ae_payload = torch.load(args.ae_checkpoint, map_location="cpu", weights_only=False)
    for name, actual, expected in (("classifier", file_sha256(checkpoint_path), prep["classifier_checkpoint_sha256"]),
                                   ("AE", file_sha256(args.ae_checkpoint), prep["ae_checkpoint_sha256"])):
        if actual != expected:
            raise ValueError(f"{name} checkpoint hash mismatch")
    sample_path = Path(prep["sample"])
    raw = load_hlt_sample(sample_path)
    signature = sample_signature(raw)
    if signature != prep["data_signature"] or signature != checkpoint["preprocessing"]["data_signature"]:
        raise ValueError("Training sample signature mismatch")
    weights, metadata = load_generator_weights(str(args.weights), raw["label"], sample=raw,
                                               qcd_label=1, weight_scope="all_events")
    if metadata["effective_physics_weight_sha256"] != prep["physics_weight_sha256"]:
        raise ValueError("Physics weights changed since preparation")
    weights = weights.double().numpy()
    with np.load(args.preparation_dir/"pilot_selection.npz", allow_pickle=False) as cache:
        x = cache["ae_score"].copy(); labels = cache["label"].copy()
        train, validation = validate_split(cache["train_indices"], cache["validation_indices"], len(labels))
        if not np.array_equal(cache["event_id"], raw["event_id"].numpy()):
            raise ValueError("Cached event-ID order mismatch")
        if not np.array_equal(cache["physics_weight"], weights):
            raise ValueError("Cached physics weights are misaligned")
        if not np.array_equal(labels, checkpoint_target_labels(raw, checkpoint["preprocessing"]).numpy()):
            raise ValueError("Cached labels mismatch")
        split = checkpoint["preprocessing"]["split"]
        if not np.array_equal(train, split["train_indices"].cpu().numpy()) or not np.array_equal(validation, split["validation_indices"].cpu().numpy()):
            raise ValueError("Cached classifier split differs from checkpoint")
        cut, selected, st, sv = freeze_selection(x, weights, train, validation)
        if cut != prep["ae_threshold"] or not np.array_equal(selected, cache["selected"]) or not np.array_equal(st, cache["selected_train_indices"]) or not np.array_equal(sv, cache["selected_validation_indices"]):
            raise ValueError("Cached threshold/selection cannot be reproduced")
    ids = raw["event_id"].numpy()
    if len(np.unique(ids, axis=0)) != len(ids):
        raise ValueError("Duplicate composite event IDs across training manifest")
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("Invalid cached AE scores")
    torch.set_num_threads(1)
    # Recover AE membership; historical checkpoints did not save AE row indices.
    ae_meta = ae_payload["weighting"]
    if ae_payload["data_signature"] != signature or ae_meta["generator"]["effective_physics_weight_sha256"] != prep["physics_weight_sha256"]:
        raise ValueError("AE training data/weight provenance differs")
    if "split_seed" not in ae_meta or "validation_fraction" not in ae_meta:
        raise ValueError("Cannot reconstruct AE split without saved seed/fraction")
    print("Reconstructing AE and classifier weight-balanced splits...", flush=True)
    ae_train, ae_val = stratified_split_indices(raw["label"], ae_meta["validation_fraction"], ae_meta["split_seed"], weights=torch.from_numpy(weights))
    ct, cv = stratified_split_indices(torch.from_numpy(labels), split["validation_fraction"], split["seed"], weights=torch.from_numpy(weights))
    reproduced = np.array_equal(ct.numpy(), train) and np.array_equal(cv.numpy(), validation)
    _, alternative_val = stratified_split_indices(torch.from_numpy(labels), split["validation_fraction"], split["seed"]+1,
                                                 weights=torch.from_numpy(weights))
    original_qcd_val=validation[labels[validation]==1]
    alternate_qcd_val=alternative_val.numpy()[labels[alternative_val.numpy()]==1]
    seed_overlap={"alternative_seed":int(split["seed"])+1,
                  "original_QCD_validation_events":len(original_qcd_val),
                  "alternative_QCD_validation_events":len(alternate_qcd_val),
                  "fraction_original_QCD_validation_retained":float(np.isin(original_qcd_val,alternate_qcd_val).mean()),
                  "rough_random_assignment_reference":float(len(alternate_qcd_val)/np.sum(labels==1)),
                  "note":"Algorithm diagnostic: sorted unique weights are not randomized by changing the seed."}
    ae_seen = np.zeros(len(x), bool); ae_seen[ae_train.numpy()] = True
    is_val = np.zeros(len(x), bool); is_val[validation] = True
    ae_losses={name:float(np.dot(weights[idx],x[idx])/weights[idx].sum())
               for name,idx in (("reconstructed_training",ae_train.numpy()),("reconstructed_validation",ae_val.numpy()))}
    recorded_loss=ae_payload.get("validation_losses",{}).get("physics")
    ae_losses["recorded_checkpoint_validation_physics_loss"]=recorded_loss
    ae_losses["validation_loss_reconstruction_consistent"]=(
        bool(np.isclose(ae_losses["reconstructed_validation"],recorded_loss,rtol=5e-4,atol=1e-7))
        if recorded_loss is not None else None)
    reconstructed_mass = {}
    mass_matches = True
    for name, indices in (("training", ae_train.numpy()), ("validation", ae_val.numpy())):
        parent = raw["label"].numpy()
        actual = {int(c): float(weights[indices[parent[indices]==c]].sum()/weights[indices].sum())
                  for c in np.unique(parent)}
        expected = ae_meta.get(f"physics_{name}_class_mass")
        reconstructed_mass[name] = actual
        if expected is None or any(not np.isclose(value, expected.get(c,expected.get(str(c),np.nan)),rtol=1e-5,atol=1e-7)
                                   for c,value in actual.items()):
            mass_matches = False
    # Recompute only a small deterministic object batch to check cache and scaler arithmetic.
    rng = np.random.default_rng(args.seed)
    check_rows = np.unique(np.r_[rng.choice(len(x), min(2048,len(x)), replace=False),
                                  np.argsort(weights)[-256:], np.argsort(x)[-256:]])
    ae = HLTAutoencoder(ae_payload["ae_config"]); ae.load_state_dict(ae_payload["ae"]); ae.eval()
    scaler = checkpoint["ae_scaler"]
    mu, std = scaler["mu"].cpu().float(), scaler["std"].cpu().float()
    if not torch.equal(mu, ae_payload["ae_scaler"]["mu"].cpu().float()) or not torch.equal(std, ae_payload["ae_scaler"]["std"].cpu().float()):
        raise ValueError("Classifier does not use the AE checkpoint's scaler")
    rescored = infer_scores(ae, scaler, {"obj":raw["obj"][check_rows], "label":raw["label"][check_rows]}, 512)
    if not np.allclose(rescored, x[check_rows], rtol=2e-4, atol=1e-6):
        raise ValueError("AE cache fails independent batch-size scoring check")
    with torch.inference_mode():
        obj = torch.nan_to_num(raw["obj"][check_rows,:,:4].reshape(len(check_rows), -1).float(), nan=0., posinf=0., neginf=0.)
        normalized = (obj-mu)/(std+1e-8)
        recon, _ = ae(normalized)
        evaluation_scores = (recon-normalized).square().mean(1).numpy()
    parity = {"rows_checked": len(check_rows), "training_cache_max_absolute_error": float(np.max(np.abs(rescored-x[check_rows]))),
              "eval_epsilon_max_relative_difference": float(np.max(np.abs(evaluation_scores-rescored)/np.maximum(rescored,1e-12))),
              "eval_epsilon_selection_flips_in_checked_rows": int(np.sum((evaluation_scores>=cut)!=(rescored>=cut)))}
    categories = raw["background_category"].numpy()
    qcd = labels == 1; q = np.flatnonzero(qcd)
    qx, qw, qv = x[q], weights[q], is_val[q]
    retentions = {name: retention(qx[qv == flag], qw[qv == flag], cut) for name, flag in (("train",False),("validation",True))}
    delta = retentions["validation"]["retention"]-retentions["train"]["retention"]
    category_rows, membership_rows = [], []
    for name, flag in (("train",False),("validation",True)):
        for category in sorted(np.unique(categories[q])):
            mask = qcd & (is_val == flag) & (categories == category)
            row = {"split":name, "category":int(category), **retention(x[mask], weights[mask], cut)}
            row["fraction_of_qcd_physics"] = float(weights[mask].sum()/weights[qcd & (is_val==flag)].sum())
            category_rows.append(row)
        for seen in (False,True):
            mask = qcd & (is_val==flag) & (ae_seen==seen)
            membership_rows.append({"classifier_split":name, "AE_membership": "reconstructed_train" if seen else "reconstructed_validation",
                                    **retention(x[mask], weights[mask], cut)})
    print("Comparing every QCD AE input coordinate...", flush=True)
    original_obj = raw["obj"][q,:,:4].float().numpy()
    obj = np.nan_to_num(original_obj, nan=0., posinf=0., neginf=0.)
    feature_rows = []
    for slot in range(obj.shape[1]):
        for field, name in enumerate(("pt","eta","phi","type")):
            row=distribution_stats(obj[:,slot,field], qw, qv, f"obj{slot:02d}_{name}")
            row.update({f"{split_name}_nonfinite_raw_rows":int(np.sum(~np.isfinite(original_obj[mask,slot,field])))
                        for split_name,mask in (("train",~qv),("validation",qv))})
            feature_rows.append(row)
    jets = obj[:,:10,0]
    features = {"AE_MSE":qx, "physics_weight":qw,
                "HT10_pt30":np.where(jets>=30,jets,0).sum(1),
                "leading_jet_pt":jets[:,0], "jet_count10_pt30":(jets>=30).sum(1).astype(float)}
    if obj.shape[1] >= 23:
        features.update(MET_pt=obj[:,22,0], muon_pt_sum=obj[:,10:14,0].sum(1),
                        electron_pt_sum=obj[:,14:18,0].sum(1), photon_pt_sum=obj[:,18:22,0].sum(1))
    feature_rows.extend(distribution_stats(v,qw,qv,k) for k,v in features.items())
    weight_edges=np.unique(quantile(qw,qw,np.arange(.1,1,.1)))
    weight_group=np.searchsorted(weight_edges,qw,side="left")
    adjustment={"generator_category":standardized_validation(qx,qw,qv,categories[q],cut),
                "physics_weight_stratum":standardized_validation(qx,qw,qv,weight_group,cut),
                "AE_membership_reconstructed":standardized_validation(qx,qw,qv,ae_seen[q].astype(int),cut)}
    weight_rows=[]
    for group in np.unique(weight_group):
        for name,flag in (("train",False),("validation",True)):
            mask=(weight_group==group)&(qv==flag)
            weight_rows.append({"split":name,"stratum":int(group),
                                "fraction_of_qcd_physics":float(qw[mask].sum()/qw[qv==flag].sum()),
                                **retention(qx[mask],qw[mask],cut)})
    boot, boot_summary = bootstrap_gap(x,weights,labels,train,validation,args.bootstrap,args.seed)
    resplits, resplit_summary = random_resplits(x,weights,labels,categories,validation,args.resplits,args.seed+1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plot_dir = args.output_dir/"plots"; plot_dir.mkdir()
    write_csv(args.output_dir/"qcd_category_composition.csv", category_rows)
    write_csv(args.output_dir/"AE_membership_overlap.csv", membership_rows)
    write_csv(args.output_dir/"input_distribution_comparison.csv", feature_rows)
    write_csv(args.output_dir/"qcd_weight_stratum_comparison.csv", weight_rows)
    np.savez_compressed(args.output_dir/"uncertainty_replicates.npz", bootstrap=boot, frozen_score_resplits=resplits)
    # 1. Full weighted AE CDF, displayed as a physics-quantile zoom.
    fig, ax = plt.subplots(figsize=(8,6))
    for name, flag in (("classifier train",False),("classifier validation",True)):
        cdf_plot(ax,qx[qv==flag],qw[qv==flag],name)
    lo,hi = quantile(qx,qw,[.001,.999]); ax.set(xlim=(lo,hi),xlabel="AE MSE (display zoom only)",ylabel="Physics-weighted CDF")
    ax.axvline(cut,c="black",ls="--",label=f"frozen cut={cut:.5g}"); ax.legend(); fig.tight_layout()
    fig.savefig(plot_dir/"qcd_AE_CDF.png",dpi=160); plt.close(fig)
    # 2. Kinematic marginals before selection.
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    for ax,key in zip(axes,("HT10_pt30","leading_jet_pt","jet_count10_pt30")):
        for name,flag in (("train",False),("validation",True)):
            cdf_plot(ax,features[key][qv==flag],qw[qv==flag],name)
        lower,upper=quantile(features[key],qw,[0,.995])
        if upper<=lower:
            lower,upper=lower-.5,upper+.5
        ax.set(xlim=(lower,upper),xlabel=key,ylabel="Weighted CDF"); ax.legend()
    fig.tight_layout(); fig.savefig(plot_dir/"qcd_input_hardness.png",dpi=160); plt.close(fig)
    # 3. Every input assessed, show largest shifts.
    ranked = sorted(feature_rows,key=lambda r: r["weighted_KS"] or 0,reverse=True)[:20]
    fig, ax = plt.subplots(figsize=(9,7)); ax.barh([r["feature"] for r in ranked][::-1],[r["weighted_KS"] for r in ranked][::-1])
    ax.set_xlabel("Descriptive weighted KS distance (not a p-value)"); fig.tight_layout()
    fig.savefig(plot_dir/"largest_input_shifts.png",dpi=160); plt.close(fig)
    # 4. The overlap makes AE-seen versus unseen effects explicit.
    fig, ax = plt.subplots(figsize=(9,5)); locations=np.arange(len(membership_rows))
    heights=[r["retention"] or 0 for r in membership_rows]
    errors=[r["standard_error_fixed_cut"] or 0 for r in membership_rows]
    ax.bar(locations,heights,yerr=errors); ax.set_xticks(locations,[r["classifier_split"]+"\n"+r["AE_membership"] for r in membership_rows],fontsize=8)
    ax.set_ylabel("QCD physics retention (empty groups shown at zero)"); fig.tight_layout()
    fig.savefig(plot_dir/"AE_membership_retention.png",dpi=160); plt.close(fig)
    # 5/6. Uncertainty versus a diagnostic randomized-assignment reference.
    for name,values,label in (("bootstrap_gap",boot,"Poisson MC replicas; refit threshold"),
                              ("randomized_split_gap",resplits,"Frozen-score randomized assignments (not new validation)")):
        fig,ax=plt.subplots(figsize=(8,5)); ax.hist(values[:,3],bins=30)
        ax.axvline(delta,c="red",label=f"observed gap={delta:.3%}"); ax.axvline(0,c="black",ls="--")
        ax.set(xlabel="Validation minus training QCD retention",ylabel="Replicates",title=label); ax.legend(); fig.tight_layout()
        fig.savefig(plot_dir/f"{name}.png",dpi=160); plt.close(fig)
    # 7. Category shares before tail selection.
    fig,ax=plt.subplots(figsize=(8,5))
    cats=sorted(np.unique(categories[q]))
    for offset,name in ((-.18,"train"),(.18,"validation")):
        ax.bar(np.arange(len(cats))+offset,[r["fraction_of_qcd_physics"] for r in category_rows if r["split"]==name],width=.36,label=name)
    ax.set_xticks(np.arange(len(cats)),[str(c) for c in cats]); ax.set(yscale="log",xlabel="QCD generator category",ylabel="Fraction of QCD physics yield"); ax.legend(); fig.tight_layout()
    fig.savefig(plot_dir/"qcd_category_composition.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for name in ("train","validation"):
        rows=[r for r in weight_rows if r["split"]==name and r["retention"] is not None]
        ax.errorbar([r["stratum"] for r in rows],[r["retention"] for r in rows],
                    yerr=[r["standard_error_fixed_cut"] for r in rows],marker="o",label=name)
    ax.set(xlabel="Physics-weight stratum (pooled QCD weighted quantiles)",ylabel="AE-tail retention"); ax.legend(); fig.tight_layout()
    fig.savefig(plot_dir/"qcd_retention_by_weight.png",dpi=160); plt.close(fig)
    report={"protocol":"cached_train_validation_AE_tail_split_audit", "heldout_opened":False,"PF_encoder_inference":False,"training_submitted":False,
            "preparation_dir":str(args.preparation_dir.resolve()),"ae_threshold":cut,
            "integrity_checks_passed":True,"saved_classifier_split_reproduced_by_current_splitter":bool(reproduced),
            "reconstructed_AE_mass_matches_checkpoint":bool(mass_matches),"reconstructed_AE_parent_mass":reconstructed_mass,
            "AE_physics_weighted_loss_reconstruction":ae_losses,
            "split_seed_sensitivity":seed_overlap,
            "AE_membership_provenance":"reconstructed from saved parent-label seed/fraction and current splitter; AE checkpoint did not archive indices",
            "splitter_source":inspect.getsource(stratified_split_indices),
            "classifier_split_parameters":{"seed":split["seed"],"validation_fraction":split["validation_fraction"]},
            "AE_split_parameters":{"seed":ae_meta["split_seed"],"validation_fraction":ae_meta["validation_fraction"]},
            "score_parity":parity,"qcd_retention":retentions,"observed_validation_minus_train_retention":float(delta),
            "bootstrap":boot_summary,"randomized_frozen_score_assignments":resplit_summary,
            "common_composition_adjustments":adjustment,"QCD_physics_weight_stratum_edges":weight_edges.tolist(),
            "AE_membership_overlap":membership_rows,"largest_descriptive_distribution_shifts":ranked,
            "limitations":["No causal attribution from frozen-score comparisons alone; no model was retrained.",
                           "Reconstructed AE membership is conditional on historical splitter matching current implementation.",
                           "Bootstrap is event-level MC precision conditional on the frozen trained AE and saved split.",
                           "Inputs are compared before selection; reconstructed HT uses only the first ten stored jets.",
                           "Many input comparisons: weighted KS values are descriptive, not multiple-testing-corrected significance."]}
    (args.output_dir/"split_audit_summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ("qcd_retention","observed_validation_minus_train_retention","score_parity","bootstrap","saved_classifier_split_reproduced_by_current_splitter","AE_membership_overlap")},indent=2))
    print(f"AUDIT COMPLETE in {time.monotonic()-started:.1f}s: {args.output_dir}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("preparation-dir","ae-checkpoint","weights","output-dir"):
        p.add_argument(f"--{name}",required=True,type=Path)
    p.add_argument("--bootstrap",type=int,default=200); p.add_argument("--resplits",type=int,default=30)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    if args.bootstrap<20 or args.resplits<10:
        p.error("Require at least 20 bootstrap and 10 diagnostic re-splits")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    audit(args)


if __name__=="__main__":
    main()
