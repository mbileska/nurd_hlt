import json

import numpy as np
import pytest

from scripts.compare_qcd_score_audit import (
    conditional_shift, fixed_point, load_scores, paired_bootstrap,
    region_yields, verify_pair, verify_saved_report, weighted_ks,
)
from scripts.evaluate_decay8_parent4_legacy import aggregate_parent4


def _scores(tmp_path, name, *, heldout=True, shift=0):
    rng = np.random.default_rng(51)
    size = 6000
    ids = np.column_stack((np.zeros(size, dtype=np.int64),
                           np.ones(size, dtype=np.int64),
                           np.ones(size, dtype=np.int64),
                           np.arange(size, dtype=np.int64)))
    ae = rng.uniform(.01, 1, size).astype(np.float32)
    md = (rng.uniform(.01, 1, size) + shift * ae).astype(np.float32)
    weight = rng.uniform(.7, 1.3, size)
    ht = rng.uniform(20, 500, size)
    path = tmp_path / f"{name}.npz"
    category_name = "background_category" if heldout else "category"
    np.savez(path, event_id=ids, **{category_name: np.zeros(size, dtype=np.int16)},
             ae_score=ae, md_score=md, physics_weight=weight,
             ht10_ge30=ht)
    return path


def test_pair_audit_recounts_saved_abcd_and_matches_events(tmp_path):
    baseline = load_scores(_scores(tmp_path, "baseline"), heldout=True)
    candidate = load_scores(_scores(tmp_path, "candidate", shift=.2), heldout=True)
    verify_pair(baseline, candidate)
    _, yields, sumw2, raw = region_yields(
        baseline["ae"], baseline["md"], baseline["weight"], .5, .5)
    saved = {
        "evaluations": {"held-out": {
            "thresholds": {"t1": .5, "t2": .5},
            "diagnostics": {
                "evaluation_protocol": "saved_training_validation_split",
                "weighted_qcd": True, "qcd_events": len(baseline["ae"]),
                "working_point": {"regions": {
                    region: {"yield": float(yields[i]),
                             "sumw2": float(sumw2[i]),
                             "raw_count": int(raw[i])}
                    for i, region in enumerate("ABCD")}}
            }}}
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(saved))
    verify_saved_report(baseline, report_path)
    saved["evaluations"]["held-out"]["diagnostics"]["working_point"]["regions"]["A"]["raw_count"] += 1
    report_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="A count/yield"):
        verify_saved_report(baseline, report_path)

    point_left, codes_left = fixed_point(baseline, baseline, .5, .1)
    point_right, codes_right = fixed_point(candidate, candidate, .5, .1)
    assert abs(point_left["validation_A_weighted_efficiency"] - .1) < .002
    assert abs(point_right["validation_A_weighted_efficiency"] - .1) < .002
    bootstrap = paired_bootstrap(baseline, candidate, codes_left, codes_right,
                                 repeats=100, seed=4)
    assert bootstrap["repeats"] == 100
    assert np.isfinite(bootstrap["delta_95pct_interval"]).all()

    candidate["event_id"][0, 3] = -1
    with pytest.raises(ValueError, match="event_id"):
        verify_pair(baseline, candidate)


def test_conditional_shift_and_weighted_ks(tmp_path):
    validation = load_scores(_scores(tmp_path, "validation", heldout=False), heldout=False)
    heldout = load_scores(_scores(tmp_path, "heldout"), heldout=True)
    assert weighted_ks(validation["md"], validation["weight"],
                       heldout["md"], heldout["weight"]) == 0
    rows, boundaries = conditional_shift(validation, heldout, model_name="baseline")
    assert len(rows) == 30
    assert len(boundaries["ae_edges"]) == 9
    assert max(row["weighted_md_ks"] for row in rows if np.isfinite(row["weighted_md_ks"])) == 0


def test_decay8_probability_aggregation_preserves_parent_probabilities():
    logits = np.array([[1., 2., 3., 4., 5., 6., 7., 8.]], dtype=np.float32)
    parent = aggregate_parent4(logits)
    leaf = np.exp(logits - logits.max(axis=1, keepdims=True))
    leaf /= leaf.sum(axis=1, keepdims=True)
    assert parent.shape == (1, 4)
    assert np.allclose(parent.sum(axis=1), 1)
    assert np.isclose(parent[0, 0], leaf[0, [0, 6, 7]].sum())
    assert np.isclose(parent[0, 3], leaf[0, [3, 4, 5]].sum())
