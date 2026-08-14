from pathlib import Path


def test_della_defaults_are_the_weighted_v4_anchor():
    script = (
        Path(__file__).resolve().parents[1] / "slurm" / "submit_train.sbatch"
    ).read_text()

    expected_defaults = (
        'RUN_TAG="${RUN_TAG:-weighted_v4_anchor_v12_',
        'TRAINING_PROFILE="${TRAINING_PROFILE:-weighted_v4_anchor}"',
        'N_BINS="${N_BINS:-20}"',
        'CRITIC_TYPE="${CRITIC_TYPE:-density_ratio}"',
        'CRITIC_WEIGHTED_SHUFFLE="${CRITIC_WEIGHTED_SHUFFLE:-0}"',
        'N_CRITIC_STEPS="${N_CRITIC_STEPS:-1}"',
        'QCD_BATCH_FRACTION="${QCD_BATCH_FRACTION:-0.0}"',
        'CLOSURE_REVERSE_PROFILE_WEIGHT="${CLOSURE_REVERSE_PROFILE_WEIGHT:-0.0}"',
        'CLOSURE_PHYSICAL_RESAMPLE="${CLOSURE_PHYSICAL_RESAMPLE:-0}"',
        'MD_PROXY_TYPE="${MD_PROXY_TYPE:-ema}"',
        'MD_EMA_MOMENTUM="${MD_EMA_MOMENTUM:-0.05}"',
        'MD_PROXY_SHRINKAGE="${MD_PROXY_SHRINKAGE:-0.0}"',
        'VAL_MD_MODE="${VAL_MD_MODE:-ema}"',
        'ABCD_CKPT_MIN_EPOCH="${ABCD_CKPT_MIN_EPOCH:-1}"',
    )

    for expected in expected_defaults:
        assert expected in script


def test_weighted_v4_uses_new_sample_and_generator_weights():
    script = (
        Path(__file__).resolve().parents[1] / "slurm" / "submit_train.sbatch"
    ).read_text()

    assert "mequinna_1M_noZB" in script
    assert "hlt_smcocktail_mequinna_train.pt" in script
    assert "weight_train.pt" in script
    assert '--gen_weights "$GEN_WEIGHT_TRAIN"' in script


def test_latest_eval_discovers_v4_anchor_and_keeps_combined_report():
    eval_script = (
        Path(__file__).resolve().parents[1]
        / "slurm"
        / "submit_eval_latest.sbatch"
    ).read_text()
    evaluator = (
        Path(__file__).resolve().parents[1] / "eval_abcd_nurd.py"
    ).read_text()

    assert "hlt_nurd_closure_bs4096_weighted_v4_anchor_v12_*" in eval_script
    assert "--reference_pt" in eval_script
    assert 'diagnostics["legacy_same_sample"]' in evaluator
