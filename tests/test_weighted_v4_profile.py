from pathlib import Path


def test_della_defaults_are_the_corrected_weighted_v4_x4_profile():
    script = (
        Path(__file__).resolve().parents[1] / "slurm" / "submit_train.sbatch"
    ).read_text()

    expected_defaults = (
        'RUN_TAG="${RUN_TAG:-weighted_v4_x4_',
        'TRAINING_PROFILE="${TRAINING_PROFILE:-weighted_v4_x4}"',
        'N_BINS="${N_BINS:-80}"',
        'THRESHOLD_TUNE_FRACTION="${THRESHOLD_TUNE_FRACTION:-0.5}"',
        'CRITIC_TYPE="${CRITIC_TYPE:-density_ratio}"',
        'CRITIC_WEIGHTED_SHUFFLE="${CRITIC_WEIGHTED_SHUFFLE:-1}"',
        'N_CRITIC_STEPS="${N_CRITIC_STEPS:-1}"',
        'QCD_BATCH_FRACTION="${QCD_BATCH_FRACTION:-0.0}"',
        'CLOSURE_REVERSE_PROFILE_WEIGHT="${CLOSURE_REVERSE_PROFILE_WEIGHT:-0.5}"',
        'CLOSURE_PHYSICAL_RESAMPLE="${CLOSURE_PHYSICAL_RESAMPLE:-0}"',
        'MD_PROXY_TYPE="${MD_PROXY_TYPE:-ema}"',
        'MD_EMA_MOMENTUM="${MD_EMA_MOMENTUM:-0.05}"',
        'MD_PROXY_SHRINKAGE="${MD_PROXY_SHRINKAGE:-0.0}"',
        'VAL_MD_MODE="${VAL_MD_MODE:-cross_fitted}"',
        'ABCD_CKPT_MIN_EPOCH="${ABCD_CKPT_MIN_EPOCH:-20}"',
        'VAL_ABCD_MIN_EFFECTIVE_EVENTS="${VAL_ABCD_MIN_EFFECTIVE_EVENTS:-20}"',
        'VAL_ABCD_MAX_RATIO_UNC="${VAL_ABCD_MAX_RATIO_UNC:-0.15}"',
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
    assert script.count(
        '--threshold_tune_fraction "$THRESHOLD_TUNE_FRACTION"') == 2
    assert '--code_commit "$CODE_COMMIT"' in script


def test_eval_is_dual_strict_and_uses_sibling_output_directories():
    eval_script = (
        Path(__file__).resolve().parents[1]
        / "slurm"
        / "submit_eval_latest.sbatch"
    ).read_text()
    assert '"$OUTDIR_ROOT/held-out"' in eval_script
    assert '"$OUTDIR_ROOT/legacy"' in eval_script
    assert "--evaluation_protocol heldout" in eval_script
    assert "--evaluation_protocol legacy_main_qcd" in eval_script
    assert "wildcard/latest discovery is disabled" in eval_script
    assert "checkpoint_main_" not in eval_script
    assert "checkpoint_abcd.pth.tar" in eval_script
    assert "ae_pretrain_*" not in eval_script
    assert "evaluation_summary.json" in eval_script
