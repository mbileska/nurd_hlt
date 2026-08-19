import os
import re
import subprocess
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


def test_campaign_launcher_cannot_be_hijacked_by_stale_run_exports():
    launcher = (
        Path(__file__).resolve().parents[1]
        / "slurm"
        / "launch_v4x4_campaign.sh"
    ).read_text()

    assert 'RUN_TAG="${1:-weighted_v4_x4_' in launcher
    assert 'AE_EXP="ae_pretrain_$RUN_TAG"' in launcher
    assert 'NURD_EXP="hlt_nurd_closure_bs4096_$RUN_TAG"' in launcher
    assert 'SKIP_AE="0"' in launcher
    assert 'CAMPAIGN_CONTRACT="weighted_v4_x4_fresh_v1"' in launcher
    assert 'EVAL_CONTRACT="dual_qcd_v1"' in launcher
    assert 'RUN_TAG="${RUN_TAG:-' not in launcher
    assert 'AE_CKPT="${AE_CKPT:-' not in launcher
    assert 'NURD_EXP="${NURD_EXP:-' not in launcher


def test_sealed_campaign_resets_train_and_eval_environment_overrides():
    root = Path(__file__).resolve().parents[1]
    train_script = (root / "slurm" / "submit_train.sbatch").read_text()
    eval_script = (root / "slurm" / "submit_eval_latest.sbatch").read_text()

    def reset_names(script, contract):
        block = script.split(f'if [[ "${{{contract}:-}}"', 1)[1].split("fi", 1)[0]
        return set(re.findall(r"\b[A-Z][A-Z0-9_]*\b", block))

    def defaulted_names(script):
        return set(
            re.findall(
                r'^([A-Z][A-Z0-9_]*)="\$\{[A-Z][A-Z0-9_]*:-',
                script,
                flags=re.MULTILINE,
            )
        )

    train_exempt = {
        "BASE",
        "CODE_DIR",
        "MIN_GPU_MEM_GB",
        "WANDB_MODE",
        "WANDB_INIT_TIMEOUT",
        "PYTORCH_CUDA_ALLOC_CONF",
        "RUN_TAG",
        "AE_EXP",
        "NURD_EXP",
        "AE_CKPT",
        "CODE_COMMIT",
    }
    assert defaulted_names(train_script) - train_exempt <= reset_names(
        train_script, "CAMPAIGN_CONTRACT"
    )

    eval_exempt = {
        "BASE",
        "CODE_DIR",
        "WANDB_MODE",
        "MPLCONFIGDIR",
        "NURD_EXP",
        "EVAL_NAME",
        "JOB_TOKEN",
    }
    eval_resets = reset_names(eval_script, "EVAL_CONTRACT")
    assert defaulted_names(eval_script) - eval_exempt <= eval_resets
    assert {"CKPT", "OUTDIR_ROOT"} <= eval_resets


def test_launcher_ignores_exported_values_from_an_old_session(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = rev-parse ]; then echo abcdef123456; fi\n"
        "exit 0\n"
    )
    fake_git.chmod(0o755)
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text("#!/bin/sh\necho 999999\n")
    fake_sbatch.chmod(0o755)

    root = Path(__file__).resolve().parents[1]
    run_tag = "weighted_v4_x4_shell_isolation_test"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "BASE": str(tmp_path / "scratch"),
            "RUN_TAG": "stale_run",
            "AE_EXP": "stale_ae",
            "NURD_EXP": "stale_nurd",
            "EVAL_NAME": "stale_eval",
            "AE_CKPT": "/stale/ae.pth",
            "CKPT": "/stale/nurd.pth.tar",
            "SKIP_AE": "1",
            "N_BINS": "7",
        }
    )
    result = subprocess.run(
        ["bash", str(root / "slurm" / "launch_v4x4_campaign.sh"), run_tag],
        cwd=root,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    assert run_tag in result.stdout
    assert f"ae_pretrain_{run_tag}/checkpoint_ae.pth" in result.stdout
    assert f"hlt_nurd_closure_bs4096_{run_tag}" in result.stdout
    assert "fresh AE + weighted V4x4 NURD" in result.stdout
    assert "stale_" not in result.stdout
