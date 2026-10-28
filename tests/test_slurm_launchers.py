import os
from pathlib import Path
import shutil
import subprocess


def test_warm_start_wrapper_forwards_exactly_two_launcher_arguments(tmp_path):
    source = Path(__file__).parents[1] / "slurm" / "launch_warm_start_nurd.sh"
    slurm = tmp_path / "slurm"
    slurm.mkdir()
    wrapper = slurm / source.name
    shutil.copy2(source, wrapper)

    capture = tmp_path / "arguments.txt"
    launcher = slurm / "launch_train3m_campaign.sh"
    launcher.write_text(
        "#!/bin/bash\n"
        "printf '%s\\n' \"$#\" \"$@\" > \"$LAUNCH_CAPTURE\"\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)

    baseline = "baseline"
    base = tmp_path / "base"
    ae = (
        base / "checkpoints/hlt/hlt"
        / f"ae_engineer_{baseline}/checkpoint_ae.pth"
    )
    classifier = (
        base / "checkpoints/hlt/hlt"
        / f"hlt_nurd_engineer_{baseline}/checkpoint_main.pth.tar"
    )
    ae.parent.mkdir(parents=True)
    classifier.parent.mkdir(parents=True)
    ae.write_bytes(b"ae")
    classifier.write_bytes(b"classifier")

    environment = os.environ.copy()
    environment.update({"BASE": str(base), "LAUNCH_CAPTURE": str(capture)})
    result = subprocess.run(
        ["bash", str(wrapper), baseline, "continuation", "0.3"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert "Warm-start baseline: baseline" in result.stdout
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "2", "continuation", "0.3",
    ]


def test_validation_scan_accepts_reused_ae_checkpoint(tmp_path):
    source = (
        Path(__file__).parents[1]
        / "slurm/launch_validation_checkpoint_scan.sh"
    )
    slurm = tmp_path / "slurm"
    slurm.mkdir()
    launcher = slurm / source.name
    shutil.copy2(source, launcher)

    base = tmp_path / "base"
    baseline_ae = (
        base / "checkpoints/hlt/hlt"
        / "ae_engineer_baseline/checkpoint_ae.pth"
    )
    continuation = (
        base / "checkpoints/hlt/hlt"
        / "hlt_nurd_engineer_continuation"
    )
    baseline_ae.parent.mkdir(parents=True)
    continuation.mkdir(parents=True)
    baseline_ae.write_bytes(b"ae")

    capture = tmp_path / "sbatch_arguments.txt"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/bin/bash\n"
        "if [[ \"$1\" == rev-parse ]]; then echo deadbeef; fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/bash\n"
        "printf '%s\\n' \"$@\" > \"$SBATCH_CAPTURE\"\n"
        "echo 12345\n",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)

    environment = os.environ.copy()
    environment.update({
        "BASE": str(base),
        "AE_EXP": "ae_engineer_baseline",
        "SBATCH_CAPTURE": str(capture),
        "PATH": f"{fake_bin}:{environment['PATH']}",
    })
    result = subprocess.run(
        ["bash", str(launcher), "continuation", "1", "3", "13"],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=environment,
    )

    assert f"AE checkpoint:   {baseline_ae}" in result.stdout
    arguments = capture.read_text(encoding="utf-8")
    assert f'AE_CKPT={baseline_ae}' in arguments
    assert "EPOCHS_CSV=1,3,13" in arguments
