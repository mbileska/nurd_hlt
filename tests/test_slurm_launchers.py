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
