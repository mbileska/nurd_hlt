import subprocess
import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_ae_trains_on_all_classes_with_generator_only_weights(tmp_path):
    generator = torch.Generator().manual_seed(17)
    labels = torch.arange(40) % 4
    obj = torch.randn(40, 2, 4, generator=generator)
    data_path = tmp_path / "train.pt"
    weight_path = tmp_path / "weights.pt"
    torch.save({"obj": obj, "label": labels}, data_path)

    generator_weights = torch.ones(40)
    generator_weights[labels == 1] = torch.linspace(
        0.25, 8.0, int((labels == 1).sum()))
    torch.save(generator_weights, weight_path)

    subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "train_ae.py"),
            "--data", str(data_path),
            "--gen_weight_path", str(weight_path),
            "--generator_weight_label", "1",
            "--epochs", "1",
            "--batch_size", "8",
            "--latent_dim", "2",
            "--enc_nodes", "8",
            "--dec_nodes", "8",
            "--val_split", "0.2",
            "--exp_name", "ae_all_classes",
            "--project_name", "hlt",
            "--local_testing", "1",
            "--manualSeed", "7",
        ],
        cwd=tmp_path,
        check=True,
    )

    checkpoint_path = (
        tmp_path / "checkpoints/hlt/hlt/ae_all_classes/checkpoint_ae.pth")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)
    weighting = checkpoint["weighting"]
    assert checkpoint["data_signature"]["n_events"] == 40
    assert weighting["method"] == "generator_only_all_events"
    assert weighting["included_labels"] == [0, 1, 2, 3]
    assert weighting["included_event_counts"] == {0: 10, 1: 10, 2: 10, 3: 10}
    assert weighting["class_balancing"] is False
    assert weighting["nuisance_balancing"] is False
    assert "class_factors" not in weighting
