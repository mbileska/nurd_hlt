import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from models.hlt_autoencoder import HLTAutoencoder
from scripts.plot_pure_supcon_projector import main as projector_main
from scripts.plot_pure_supcon_projector import plot_kde, selected_indices
from train_hlt import main
from utils.hlt_weights import load_generator_weights, sample_signature


def test_pure_supcon_checkpoint_uses_validation_supcon(tmp_path, monkeypatch):
    generator = torch.Generator().manual_seed(13)
    n = 128
    labels = torch.arange(n) % 4
    pf = torch.randn(n, 6, 7, generator=generator)
    pf[:, :, 0] = pf[:, :, 0].abs() + .1
    obj = torch.randn(n, 2, 4, generator=generator)
    sample = {"pf": pf, "obj": obj, "label": labels}
    data = tmp_path / "train.pt"
    weights = tmp_path / "weights.pt"
    ae_path = tmp_path / "ae.pth"
    torch.save(sample, data)
    torch.save(torch.ones(n), weights)
    _, weight_meta = load_generator_weights(
        str(weights), labels, qcd_label=1, sample=sample)
    ae_config = {
        "features": 8, "latent_dim": 3,
        "encoder_config": {"nodes": [12]},
        "decoder_config": {"nodes": [12, 8]},
    }
    ae = HLTAutoencoder(ae_config)
    torch.save({
        "ae": ae.state_dict(), "ae_config": ae_config,
        "ae_scaler": {"mu": torch.zeros(8), "std": torch.ones(8)},
        "data_signature": sample_signature(sample),
        "weighting": {"generator": weight_meta},
    }, ae_path)
    monkeypatch.chdir(tmp_path)
    args = [
        "--data", str(data), "--gen_weight_path", str(weights),
        "--ae_ckpt", str(ae_path), "--epochs", "2", "--batch_size", "16",
        "--balance_strata", "1", "--contrast_weight", "0.3",
        "--ce_weight", "0", "--nurd_enabled", "0",
        "--embed_size", "8", "--latent_dim", "3", "--proj_dim", "3",
        "--num_heads", "1", "--num_layers", "1", "--dim_ff", "16",
        "--linear_dim", "2", "--local_testing", "1",
        "--exp_name", "pure_supcon_smoke",
    ]
    main(args)
    path = Path("checkpoints/hlt/hlt/pure_supcon_smoke/checkpoint_main.pth.tar")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["checkpoint_role"] == "best_weighted_validation_supcon"
    assert payload["selection_metric"] == "training-fit-weighted_validation_supcon"
    assert payload["selection_value"] == pytest.approx(
        payload["validation_metrics"]["contrastive_loss"])
    assert payload["validation_metrics"]["contrastive_loss"] > 0
    assert payload["classification_head_trained"] is False
    assert payload["nurd_active"] is False
    history = json.loads(Path(
        "checkpoints/hlt/hlt/pure_supcon_smoke/training_history.json").read_text())
    assert len(history) == 2
    projector_out = tmp_path / "projector_plots"
    monkeypatch.setattr(sys, "argv", [
        "plot_pure_supcon_projector.py", "--checkpoint", str(path),
        "--sample", str(data), "--output-dir", str(projector_out),
        "--per-class", "6", "--batch-size", "8",
    ])
    projector_main()
    assert (projector_out / "projector_pca_kde_by_class.png").stat().st_size > 0
    with pytest.raises(ValueError, match="Pure SupCon requires"):
        main(args + ["--nurd_enabled", "1"])


def test_projector_kde_is_deterministic_and_written(tmp_path):
    labels = np.repeat(np.arange(8), 12)
    first, counts = selected_indices(labels, 6, 42)
    second, _ = selected_indices(labels, 6, 42)
    assert np.array_equal(first, second)
    assert set(counts.values()) == {6}
    rng = np.random.default_rng(5)
    projected = np.vstack([
        rng.normal(loc=(i % 4, i // 4), scale=.2, size=(6, 2))
        for i in range(8)
    ])
    output = tmp_path / "kde.png"
    plot_kde(projected, labels[first], {}, output)
    assert output.stat().st_size > 0
