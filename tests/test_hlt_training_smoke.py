import pytest
import torch

from models.hlt_autoencoder import HLTAutoencoder
from train_hlt import main
from utils.hlt_weights import load_generator_weights, sample_signature


@pytest.mark.parametrize("nurd_enabled", [0, 1])
def test_one_epoch_continuous_density_ratio_training_smoke(
    tmp_path, monkeypatch, nurd_enabled
):
    generator = torch.Generator().manual_seed(13)
    n_events = 96
    labels = torch.arange(n_events) % 4
    pf = torch.randn(n_events, 6, 7, generator=generator)
    pf[:, :, 0] = pf[:, :, 0].abs() + 0.1
    obj = torch.randn(n_events, 2, 4, generator=generator)
    data_path = tmp_path / "train.pt"
    weights_path = tmp_path / "weights.pt"
    ae_path = tmp_path / "ae.pth"
    torch.save({"pf": pf, "obj": obj, "label": labels}, data_path)
    generator_weights = torch.ones(n_events)
    generator_weights[labels == 1] = torch.linspace(
        0.1, 20.0, int((labels == 1).sum()))
    torch.save(generator_weights, weights_path)
    sample = {"pf": pf, "obj": obj, "label": labels}
    _, generator_metadata = load_generator_weights(
        str(weights_path), labels, qcd_label=1, sample=sample)

    features = obj.shape[1] * 4
    ae_config = {
        "features": features,
        "latent_dim": 3,
        "encoder_config": {"nodes": [12]},
        "decoder_config": {"nodes": [12, features]},
        "alpha": 1.0,
    }
    ae = HLTAutoencoder(ae_config)
    torch.save({
        "ae": ae.state_dict(),
        "ae_config": ae_config,
        "ae_scaler": {
            "mu": torch.zeros(features),
            "std": torch.ones(features),
        },
        "data_signature": sample_signature(sample),
        "weighting": {"generator": generator_metadata},
    }, ae_path)

    monkeypatch.chdir(tmp_path)
    experiment = f"smoke_nurd_{nurd_enabled}"
    main([
        "--data", str(data_path),
        "--ae_ckpt", str(ae_path),
        "--gen_weight_path", str(weights_path),
        "--epochs", "1",
        "--batch_size", "16",
        "--balance_strata", "3",
        "--critic_steps", "1",
        "--nurd_enabled", str(nurd_enabled),
        "--critic_start_epoch", "0",
        "--embed_size", "8",
        "--latent_dim", "3",
        "--proj_dim", "3",
        "--num_heads", "1",
        "--num_layers", "1",
        "--dim_ff", "16",
        "--linear_dim", "2",
        "--local_testing", "1",
        "--exp_name", experiment,
    ])
    checkpoint = (
        tmp_path / f"checkpoints/hlt/hlt/{experiment}/checkpoint_main.pth.tar")
    assert checkpoint.exists()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["preprocessing"]["nuisance_transform"]["kind"].startswith(
        "standardize_continuous")
    assert payload["validation_metrics"]["balanced_accuracy"] >= 0.0
    assert payload["nurd_active"] is bool(nurd_enabled)
    assert payload["effective_lambda_info"] == pytest.approx(
        1.0 if nurd_enabled else 0.0)
    if nurd_enabled:
        assert payload["train_metrics"]["critic_loss"] > 0.0
    else:
        assert payload["train_metrics"]["critic_loss"] == 0.0


def test_one_epoch_decay8_target_training_smoke(tmp_path, monkeypatch):
    generator = torch.Generator().manual_seed(23)
    categories = torch.tensor([15, 0, 27, 6, 9, 12, 18, 21]).repeat(12)
    parents = torch.tensor([0, 1, 2, 3, 3, 3, 0, 0]).repeat(12)
    n_events = parents.numel()
    pf = torch.randn(n_events, 6, 7, generator=generator)
    pf[:, :, 0] = pf[:, :, 0].abs() + 0.1
    obj = torch.randn(n_events, 2, 4, generator=generator)
    sample = {
        "pf": pf,
        "obj": obj,
        "label": parents,
        "background_category": categories,
    }
    data_path = tmp_path / "decay8.pt"
    weights_path = tmp_path / "weights.pt"
    ae_path = tmp_path / "ae.pth"
    torch.save(sample, data_path)
    weights = torch.ones(n_events)
    torch.save(weights, weights_path)
    _, generator_metadata = load_generator_weights(
        str(weights_path), parents, qcd_label=1, sample=sample)

    features = obj.shape[1] * 4
    ae_config = {
        "features": features,
        "latent_dim": 3,
        "encoder_config": {"nodes": [12]},
        "decoder_config": {"nodes": [12, features]},
        "alpha": 1.0,
    }
    ae = HLTAutoencoder(ae_config)
    torch.save({
        "ae": ae.state_dict(),
        "ae_config": ae_config,
        "ae_scaler": {
            "mu": torch.zeros(features),
            "std": torch.ones(features),
        },
        "data_signature": sample_signature(sample),
        "weighting": {"generator": generator_metadata},
    }, ae_path)

    monkeypatch.chdir(tmp_path)
    main([
        "--data", str(data_path),
        "--ae_ckpt", str(ae_path),
        "--gen_weight_path", str(weights_path),
        "--target_scheme", "decay8",
        "--epochs", "1",
        "--batch_size", "16",
        "--balance_strata", "2",
        "--critic_steps", "1",
        "--nurd_enabled", "1",
        "--embed_size", "8",
        "--latent_dim", "3",
        "--proj_dim", "3",
        "--num_heads", "1",
        "--num_layers", "1",
        "--dim_ff", "16",
        "--linear_dim", "2",
        "--local_testing", "1",
        "--exp_name", "smoke_decay8",
    ])

    checkpoint = (
        tmp_path / "checkpoints/hlt/hlt/smoke_decay8/checkpoint_main.pth.tar")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["state_dict_model"]["classifier.weight"].shape[0] == 8
    assert payload["preprocessing"]["target_scheme"] == "decay8"
    assert payload["preprocessing"]["target_class_names"][7] == "Z->tautau"
