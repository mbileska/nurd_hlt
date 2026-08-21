import torch

from models.hlt_autoencoder import HLTAutoencoder
from train_hlt import main
from utils.hlt_weights import load_generator_weights, sample_signature


def test_one_epoch_continuous_density_ratio_training_smoke(tmp_path, monkeypatch):
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
    main([
        "--data", str(data_path),
        "--ae_ckpt", str(ae_path),
        "--gen_weight_path", str(weights_path),
        "--epochs", "1",
        "--batch_size", "16",
        "--balance_strata", "3",
        "--critic_steps", "1",
        "--embed_size", "8",
        "--latent_dim", "3",
        "--proj_dim", "3",
        "--num_heads", "1",
        "--num_layers", "1",
        "--dim_ff", "16",
        "--linear_dim", "2",
        "--local_testing", "1",
        "--exp_name", "smoke",
    ])
    checkpoint = tmp_path / "checkpoints/hlt/hlt/smoke/checkpoint_main.pth.tar"
    assert checkpoint.exists()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["preprocessing"]["nuisance_transform"]["kind"].startswith(
        "standardize_continuous")
    assert payload["validation_metrics"]["balanced_accuracy"] >= 0.0
