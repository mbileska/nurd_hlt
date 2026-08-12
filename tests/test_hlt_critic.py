import torch

from models.hlt_con import HLTCritic


def test_multiresolution_critic_heads():
    critic = HLTCritic(
        latent_dim=6,
        num_classes=4,
        n_bins=40,
        critic_type="bin_pred",
        bin_resolutions=[10, 20, 40],
    )
    latent = torch.randn(16, 6)
    labels = torch.ones(16, 1)
    outputs = critic(latent, labels)

    assert sorted(outputs) == [10, 20, 40]
    assert outputs[10].shape == (16, 10)
    assert outputs[20].shape == (16, 20)
    assert outputs[40].shape == (16, 40)
