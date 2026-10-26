import torch
import torch.nn as nn

from dataset.hlt_smcocktail_dataset import build_hlt_datasets
from utils.hlt_weights import effective_mass_by_class


class ZeroAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, inputs):
        return torch.zeros_like(inputs) + self.anchor * 0.0, inputs[:, :2]


def test_dataset_uses_continuous_nuisance_and_training_fitted_weights(tmp_path):
    generator = torch.Generator().manual_seed(7)
    n_events = 240
    labels = torch.arange(n_events) % 4
    pf = torch.randn(n_events, 8, 7, generator=generator)
    pf[:, :, 0] = pf[:, :, 0].abs() + 0.1
    obj = torch.randn(n_events, 3, 4, generator=generator)
    obj = obj * (1.0 + labels.float().view(-1, 1, 1) * 0.2)
    data_path = tmp_path / "sample.pt"
    weight_path = tmp_path / "weights.pt"
    torch.save({"pf": pf, "obj": obj, "label": labels}, data_path)
    generator_weights = torch.ones(n_events)
    generator_weights[labels == 1] = torch.linspace(
        0.01, 100.0, int((labels == 1).sum()))
    torch.save(generator_weights, weight_path)

    feature_count = obj.shape[1] * 4
    train, validation, preprocessing = build_hlt_datasets(
        str(data_path),
        ZeroAutoencoder(),
        val_split=0.25,
        seed=11,
        gen_weight_path=str(weight_path),
        qcd_label=1,
        ae_scaler={
            "mu": torch.zeros(feature_count),
            "std": torch.ones(feature_count),
        },
        balance_strata=4,
        ae_batch_size=32,
    )

    assert train.nuisance.dtype == torch.float32
    assert torch.unique(train.nuisance).numel() > 4
    assert len(train[0]) == 6
    assert preprocessing["nuisance_transform"]["kind"].startswith(
        "standardize_continuous")
    assert preprocessing["weighting"]["balance_spec"]["requested_strata"] == 4
    balance_spec = preprocessing["weighting"]["balance_spec"]
    assert balance_spec["binning"] == "log_fixed"
    assert balance_spec["transform"]["kind"] == "scaled_log1p"
    assert balance_spec["weight_clipping"]["quantile"] == 0.995
    assert balance_spec["weight_clipping"]["enabled"] is True
    train_mass = effective_mass_by_class(
        train.labels, train.effective_weights)
    assert all(abs(value - 0.25) < 1e-5 for value in train_mass.values())
    assert set(train.indices.tolist()).isdisjoint(validation.indices.tolist())


def test_dataset_can_opt_into_decay_channel_targets(tmp_path):
    generator = torch.Generator().manual_seed(19)
    categories = torch.tensor([15, 18, 21, 0, 27, 6, 9, 12]).repeat(20)
    parents = torch.tensor([0, 0, 0, 1, 2, 3, 3, 3]).repeat(20)
    n_events = parents.numel()
    pf = torch.randn(n_events, 5, 7, generator=generator)
    pf[:, :, 0] = pf[:, :, 0].abs() + 0.1
    obj = torch.randn(n_events, 2, 4, generator=generator)
    data_path = tmp_path / "decay8.pt"
    weight_path = tmp_path / "weights.pt"
    torch.save({
        "pf": pf,
        "obj": obj,
        "label": parents,
        "background_category": categories,
    }, data_path)
    torch.save(torch.ones(n_events), weight_path)

    train, validation, preprocessing = build_hlt_datasets(
        str(data_path),
        ZeroAutoencoder(),
        val_split=0.25,
        seed=3,
        gen_weight_path=str(weight_path),
        qcd_label=1,
        target_scheme="decay8",
        ae_scaler={
            "mu": torch.zeros(obj.shape[1] * 4),
            "std": torch.ones(obj.shape[1] * 4),
        },
        balance_strata=2,
        ae_batch_size=32,
    )

    assert sorted(torch.unique(train.labels).tolist()) == list(range(8))
    assert sorted(torch.unique(validation.labels).tolist()) == list(range(8))
    assert preprocessing["target_scheme"] == "decay8"
    assert preprocessing["target_class_names"][4] == "W->munu"
    assert preprocessing["target_class_names"][6] == "Z->mumu"
    assert preprocessing["qcd_label"] == 1
