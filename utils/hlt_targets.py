"""Explicit, opt-in target taxonomies for HLT background training.

The converted train3M sample always stores the original four parent labels and
the finer generator category.  The baseline therefore remains unchanged, while
``decay8`` can derive W/Z decay-channel targets without rewriting the data.
HT and QCD hard-pT bins remain strata inside a physics class, not targets.
"""

from __future__ import annotations

from typing import Mapping, Tuple

import torch


TARGET_SCHEMES = ("parent4", "decay8")

PARENT4_NAMES = {
    0: "Z+jets",
    1: "QCD",
    2: "ttbar",
    3: "W+jets",
}

# Keep QCD and ttbar at their historical indices.  This preserves qcd_label=1
# throughout the existing ABCD code while making the decay labels explicit.
DECAY8_NAMES = {
    0: "Z->ee",
    1: "QCD",
    2: "ttbar",
    3: "W->enu",
    4: "W->munu",
    5: "W->taunu",
    6: "Z->mumu",
    7: "Z->tautau",
}


def _vector(sample: Mapping, key: str, maximum_events: int) -> torch.Tensor:
    if key not in sample:
        raise KeyError(
            f"Target construction requires the event-aligned field {key!r}.")
    value = torch.as_tensor(sample[key]).long().reshape(-1).cpu()
    if maximum_events > 0:
        value = value[:maximum_events]
    return value


def derive_target_labels(
    sample: Mapping,
    scheme: str = "parent4",
    maximum_events: int = -1,
) -> Tuple[torch.Tensor, dict]:
    """Return event labels and a serializable taxonomy description.

    ``parent4`` returns the historical Z/QCD/ttbar/W labels. ``decay8`` uses
    ``background_category`` to split W and Z by boson decay while deliberately
    collapsing their HT bins.  QCD pT bins and ttbar decay modes also remain
    within their parent target for this comparison.
    """
    if scheme not in TARGET_SCHEMES:
        raise ValueError(
            f"target scheme must be one of {TARGET_SCHEMES}, got {scheme!r}.")

    parent = _vector(sample, "label", maximum_events)
    if scheme == "parent4":
        unknown = sorted(set(parent.tolist()) - set(PARENT4_NAMES))
        if unknown:
            raise ValueError(f"parent4 found unknown parent labels: {unknown}")
        return parent.clone(), {
            "scheme": scheme,
            "class_names": dict(PARENT4_NAMES),
            "qcd_label": 1,
            "source_parent_qcd_label": 1,
            "definition": "Z+jets, QCD, ttbar, W+jets",
        }

    category = _vector(sample, "background_category", maximum_events)
    if category.numel() != parent.numel():
        raise ValueError(
            "background_category and parent label lengths do not match.")

    target = torch.full_like(parent, -1)
    expected_parent = torch.full_like(parent, -1)

    definitions = (
        ((category >= 15) & (category <= 17), 0, 0),  # Z -> ee
        ((category >= 0) & (category <= 5), 1, 1),    # QCD pT bins
        ((category >= 27) & (category <= 36), 2, 2),  # ttbar modes
        ((category >= 6) & (category <= 8), 3, 3),    # W -> e nu
        ((category >= 9) & (category <= 11), 4, 3),   # W -> mu nu
        ((category >= 12) & (category <= 14), 5, 3),  # W -> tau nu
        ((category >= 18) & (category <= 20), 6, 0),  # Z -> mu mu
        ((category >= 21) & (category <= 23), 7, 0),  # Z -> tau tau
    )
    for mask, target_label, parent_label in definitions:
        target[mask] = target_label
        expected_parent[mask] = parent_label

    invalid = target < 0
    if invalid.any():
        unknown = sorted(set(category[invalid].tolist()))
        raise ValueError(
            "decay8 cannot map background categories: " +
            ", ".join(str(value) for value in unknown))
    inconsistent = parent != expected_parent
    if inconsistent.any():
        examples = sorted(set(zip(
            category[inconsistent].tolist(), parent[inconsistent].tolist())))
        raise ValueError(
            "Parent labels disagree with background categories; examples "
            f"(category, parent)={examples[:8]}")

    return target, {
        "scheme": scheme,
        "class_names": dict(DECAY8_NAMES),
        "qcd_label": 1,
        "source_parent_qcd_label": 1,
        "definition": (
            "Z->ee, QCD, ttbar, W->enu, W->munu, W->taunu, "
            "Z->mumu, Z->tautau; HT/pT bins remain strata"),
    }


def checkpoint_target_labels(sample: Mapping, preprocessing: Mapping) -> torch.Tensor:
    """Reproduce the exact classifier labels stored in a training checkpoint."""
    scheme = str(preprocessing.get("target_scheme", "parent4"))
    labels, _metadata = derive_target_labels(sample, scheme=scheme)
    raw_map = preprocessing.get("label_map")
    if not isinstance(raw_map, Mapping):
        raise ValueError("Checkpoint preprocessing is missing label_map.")
    label_map = {int(key): int(value) for key, value in raw_map.items()}
    unknown = sorted(set(labels.tolist()) - set(label_map))
    if unknown:
        raise ValueError(
            f"Evaluation sample contains target labels absent from checkpoint: {unknown}")
    remapped = torch.empty_like(labels)
    for source, destination in label_map.items():
        remapped[labels == source] = destination
    return remapped
