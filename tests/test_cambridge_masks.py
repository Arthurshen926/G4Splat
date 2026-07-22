import json
import pickle

import torch

from matcha.cambridge_masks import CambridgeMaskLookup, CambridgeTreeWeightLookup


def test_mask_lookup_combines_dynamic_sky_and_valid_border(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "name_mapping.json").write_text(
        json.dumps({"seq1__frame00001.png": "seq1/frame00001.png"})
    )
    masks = (
        torch.tensor([[True, True], [False, True]]),
        torch.tensor([[True, False], [True, True]]),
        torch.tensor([[True, True], [True, False]]),
        torch.zeros((2, 2), dtype=torch.bool),
    )
    mask_pickle = tmp_path / "masks.pkl"
    with mask_pickle.open("wb") as handle:
        pickle.dump({"seq1/frame00001.png": masks}, handle)

    lookup = CambridgeMaskLookup(dataset, mask_pickle, mask_indices=[0, 1, 2])
    keep = lookup.get_mask("seq1__frame00001.png", (2, 2), torch.device("cpu"))

    assert torch.equal(keep, torch.tensor([[True, False], [False, False]]))


def test_mask_lookup_caches_combined_resized_cpu_mask(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "name_mapping.json").write_text(
        json.dumps({"seq1__frame00001.png": "seq1/frame00001.png"})
    )
    masks = tuple(torch.ones((2, 2), dtype=torch.bool) for _ in range(3))
    mask_pickle = tmp_path / "masks.pkl"
    with mask_pickle.open("wb") as handle:
        pickle.dump({"seq1/frame00001.png": masks}, handle)

    lookup = CambridgeMaskLookup(dataset, mask_pickle, mask_indices=[0, 1, 2])
    first = lookup.get_mask("seq1__frame00001.png", (4, 4), torch.device("cpu"))
    second = lookup.get_mask("seq1__frame00001.png", (4, 4), torch.device("cpu"))

    assert first.data_ptr() == second.data_ptr()
    assert len(lookup._resized_mask_cache) == 1


def test_mask_lookup_supports_external_adapter_staged_keyed_pickle(tmp_path):
    """The clean external adapter intentionally rekeys masks to COLMAP names."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "name_mapping.json").write_text(
        json.dumps({"seq1__frame00001.png": "seq1/frame00001.png"})
    )
    masks = tuple(torch.ones((2, 2), dtype=torch.bool) for _ in range(3))
    mask_pickle = tmp_path / "masks.pkl"
    with mask_pickle.open("wb") as handle:
        pickle.dump({"seq1__frame00001.png": masks}, handle)

    lookup = CambridgeMaskLookup(dataset, mask_pickle, mask_indices=[0, 1, 2])

    assert lookup.source_name_for("seq1__frame00001") == "seq1__frame00001.png"
    assert lookup.get_mask("seq1__frame00001", (2, 2), torch.device("cpu")).all()


def test_mask_lookup_decodes_staged_name_missing_from_subset_mapping(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "name_mapping.json").write_text("{}")
    masks = tuple(torch.ones((2, 2), dtype=torch.bool) for _ in range(3))
    mask_pickle = tmp_path / "masks.pkl"
    with mask_pickle.open("wb") as handle:
        pickle.dump({"seq4/frame00222.png": masks}, handle)

    lookup = CambridgeMaskLookup(dataset, mask_pickle, mask_indices=[0, 2])

    assert lookup.source_name_for("seq4__frame00222") == "seq4/frame00222.png"


def test_mask_lookup_with_indices_shares_loaded_pickle(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "name_mapping.json").write_text(
        json.dumps({"seq1__frame00001.png": "seq1/frame00001.png"})
    )
    masks = (
        torch.zeros((2, 2), dtype=torch.bool),
        torch.ones((2, 2), dtype=torch.bool),
        torch.zeros((2, 2), dtype=torch.bool),
    )
    mask_pickle = tmp_path / "masks.pkl"
    with mask_pickle.open("wb") as handle:
        pickle.dump({"seq1/frame00001.png": masks}, handle)

    lookup = CambridgeMaskLookup(dataset, mask_pickle, mask_indices=[0, 1, 2])
    sky_lookup = lookup.with_indices([1])

    assert sky_lookup.masks is lookup.masks
    assert sky_lookup.staged_to_source is lookup.staged_to_source
    assert sky_lookup.mask_indices == [1]
    assert sky_lookup.get_mask("seq1__frame00001.png", (2, 2), torch.device("cpu")).all()


def test_tree_weights_preserve_building_and_soft_weight_tree(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "name_mapping.json").write_text(
        json.dumps({"seq1__frame00001.png": "seq1/frame00001.png"})
    )
    masks = (
        torch.ones((3, 3), dtype=torch.bool),
        torch.ones((3, 3), dtype=torch.bool),
        torch.ones((3, 3), dtype=torch.bool),
        torch.tensor([[True, True, True], [True, False, True], [True, True, True]]),
    )
    mask_pickle = tmp_path / "masks.pkl"
    with mask_pickle.open("wb") as handle:
        pickle.dump({"seq1/frame00001.png": masks}, handle)
    support_dir = tmp_path / "support"
    support_dir.mkdir()
    import numpy as np
    np.save(support_dir / "seq1__frame00001.npy", np.ones((3, 3), dtype=np.float32))

    lookup = CambridgeMaskLookup(dataset, mask_pickle, mask_indices=[0, 1, 2])
    weights = CambridgeTreeWeightLookup(
        lookup,
        support_dir=support_dir,
        sky_feather_pixels=0,
        tree_feather_pixels=0,
    )
    rgb, geometry, planar = weights.weights(
        "seq1__frame00001.png", (3, 3), torch.device("cpu")
    )

    assert rgb[0, 0] == 1.0
    assert torch.isclose(rgb[1, 1], torch.tensor(0.75))
    assert torch.isclose(geometry[1, 1], torch.tensor(0.30))
    assert planar[1, 1] == 0.0

    audit = weights.support_coverage_audit(["seq1__frame00001.png"])
    assert audit["nonzero_map_count"] == 1
    assert audit["zero_support_map_count"] == 0
    assert audit["nonzero_map_coverage_fraction"] == 1.0
    assert audit["mean_support_value_across_present_maps"] == 1.0
    assert audit["max_support_value_across_present_maps"] == 1.0
    assert audit["explicit_map_behavior"] == "all_explicit_maps_have_nonzero_support"


def test_tree_support_coverage_distinguishes_missing_evidence_from_zero_support(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "name_mapping.json").write_text(json.dumps({
        "seq1__frame00001.png": "seq1/frame00001.png",
        "seq1__frame00002.png": "seq1/frame00002.png",
    }))
    masks = tuple(torch.ones((2, 2), dtype=torch.bool) for _ in range(4))
    mask_pickle = tmp_path / "masks.pkl"
    with mask_pickle.open("wb") as handle:
        pickle.dump({
            "seq1/frame00001.png": masks,
            "seq1/frame00002.png": masks,
        }, handle)
    support_dir = tmp_path / "support"
    support_dir.mkdir()
    import numpy as np
    np.save(support_dir / "seq1__frame00001.npy", np.zeros((2, 2), dtype=np.float32))

    weights = CambridgeTreeWeightLookup(
        CambridgeMaskLookup(dataset, mask_pickle, mask_indices=[0, 1, 2]),
        support_dir=support_dir,
    )
    audit = weights.support_coverage_audit([
        "seq1__frame00001.png", "seq1__frame00002.png"
    ])

    assert audit["present_map_count"] == 1
    assert audit["missing_map_count"] == 1
    assert audit["map_coverage_fraction"] == 0.5
    assert audit["nonzero_map_count"] == 0
    assert audit["zero_support_map_count"] == 1
    assert audit["nonzero_map_coverage_fraction"] == 0.0
    assert audit["mean_support_value_across_present_maps"] == 0.0
    assert audit["max_support_value_across_present_maps"] == 0.0
    assert audit["explicit_map_behavior"] == "all_explicit_maps_zero_support_tree_floor"
    assert audit["missing_view_examples"] == ["seq1__frame00002.png"]
    assert audit["missing_map_behavior"] == "zero_support_tree_floor_not_canonical_evidence"


def test_tree_neutral_support_policy_does_not_numerically_merge_unknown_with_zero(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "name_mapping.json").write_text(
        json.dumps({"seq1__frame00001.png": "seq1/frame00001.png"})
    )
    masks = tuple(torch.ones((2, 2), dtype=torch.bool) for _ in range(4))
    mask_pickle = tmp_path / "masks.pkl"
    with mask_pickle.open("wb") as handle:
        pickle.dump({"seq1/frame00001.png": masks}, handle)

    weights = CambridgeTreeWeightLookup(
        CambridgeMaskLookup(dataset, mask_pickle, mask_indices=[0, 1, 2]),
        support_dir=tmp_path / "missing_support",
        missing_support_policy="neutral",
        neutral_support_value=0.5,
    )

    support = weights.support("seq1__frame00001.png", (2, 2), torch.device("cpu"))
    audit = weights.support_coverage_audit(["seq1__frame00001.png"])

    assert torch.allclose(support, torch.full((2, 2), 0.5))
    assert weights.support_valid("seq1__frame00001.png") is False
    assert audit["missing_map_behavior"] == "neutral_prior_unknown_evidence"
    assert audit["unknown_evidence_is_distinct_from_zero_support"] is True
