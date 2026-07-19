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
