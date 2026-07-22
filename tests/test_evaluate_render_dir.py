import json
from pathlib import Path

import pytest
import torch

from scripts.evaluate_render_dir import (
    mask_key_for_source,
    mask_resolution_audit,
    render_source_mapping,
    staged_mask_key_by_source,
    tree_static_strata,
)


def test_subset_render_mapping_uses_staged_filename_not_position(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "name_mapping.json").write_text(
        json.dumps(
            {
                "seq4__frame00037.png": "seq4/frame00037.png",
                "seq4__frame00038.png": "seq4/frame00038.png",
                "seq12__frame00097.png": "seq12/frame00097.png",
            }
        )
    )

    mapped = render_source_mapping(
        dataset,
        ["seq12__frame00097.png", "seq4__frame00038.png"],
    )

    assert mapped == {
        "seq12__frame00097.png": "seq12/frame00097.png",
        "seq4__frame00038.png": "seq4/frame00038.png",
    }


def test_lazy_indexed_render_mapping_uses_camera_index_not_file_position(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "name_mapping.json").write_text(
        json.dumps(
            {
                "seq10__frame00001.png": "seq10/frame00001.png",
                "seq12__frame00097.png": "seq12/frame00097.png",
                "seq4__frame00037.png": "seq4/frame00037.png",
            }
        )
    )

    mapped = render_source_mapping(dataset, ["00002.png", "00000.png"])

    assert mapped == {
        "00002.png": "seq4/frame00037.png",
        "00000.png": "seq10/frame00001.png",
    }


def test_unknown_full_cardinality_render_names_are_not_positionally_guessed(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "name_mapping.json").write_text(
        json.dumps(
            {
                "seq10__frame00001.png": "seq10/frame00001.png",
                "seq12__frame00097.png": "seq12/frame00097.png",
            }
        )
    )

    with pytest.raises(RuntimeError, match="without guessing"):
        render_source_mapping(dataset, ["foreign_a.png", "foreign_b.png"])


def test_staged_adapter_mask_keys_resolve_from_rendered_source_identity(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "name_mapping.json").write_text(
        json.dumps({"seq12__frame00097.png": "seq12/frame00097.png"})
    )
    staged_by_source = staged_mask_key_by_source(dataset)

    key, mode = mask_key_for_source(
        {"seq12__frame00097.png": (torch.ones((2, 2), dtype=torch.bool),)},
        source_name="seq12/frame00097.png",
        staged_by_source=staged_by_source,
        mask_pickle=tmp_path / "adapter_masks.pkl",
    )

    assert key == "seq12__frame00097.png"
    assert mode == "staged_key"


def test_raw_source_mask_key_remains_preferred_when_both_layouts_exist(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "name_mapping.json").write_text(
        json.dumps({"seq12__frame00097.png": "seq12/frame00097.png"})
    )
    staged_by_source = staged_mask_key_by_source(dataset)
    masks = {
        "seq12/frame00097.png": (torch.ones((2, 2), dtype=torch.bool),),
        "seq12__frame00097.png": (torch.zeros((2, 2), dtype=torch.bool),),
    }

    key, mode = mask_key_for_source(
        masks,
        source_name="seq12/frame00097.png",
        staged_by_source=staged_by_source,
        mask_pickle=tmp_path / "mixed_masks.pkl",
    )

    assert key == "seq12/frame00097.png"
    assert mode == "source_key"


def test_mask_resolution_audit_marks_resampled_ulfloc_style_metrics():
    audit = mask_resolution_audit(
        raw_mask_shape_counts={(360, 640): 3},
        rendered_shape_counts={(540, 960): 1},
        views_requiring_resampling=1,
    )

    assert not audit["native_pixel_geometry"]
    assert audit["contract"] == "nearest_resampled_mask_pixels"


def test_tree_static_strata_separates_tree_from_ordinary_static_pixels():
    static_keep = torch.tensor([[True, True], [False, True]])
    tree_keep = torch.tensor([[True, False], [True, False]])

    strata = tree_static_strata(static_keep, tree_keep)

    assert torch.equal(
        strata["non_tree_static"],
        torch.tensor([[True, False], [False, False]]),
    )
    assert torch.equal(
        strata["tree_static"],
        torch.tensor([[False, True], [False, True]]),
    )
