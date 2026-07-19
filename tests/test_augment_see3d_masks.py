import numpy as np
import torch

from scripts.augment_see3d_masks import filter_components, read_resized_mask


class _Camera:
    def __init__(self, center):
        center = np.asarray(center, dtype=np.float32)
        c2w = torch.eye(4)
        c2w[3, :3] = torch.from_numpy(center)
        self.world_view_transform = torch.linalg.inv(c2w)
        self.camera_center = torch.from_numpy(center)


def _reference_selector():
    import importlib.util
    from pathlib import Path
    import sys

    path = Path(__file__).parents[1] / "2d-gaussian-splatting/render_novel_views.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("render_novel_views_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_filter_components_keeps_only_bounded_regions():
    mask = np.zeros((100, 100), dtype=bool)
    mask[5:7, 5:7] = True
    mask[20:40, 20:40] = True
    mask[50:95, 50:95] = True

    filtered, components = filter_components(
        mask,
        min_area_fraction=0.01,
        max_component_fraction=0.10,
    )

    assert filtered.sum() == 400
    assert sum(component["accepted"] for component in components) == 1


def test_filter_components_returns_empty_for_empty_mask():
    filtered, components = filter_components(
        np.zeros((12, 9), dtype=bool),
        min_area_fraction=0.01,
        max_component_fraction=0.10,
    )

    assert not filtered.any()
    assert components == []


def test_external_valid_mask_is_inverted_to_artifact_mask(tmp_path):
    from PIL import Image

    valid = np.full((4, 6), 255, dtype=np.uint8)
    valid[1:3, 2:5] = 0
    path = tmp_path / "frame.valid_mask.png"
    Image.fromarray(valid, mode="L").save(path)

    artifact = read_resized_mask(path, (12, 8), invert=True)

    assert artifact.shape == (8, 12)
    assert artifact.any()
    assert not artifact[0, 0]


def test_pose_aware_reference_selection_covers_distant_targets():
    select = _reference_selector().select_reference_viewpoints
    references = [_Camera([0, 0, 0]), _Camera([1, 0, 0]), _Camera([9, 0, 0]), _Camera([10, 0, 0])]
    targets = [_Camera([0.2, 0, 0]), _Camera([9.8, 0, 0])]

    selected = select(references, targets, 2)

    centers = sorted(float(camera.camera_center[0]) for camera in selected)
    assert centers[0] <= 1.0
    assert centers[1] >= 9.0


def test_novel_view_quality_rejects_flat_no_edit_view(tmp_path):
    from PIL import Image

    module = _reference_selector()
    rgb = np.full((32, 48, 3), 80, dtype=np.uint8)
    known = np.full((32, 48), 255, dtype=np.uint8)
    Image.fromarray(rgb).save(tmp_path / "rgb.png")
    Image.fromarray(known).save(tmp_path / "mask.png")

    quality = module.novel_view_quality(tmp_path / "rgb.png", tmp_path / "mask.png")

    assert not quality["accepted"]
    assert quality["edit_fraction"] == 0.0
