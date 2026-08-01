import numpy as np
import pytest
import torch

from outdoor.foliage_geometry import (
    enforce_strict_foliage_ray_intervals,
    half_open_raster_coordinates,
    strict_foliage_hit_interval_bounds,
)
from outdoor.training_evidence import FoliageRayEvidence


def test_float32_endpoint_rounding_preserves_half_open_raster():
    pixels = np.asarray(
        [[1919.99999, 1079.99999], [0.0, 0.0]], dtype=np.float64
    )
    sizes = np.asarray([[1920, 1080], [1920, 1080]], dtype=np.int32)

    repaired = half_open_raster_coordinates(pixels, sizes)

    assert repaired.dtype == np.float32
    assert np.all(repaired >= 0)
    assert np.all(repaired < sizes)
    np.testing.assert_array_equal(repaired[1], np.asarray([0.0, 0.0]))


def test_materially_invalid_ray_coordinate_is_not_silently_clipped():
    with pytest.raises(RuntimeError, match="outside"):
        half_open_raster_coordinates(
            np.asarray([[1920.25, 100.0]], dtype=np.float32),
            np.asarray([[1920, 1080]], dtype=np.int32),
        )


def test_legacy_exact_endpoint_is_migrated_but_gross_error_is_rejected():
    payload = {
        "camera_ids": torch.tensor([7], dtype=torch.int32),
        "pixels": torch.tensor([[1375.0, 1080.0]]),
        "source_image_sizes": torch.tensor(
            [[1920, 1080]], dtype=torch.int32
        ),
        "free_end_depth": torch.tensor([4.0]),
        "hit_start_depth": torch.tensor([4.1]),
        "hit_end_depth": torch.tensor([4.3]),
        "observation_type": torch.tensor([-1], dtype=torch.int8),
        "confidence": torch.tensor([0.8]),
        "offsets": torch.tensor([0, 1], dtype=torch.int64),
        "depth_coordinate": "camera_z",
    }
    evidence = FoliageRayEvidence(payload)
    assert evidence.endpoint_rounding_repairs == 1
    assert float(evidence.pixels[0, 1]) < 1080.0

    payload["pixels"] = torch.tensor([[1375.0, 1080.25]])
    with pytest.raises(RuntimeError, match="outside"):
        FoliageRayEvidence(payload)


def test_hit_posterior_free_space_stops_at_lower_interval_bound():
    depth = np.asarray([5.0, 12.0])
    sigma = np.asarray([0.2, 2.0])

    free_end, hit_start, hit_end = strict_foliage_hit_interval_bounds(
        depth,
        sigma,
        free_space_margin=0.25,
    )

    assert np.all(free_end <= hit_start)
    np.testing.assert_array_less(hit_start, hit_end)
    # The uncertain row would previously overlap by 4.75 m.
    assert free_end[1] == pytest.approx(hit_start[1])
    assert hit_start[1] == pytest.approx(7.0)


def test_persisted_interval_repair_changes_hits_but_not_negative_rows():
    free = np.asarray([9.75, 4.0], dtype=np.float32)
    start = np.asarray([5.0, 3.0], dtype=np.float32)
    end = np.asarray([15.0, 6.0], dtype=np.float32)
    kind = np.asarray([1, -1], dtype=np.int8)

    repaired_free, repaired_start, repaired_end, audit = (
        enforce_strict_foliage_ray_intervals(
            free, start, end, kind
        )
    )

    assert repaired_free[0] == repaired_start[0] == 5.0
    assert repaired_end[0] == 15.0
    assert repaired_free[1] == free[1]
    assert repaired_start[1] == start[1]
    assert repaired_end[1] == end[1]
    assert audit["overlap_rows_before"] == 1
    assert audit["overlap_rows_after"] == 0
    assert audit["changed_hit_rows"] == 1


@pytest.mark.parametrize(
    ("free_end", "hit_start", "hit_end"),
    [
        (4.2, 4.1, 4.4),
        (-0.1, 4.1, 4.4),
        (4.0, 4.4, 4.4),
        (4.0, float("nan"), 4.4),
    ],
)
def test_runtime_rejects_invalid_hit_interval(
    free_end, hit_start, hit_end
):
    payload = {
        "camera_ids": torch.tensor([7], dtype=torch.int32),
        "pixels": torch.tensor([[100.0, 80.0]]),
        "source_image_sizes": torch.tensor(
            [[1920, 1080]], dtype=torch.int32
        ),
        "free_end_depth": torch.tensor([free_end]),
        "hit_start_depth": torch.tensor([hit_start]),
        "hit_end_depth": torch.tensor([hit_end]),
        "observation_type": torch.tensor([1], dtype=torch.int8),
        "confidence": torch.tensor([0.8]),
        "offsets": torch.tensor([0, 1], dtype=torch.int64),
        "depth_coordinate": "camera_z",
    }

    with pytest.raises(RuntimeError, match="strict"):
        FoliageRayEvidence(payload)


def test_ray_table_grouping_preserves_source_row_order_per_camera():
    camera_ids = torch.tensor([7, 3, 7, 3, 7], dtype=torch.int32)
    count = len(camera_ids)
    evidence = FoliageRayEvidence(
        {
            "camera_ids": camera_ids,
            "pixels": torch.tensor(
                [[10.0 + index, 20.0] for index in range(count)]
            ),
            "source_image_sizes": torch.tensor(
                [[1920, 1080]] * count, dtype=torch.int32
            ),
            "free_end_depth": torch.full((count,), 4.0),
            "hit_start_depth": torch.full((count,), 4.1),
            "hit_end_depth": torch.full((count,), 4.3),
            "observation_type": torch.ones(count, dtype=torch.int8),
            "confidence": torch.full((count,), 0.8),
            "offsets": torch.empty(0, dtype=torch.int64),
            "depth_coordinate": "camera_z",
        }
    )

    assert evidence.camera_id_values.tolist() == [3, 7]
    assert evidence.camera_rows[3].tolist() == [1, 3]
    assert evidence.camera_rows[7].tolist() == [0, 2, 4]
