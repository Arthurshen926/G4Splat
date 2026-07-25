import numpy as np

from outdoor.surfel_uv_bake import (
    ROLE_RIGID_EVIDENCE,
    ROLE_UV_RETAINED,
    bake_surfel_uv_ownership,
)


def _vertices(count=3):
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
        ("geometry_confidence", "f4"),
        ("protected_flag", "f4"),
        ("primitive_class", "f4"),
    ]
    value = np.zeros(count, dtype=dtype)
    value["x"] = np.arange(count, dtype=np.float32) * 10
    value["z"] = 5
    value["opacity"] = np.log(0.8 / 0.2)
    value["rot_0"] = 1
    value["geometry_confidence"] = 0.2
    return value


def _statistics(count=3):
    return {
        "canopy_responsibility": np.zeros(count, dtype=np.float32),
        "rigid_responsibility": np.ones(count, dtype=np.float32),
        "maximum_view_rigid_responsibility": np.ones(
            count, dtype=np.float32
        ),
        "support_views": np.full(count, 4, dtype=np.int16),
        "maximum_projected_radius": np.full(
            count, 100, dtype=np.float32
        ),
        "minimum_camera_depth": np.full(count, 5, dtype=np.float32),
    }


def test_uv_equivalent_replaces_parent_and_preserves_provenance():
    vertices = _vertices()
    atlas = np.zeros((1, 3, 3), dtype=np.float32)
    atlas[0, 1, 2] = 1
    evidence = np.zeros((1, 3, 3, 4), dtype=np.float32)
    weight = np.ones((1, 3, 3), dtype=np.float32)
    result = bake_surfel_uv_ownership(
        vertices,
        np.array([1]),
        atlas,
        evidence,
        weight,
        _statistics(),
        mode="uv-equivalent",
        child_sigma_over_spacing=0.5,
        minimum_child_peak_alpha=1e-5,
    )
    child = result.vertices[-1]
    assert len(result.vertices) == 3
    assert result.source_parent_index[-1] == 1
    assert result.source_uv_x[-1] == 2
    assert result.source_uv_y[-1] == 1
    assert result.ownership_role[-1] == ROLE_UV_RETAINED
    # Identity rotation and unit parent scale: u=+3 moves along world x.
    assert np.isclose(child["x"], 13.0)
    assert np.isclose(child["y"], 0.0)
    assert np.isclose(
        child["scale_0"], np.log(1.5), atol=1e-6
    )


def test_rigid_clean_removes_canopy_only_and_unsafe_candidates():
    vertices = _vertices()
    statistics = _statistics()
    statistics["canopy_responsibility"][0] = 0.9
    statistics["rigid_responsibility"][0] = 0.0
    statistics["maximum_view_rigid_responsibility"][0] = 0.0
    statistics["maximum_projected_radius"][1] = 5000
    atlas = np.ones((1, 3, 3), dtype=np.float32)
    evidence = np.zeros((1, 3, 3, 4), dtype=np.float32)
    evidence[..., 1] = 1
    weight = np.ones((1, 3, 3), dtype=np.float32)
    result = bake_surfel_uv_ownership(
        vertices,
        np.array([1]),
        atlas,
        evidence,
        weight,
        statistics,
        mode="rigid-clean",
        maximum_bake_radius=4096,
        minimum_child_peak_alpha=1e-5,
    )
    assert result.source_parent_index.tolist() == [2]
    assert result.audit["removed_global_canopy_only_count"] == 1
    assert result.audit["unsafe_candidate_parent_count"] == 1
    assert result.audit["baked_child_count"] == 0


def test_rigid_clean_removes_historical_foliage_primitives():
    vertices = _vertices()
    vertices["primitive_class"][0] = 1
    result = bake_surfel_uv_ownership(
        vertices,
        np.array([], dtype=np.int64),
        np.ones((0, 3, 3), dtype=np.float32),
        np.zeros((0, 3, 3, 4), dtype=np.float32),
        np.zeros((0, 3, 3), dtype=np.float32),
        _statistics(),
        mode="rigid-clean",
    )
    assert result.source_parent_index.tolist() == [1, 2]
    assert result.audit["removed_nonstructural_count"] == 1


def test_rigid_clean_keeps_only_rigid_dominant_cells():
    vertices = _vertices()
    atlas = np.ones((1, 3, 3), dtype=np.float32)
    evidence = np.zeros((1, 3, 3, 4), dtype=np.float32)
    weight = np.ones((1, 3, 3), dtype=np.float32)
    evidence[0, 1, 1, 0] = 0.1
    evidence[0, 1, 1, 1] = 0.7
    evidence[0, 1, 2, 0] = 0.9
    evidence[0, 1, 2, 1] = 0.1
    result = bake_surfel_uv_ownership(
        vertices,
        np.array([1]),
        atlas,
        evidence,
        weight,
        _statistics(),
        mode="rigid-clean",
        minimum_child_peak_alpha=1e-5,
    )
    child_rows = result.ownership_role == ROLE_RIGID_EVIDENCE
    assert child_rows.sum() == 1
    assert result.source_uv_x[child_rows].item() == 1
    assert result.source_uv_y[child_rows].item() == 1
    assert result.vertices["protected_flag"][child_rows].item() == 1
