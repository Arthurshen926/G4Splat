import numpy as np

from outdoor.inverse_depth import fuse_inverse_depth_sources


def test_world_metric_sources_fuse_without_hidden_scale_change():
    plane = np.full((4, 4), 12.0, dtype=np.float32)
    chart = np.full((4, 4), 12.0, dtype=np.float32)
    result = fuse_inverse_depth_sources(
        plane,
        np.ones_like(plane),
        chart,
        np.ones_like(chart),
        None,
        plane_valid_mask=np.ones_like(plane, dtype=bool),
        plane_support_view_count=np.full_like(plane, 3, dtype=np.uint8),
        chart_support_view_count=np.ones_like(plane, dtype=np.uint8),
    )
    assert np.allclose(result["depth"], 12.0, atol=1e-5)
    assert np.all(result["support_view_count"] == 3)
