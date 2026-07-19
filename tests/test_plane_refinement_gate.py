import numpy as np

from scripts.gate_plane_refinement import gate_plane_depth


def test_plane_gate_falls_back_whole_chart_on_low_support():
    aligned = np.ones((10, 10), dtype=np.float32)
    refined = aligned * 2.0
    support = np.zeros((10, 10), dtype=bool)
    support[:2] = True

    output, confidence, record = gate_plane_depth(
        aligned,
        refined,
        support,
        min_support_fraction=0.5,
        max_relative_p90=0.25,
        max_gt25_fraction=0.25,
        max_pixel_relative_change=0.5,
    )

    assert record["chart_fallback"]
    assert "insufficient_support" in record["reasons"]
    assert np.allclose(output, aligned)
    assert np.array_equal(confidence, support)


def test_plane_gate_keeps_safe_changes_and_reverts_local_outlier():
    aligned = np.ones((10, 10), dtype=np.float32)
    refined = aligned * 1.05
    refined[5, 5] = 3.0
    support = np.ones((10, 10), dtype=bool)

    output, _, record = gate_plane_depth(
        aligned,
        refined,
        support,
        min_support_fraction=0.5,
        max_relative_p90=0.25,
        max_gt25_fraction=0.25,
        max_pixel_relative_change=0.5,
    )

    assert not record["chart_fallback"]
    assert output[5, 5] == aligned[5, 5]
    assert np.isclose(output[0, 0], refined[0, 0])
