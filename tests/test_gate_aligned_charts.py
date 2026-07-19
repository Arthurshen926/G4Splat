import numpy as np

from scripts.gate_aligned_charts import gate_charts


def test_gate_charts_zeroes_only_extreme_depth_conflicts():
    prior = np.ones((2, 4, 4), dtype=np.float32)
    depth = prior.copy()
    depth[1] = 2.0
    payload = {
        "depths": depth,
        "prior_depths": prior,
        "confs": np.ones_like(prior),
        "pts3d": np.zeros((2, 4, 4, 3), dtype=np.float32),
    }

    gated, records = gate_charts(
        payload,
        ["clean.png", "bad.png"],
        [np.ones((4, 4), dtype=bool), np.ones((4, 4), dtype=bool)],
        max_relative_p90=0.5,
        max_gt25_fraction=0.5,
        min_valid_fraction=0.05,
    )

    assert not records[0]["rejected"]
    assert records[1]["rejected"]
    assert np.all(gated["confs"][0] == 1)
    assert np.all(gated["confs"][1] == -2)
    assert np.all(gated["depths"][1] == 0)
    assert np.all(gated["prior_depths"][1] == 0)
    assert np.all(gated["pts3d"][1] == 0)
    assert np.all(gated["depths"][0] == 1)
    assert np.all(payload["confs"] == 1)
    assert np.all(payload["depths"][1] == 2)


def test_gate_charts_rejects_almost_empty_chart_and_records_reason():
    prior = np.ones((1, 10, 10), dtype=np.float32)
    payload = {
        "depths": prior.copy(),
        "prior_depths": prior,
        "confs": np.zeros_like(prior),
    }
    payload["confs"][0, :2, :2] = 1.0

    gated, records = gate_charts(
        payload,
        ["empty.png"],
        [np.ones((10, 10), dtype=bool)],
        max_relative_p90=0.5,
        max_gt25_fraction=0.5,
        min_valid_fraction=0.5,
    )

    assert records[0]["rejected"]
    assert records[0]["rejection_reasons"] == ["insufficient_support"]
    assert not gated["alignment_gate_valid"][0]


def test_gate_charts_hard_filters_joint_absolute_outliers():
    prior = np.ones((1, 4, 4), dtype=np.float32)
    depth = prior.copy()
    depth[0, 0, 0] = 100.0
    prior[0, 0, 0] = 100.0
    points = np.zeros((1, 4, 4, 3), dtype=np.float32)
    points[0, 0, 1, 0] = 100.0
    payload = {"depths": depth, "prior_depths": prior, "confs": np.ones_like(depth), "pts": points}

    gated, records = gate_charts(
        payload,
        ["joint_failure.png"],
        None,
        max_relative_p90=0.5,
        max_gt25_fraction=0.5,
        min_valid_fraction=0.05,
        max_abs_depth=50.0,
        max_abs_point=50.0,
    )

    assert not records[0]["rejected"]
    assert records[0]["absolute_outlier_fraction"] == 2 / 16
    assert gated["confs"][0, 0, 0] == -2
    assert gated["confs"][0, 0, 1] == -2
    assert gated["depths"][0, 0, 0] == 0
    assert gated["pts"][0, 0, 1, 0] == 0
