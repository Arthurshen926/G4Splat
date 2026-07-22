from scripts.compare_render_metrics import build_metric_comparison


def _view(*, raw, static, tree=None):
    payload = {
        "psnr": raw,
        "ssim": raw / 100.0,
        "mae": 1.0 / raw,
        "rmse": 2.0 / raw,
        "masked": {
            "dynamic_valid": {"psnr": raw, "ssim": raw / 100.0, "mae": 1.0, "rmse": 2.0},
            "static_valid": {"psnr": static, "ssim": static / 100.0, "mae": 1.0, "rmse": 2.0},
        },
        "ulfloc_legacy": {"psnr": raw, "ssim": raw / 100.0, "mae": 1.0, "rmse": 2.0},
        "tree_stratified": {"non_tree_static": {"psnr": static, "ssim": static / 100.0, "mae": 1.0, "rmse": 2.0}},
    }
    if tree is not None:
        payload["tree_stratified"]["tree_static"] = {
            "psnr": tree,
            "ssim": tree / 100.0,
            "mae": 1.0,
            "rmse": 2.0,
        }
    return payload


def _report(view_a, view_b):
    return {
        "mean": {"psnr": 10.0, "ssim": 0.1, "mae": 0.1, "rmse": 0.2},
        "masked": {
            "dynamic_valid": {"mean": {"psnr": 10.0, "ssim": 0.1, "mae": 0.1, "rmse": 0.2}},
            "static_valid": {"mean": {"psnr": 10.0, "ssim": 0.1, "mae": 0.1, "rmse": 0.2}},
        },
        "protocol": {"ulfloc_legacy": {"mean": {"psnr": 10.0, "ssim": 0.1, "mae": 0.1, "rmse": 0.2}}},
        "tree_stratified": {
            "non_tree_static": {"mean": {"psnr": 10.0, "ssim": 0.1, "mae": 0.1, "rmse": 0.2}},
            "tree_static": {"mean": {"psnr": 10.0, "ssim": 0.1, "mae": 0.1, "rmse": 0.2}},
        },
        "per_view": {"a.png": view_a, "b.png": view_b},
    }


def test_comparison_is_paired_and_handles_missing_tree_support():
    baseline = _report(_view(raw=10.0, static=10.0, tree=10.0), _view(raw=20.0, static=20.0))
    candidate = _report(_view(raw=11.0, static=12.0, tree=13.0), _view(raw=18.0, static=18.0))

    comparison = build_metric_comparison(
        baseline,
        candidate,
        bootstrap_samples=200,
        seed=0,
    )

    assert comparison["view_identity"]["shared_view_count"] == 2
    assert comparison["metrics"]["static_valid"]["psnr"]["paired_view_count"] == 2
    assert comparison["metrics"]["static_valid"]["psnr"]["paired_mean_delta"] == 0.0
    # Only a.png has valid tree support in both reports.
    assert comparison["metrics"]["tree_static"]["psnr"]["paired_view_count"] == 1
    assert comparison["metrics"]["tree_static"]["psnr"]["paired_mean_delta"] == 3.0


def test_comparison_pairs_native_indices_with_external_names_by_source_image():
    baseline = _report(_view(raw=10.0, static=10.0), _view(raw=20.0, static=20.0))
    candidate = _report(_view(raw=11.0, static=11.0), _view(raw=21.0, static=21.0))
    baseline["per_view"] = {
        "00000.png": {**baseline["per_view"]["a.png"], "source_image": "seq1/frame00001.png"},
        "00001.png": {**baseline["per_view"]["b.png"], "source_image": "seq1/frame00002.png"},
    }
    candidate["per_view"] = {
        "seq1__frame00001.png": {
            **candidate["per_view"]["a.png"],
            "source_image": "seq1/frame00001.png",
        },
        "seq1__frame00002.png": {
            **candidate["per_view"]["b.png"],
            "source_image": "seq1/frame00002.png",
        },
    }

    comparison = build_metric_comparison(
        baseline,
        candidate,
        bootstrap_samples=200,
        seed=0,
    )

    assert comparison["view_identity"]["shared_view_count"] == 2
    assert comparison["metrics"]["static_valid"]["psnr"]["paired_mean_delta"] == 1.0


def test_comparison_normalizes_matcha_root_level_masked_per_view_metrics():
    # Older MAtCha reports place masks in a parallel root-level map.  Its raw
    # per-view record still supplies the source-image identity used for
    # pairing with G4's current nested representation.
    matcha = {
        "mean": {"psnr": 10.0, "ssim": 0.1, "mae": 0.1, "rmse": 0.2},
        "masked": {
            "dynamic_valid": {
                "mean": {"psnr": 10.0, "ssim": 0.1, "mae": 0.1, "rmse": 0.2},
                "per_view": {
                    "00000.png": {"psnr": 12.0, "ssim": 0.12, "mae": 0.1, "rmse": 0.2}
                },
            },
            "static_valid": {
                "mean": {"psnr": 10.0, "ssim": 0.1, "mae": 0.1, "rmse": 0.2},
                "per_view": {
                    "00000.png": {"psnr": 13.0, "ssim": 0.13, "mae": 0.1, "rmse": 0.2}
                },
            },
        },
        "per_view": {
            "00000.png": {"psnr": 11.0, "ssim": 0.11, "mae": 0.1, "rmse": 0.2, "source_image": "seq1/frame00001.png"}
        },
    }
    g4 = {
        "mean": {"psnr": 10.0, "ssim": 0.1, "mae": 0.1, "rmse": 0.2},
        "masked": {
            "dynamic_valid": {"mean": {"psnr": 10.0, "ssim": 0.1, "mae": 0.1, "rmse": 0.2}},
            "static_valid": {"mean": {"psnr": 10.0, "ssim": 0.1, "mae": 0.1, "rmse": 0.2}},
        },
        "per_view": {
            "seq1__frame00001.png": {
                "psnr": 11.0,
                "ssim": 0.11,
                "mae": 0.1,
                "rmse": 0.2,
                "source_image": "seq1/frame00001.png",
                "masked": {
                    "dynamic_valid": {"psnr": 14.0, "ssim": 0.14, "mae": 0.1, "rmse": 0.2},
                    "static_valid": {"psnr": 16.0, "ssim": 0.16, "mae": 0.1, "rmse": 0.2},
                },
            }
        },
    }

    comparison = build_metric_comparison(matcha, g4, bootstrap_samples=200, seed=0)

    assert comparison["metrics"]["dynamic_valid"]["psnr"]["paired_mean_delta"] == 2.0
    assert comparison["metrics"]["static_valid"]["psnr"]["paired_mean_delta"] == 3.0


def test_comparison_refuses_to_label_different_population_aggregates_as_a_delta():
    baseline = _report(_view(raw=10.0, static=10.0), _view(raw=20.0, static=20.0))
    candidate = _report(_view(raw=11.0, static=11.0), _view(raw=21.0, static=21.0))
    candidate["per_view"]["extra.png"] = _view(raw=100.0, static=100.0)
    candidate["mean"]["psnr"] = 44.0

    comparison = build_metric_comparison(baseline, candidate, bootstrap_samples=200, seed=0)
    result = comparison["metrics"]["raw"]["psnr"]

    assert not result["aggregate_comparable"]
    assert result["aggregate_delta"] is None
    assert result["paired_baseline_mean"] == 15.0
    assert result["paired_candidate_mean"] == 16.0
    assert result["paired_mean_delta"] == 1.0
