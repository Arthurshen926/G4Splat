import numpy as np

from scripts.compare_render_dirs import select_views, tree_crop_box


def test_comparison_selects_worst_and_regressed_views_without_duplicates():
    baseline = {
        "per_view": {
            "00000.png": {"psnr": 20.0},
            "00001.png": {"psnr": 30.0},
            "00002.png": {"psnr": 21.0},
        }
    }
    candidate = {
        "per_view": {
            "00000.png": {"psnr": 19.0},
            "00001.png": {"psnr": 15.0},
            "00002.png": {"psnr": 22.0},
        }
    }

    selected = select_views(baseline, candidate, count=3)

    assert selected == ["00001.png", "00000.png", "00002.png"]


def test_comparison_can_select_largest_psnr_gains():
    baseline = {
        "per_view": {
            "00000.png": {"psnr": 20.0},
            "00001.png": {"psnr": 30.0},
            "00002.png": {"psnr": 21.0},
        }
    }
    candidate = {
        "per_view": {
            "00000.png": {"psnr": 19.0},
            "00001.png": {"psnr": 30.5},
            "00002.png": {"psnr": 23.0},
        }
    }

    selected = select_views(baseline, candidate, count=2, strategy="best_gain")

    assert selected == ["00002.png", "00001.png"]


def test_comparison_can_rank_tree_static_gains_and_skip_empty_tree_views():
    baseline = {
        "per_view": {
            "00000.png": {
                "psnr": 40.0,
                "tree_stratified": {"tree_static": {"psnr": 16.0}},
            },
            "00001.png": {
                "psnr": 10.0,
                "tree_stratified": {"tree_static": {"psnr": 20.0}},
            },
            "00002.png": {"psnr": 30.0, "tree_stratified": {}},
        }
    }
    candidate = {
        "per_view": {
            "00000.png": {
                "psnr": 39.0,
                "tree_stratified": {"tree_static": {"psnr": 19.0}},
            },
            "00001.png": {
                "psnr": 12.0,
                "tree_stratified": {"tree_static": {"psnr": 20.5}},
            },
            "00002.png": {"psnr": 35.0, "tree_stratified": {}},
        }
    }

    selected = select_views(
        baseline, candidate, count=2, strategy="tree_best_gain"
    )

    assert selected == ["00000.png", "00001.png"]


def test_comparison_rejects_tree_selection_when_all_tree_masks_are_empty():
    metrics = {"per_view": {"00000.png": {"psnr": 20.0}}}

    try:
        select_views(metrics, metrics, count=1, strategy="tree_worst")
    except ValueError as error:
        assert "No common views" in str(error)
    else:
        raise AssertionError("Expected empty tree selection to fail explicitly")


def test_tree_crop_tracks_the_densest_tree_region():
    tree_keep = np.ones((60, 100), dtype=bool)
    tree_keep[40:58, 75:98] = False

    left, top, right, bottom = tree_crop_box(tree_keep, (100, 60))

    assert left > 0
    assert top > 0
    assert left <= 75 < right
    assert top <= 40 < bottom
