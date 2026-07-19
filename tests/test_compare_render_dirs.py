from scripts.compare_render_dirs import select_views


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
