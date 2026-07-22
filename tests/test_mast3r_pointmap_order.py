from matcha.pointmap.ordering import pointmap_paths_for_camera_filepaths


def test_pointmaps_follow_camera_identity_not_filesystem_sort_order(tmp_path):
    pointmaps = tmp_path / "pointmaps"
    pointmaps.mkdir()
    (pointmaps / "alpha.json").write_text("{}")
    (pointmaps / "bravo.json").write_text("{}")

    resolved = pointmap_paths_for_camera_filepaths(
        ["/charts/bravo.png", "/charts/alpha.png"],
        pointmaps,
    )

    assert [path.name for path in resolved] == ["bravo.json", "alpha.json"]


def test_pointmap_resolution_rejects_missing_camera_identity(tmp_path):
    pointmaps = tmp_path / "pointmaps"
    pointmaps.mkdir()
    (pointmaps / "alpha.json").write_text("{}")

    try:
        pointmap_paths_for_camera_filepaths(["missing.png"], pointmaps)
    except FileNotFoundError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("missing pointmap identity must be rejected")
