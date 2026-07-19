from pathlib import Path

import pytest

from scripts.prepare_matcha_warmstart import prepare_adapter


def _write_ply(path: Path) -> None:
    path.parent.mkdir(parents=True)
    path.write_bytes(
        b"ply\n"
        b"format binary_little_endian 1.0\n"
        b"element vertex 0\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
        b"property float opacity\nproperty float scale_0\nproperty float scale_1\n"
        b"property float rot_0\nproperty float rot_1\n"
        b"property float rot_2\nproperty float rot_3\n"
        b"property float mip_filter\nend_header\n"
    )


def _fixture(tmp_path: Path):
    scene = tmp_path / "matcha_scene"
    model = tmp_path / "matcha_model"
    (scene / "images").mkdir(parents=True)
    from PIL import Image

    Image.new("RGB", (1920, 1080)).save(scene / "images" / "frame.png")
    (scene / "sparse").mkdir()
    (scene / "pointmaps").mkdir()
    (scene / "charts_data.npz").write_bytes(b"charts")
    model.mkdir()
    (model / "cfg_args").write_text(
        f"Namespace(sh_degree=3, source_path={str(scene)!r}, "
        f"model_path={str(model)!r}, images='images', resolution=-1, "
        "white_background=False, data_device='cpu', eval=False)"
    )
    _write_ply(model / "point_cloud" / "iteration_30000" / "point_cloud.ply")
    return scene, model


def test_adapter_isolates_mutable_chart_data_and_links_large_inputs(tmp_path):
    scene, model = _fixture(tmp_path)
    output = tmp_path / "adapter"

    manifest = prepare_adapter(
        matcha_scene=scene,
        matcha_model=model,
        output=output,
        iteration=30000,
    )

    adapter_scene = output / "mast3r_sfm"
    assert not (adapter_scene / "charts_data.npz").is_symlink()
    assert (adapter_scene / "images").is_symlink()
    assert (adapter_scene / "sparse").is_symlink()
    assert (output / "baseline_model" / "point_cloud" / "iteration_30000" / "point_cloud.ply").is_symlink()
    assert manifest["ply"]["has_mip_filter"] is True
    assert manifest["legacy_resolution_translation"] == 1600
    assert "resolution=1600" in (output / "baseline_model" / "cfg_args").read_text()


def test_adapter_rejects_a_model_from_another_coordinate_frame(tmp_path):
    scene, model = _fixture(tmp_path)
    other_scene = tmp_path / "other_scene"
    other_scene.mkdir()

    with pytest.raises(ValueError, match="coordinate mismatch"):
        prepare_adapter(
            matcha_scene=other_scene,
            matcha_model=model,
            output=tmp_path / "adapter",
            iteration=30000,
        )
