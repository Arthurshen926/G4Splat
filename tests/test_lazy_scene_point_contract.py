import json
from pathlib import Path
import sys
from types import SimpleNamespace

SURFEL_ROOT = Path(__file__).resolve().parents[1] / "2d-gaussian-splatting"
sys.path.insert(0, str(SURFEL_ROOT))

from outdoor.lazy_scene import rgb_source_contract
from scene import dataset_readers


def test_camera_only_scene_does_not_open_colmap_point_geometry(
    tmp_path, monkeypatch
):
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "images.bin").write_bytes(b"camera container")
    (sparse / "cameras.bin").write_bytes(b"camera container")
    dense = tmp_path / "dense-view-sparse" / "0"
    dense.mkdir(parents=True)
    # A stray point artifact must not select a different camera container
    # when this API is explicitly camera-only.
    (dense / "points3D.ply").write_bytes(b"must not be inspected")
    cameras = [SimpleNamespace(image_name="frame")]
    opened_camera_paths = []

    def read_camera_container(path):
        opened_camera_paths.append(Path(path))
        return {}

    monkeypatch.setattr(
        dataset_readers, "read_extrinsics_binary", read_camera_container
    )
    monkeypatch.setattr(
        dataset_readers, "read_intrinsics_binary", read_camera_container
    )
    monkeypatch.setattr(
        dataset_readers,
        "readColmapCameras",
        lambda **_: cameras,
    )
    monkeypatch.setattr(
        dataset_readers,
        "getNerfppNorm",
        lambda _: {"translate": [0, 0, 0], "radius": 1.0},
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("COLMAP point geometry was opened")

    monkeypatch.setattr(dataset_readers, "fetchPly", forbidden)
    monkeypatch.setattr(dataset_readers, "read_points3D_binary", forbidden)
    result = dataset_readers.readColmapSceneInfo(
        str(tmp_path),
        "images",
        False,
        load_images=False,
        load_point_cloud=False,
    )

    assert result.point_cloud is None
    assert result.ply_path is None
    assert result.train_cameras == cameras
    assert opened_camera_paths
    assert all("dense-view-sparse" not in str(path) for path in opened_camera_paths)


def test_rgb_target_producer_is_part_of_immutable_contract(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "b.png").write_bytes(b"b")
    (images / "a.jpg").write_bytes(b"a")
    (tmp_path / "input_manifest.json").write_text(
        json.dumps(
            {
                "contract": "shared-target-v1",
                "rgb_target_storage": "torch-bilinear",
                "canonical_image_size_wh": [640, 360],
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(source_path=tmp_path, images="images")

    contract = rgb_source_contract(args)

    assert contract["image_root"] == str(images.resolve())
    assert contract["image_count"] == 2
    assert contract["content_bytes"] == 2
    original_digest = contract["content_mapping_sha256"]
    assert contract["producer_contract"] == "shared-target-v1"
    assert contract["target_storage"] == "torch-bilinear"
    assert contract["producer_manifest_sha256"] is not None

    # The filename inventory is unchanged, but the immutable RGB target is
    # not.  A name-set-only contract would silently accept this corruption.
    (images / "a.jpg").write_bytes(b"changed")
    changed = rgb_source_contract(args)
    assert changed["name_set_sha256"] == contract["name_set_sha256"]
    assert changed["content_mapping_sha256"] != original_digest
