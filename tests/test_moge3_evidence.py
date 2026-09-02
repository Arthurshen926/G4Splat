import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from outdoor.moge3_evidence import (
    MOGE3_AUDITED_COMMIT,
    MOGE3_INDEX_VERSION,
    MOGE3_RUNTIME_CACHE_VERSION,
    atomic_save_view,
    build_index,
    exact_k_unproject_depth,
    exact_pixel_intrinsics,
    horizontal_fov_degrees,
    load_index,
    load_runtime_cache,
    load_view,
    make_view_payload,
)
from outdoor.moge3_chart_base import (
    MOGE3_CHART_BASE_VERSION,
    chart_base_variants,
    load_chart_base,
    load_chart_base_metadata,
    robust_scene_scale as robust_chart_scene_scale,
)
from outdoor.role_aware_initialization import (
    _moge3_foliage_samples,
    _robust_moge3_scene_scale,
    _resolve_moge3_foliage_scene_scale,
)


def _camera_record(*, image_sha256: str = "a" * 64) -> dict:
    return {
        "frame_index": 0,
        "image_name": "seq0__frame000.png",
        "source_image_name": "seq0__frame000.png",
        "camera": {
            "camera_id": 7,
            "model": "PINHOLE",
            "width": 5,
            "height": 5,
            "params": [4.0, 5.0, 1.25, 2.25],
            "fx": 4.0,
            "fy": 5.0,
            "cx": 1.25,
            "cy": 2.25,
            "distortion": [],
        },
        "qvec_input_norm": 1.0,
        "T_world_to_camera": np.eye(4).tolist(),
        "T_camera_to_world": np.eye(4).tolist(),
        "camera_center_world": [0.0, 0.0, 0.0],
        "sequence_id": "seq0",
        "split": "database_train",
        "image_sha256": image_sha256,
    }


def _payload(record: dict) -> dict[str, np.ndarray]:
    depth = np.full((5, 5), 2.0, dtype=np.float32)
    direct = np.zeros((5, 5, 3), dtype=np.float32)
    direct[..., 2] = 1.0
    steps = np.stack((depth * 1.02, depth * 1.01, depth, depth), axis=0)
    return make_view_payload(
        depth_m=depth,
        normal_direct_camera=direct,
        model_valid_mask=np.ones((5, 5), dtype=bool),
        depth_per_step_m=steps,
        K_px=exact_pixel_intrinsics(record["camera"]),
        model_intrinsics_normalized=np.asarray(
            [[0.8, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        camera_record=record,
        image_sha256=record["image_sha256"],
        model_id="Ruicheng/moge-3-vitl",
        model_revision="immutable-model-revision",
        model_checkpoint_sha256="a" * 64,
        refine_steps=3,
        resolution_level=9,
        use_fp16=True,
    )


def _generator_contract() -> dict:
    return {
        "checkpoint_sha256": "a" * 64,
        "refine_steps": 3,
        "resolution_level": 9,
        "use_fp16": True,
        "moge_git_revision": MOGE3_AUDITED_COMMIT,
    }


def test_exact_k_unprojection_preserves_off_center_cambridge_principal_point():
    record = _camera_record()
    K = exact_pixel_intrinsics(record["camera"])
    depth = np.full((5, 5), 2.0, dtype=np.float32)
    points = exact_k_unproject_depth(depth, K)
    assert points[2, 1].tolist() == pytest.approx([-0.125, -0.1, 2.0])
    # A hard-coded normalized centre would put the same pixel elsewhere.
    assert points[2, 1, 0] != pytest.approx((1.0 - 2.5) / 4.0 * 2.0)
    assert horizontal_fov_degrees(width=5, fx=4.0) == pytest.approx(
        math.degrees(2.0 * math.atan(5.0 / 8.0))
    )


def test_exact_k_contract_rejects_distorted_or_unsupported_cameras():
    record = _camera_record()
    camera = dict(record["camera"])
    camera["model"] = "OPENCV"
    with pytest.raises(ValueError, match="already-undistorted"):
        exact_pixel_intrinsics(camera)
    camera["model"] = "PINHOLE"
    camera["distortion"] = [0.01]
    with pytest.raises(ValueError, match="non-zero lens distortion"):
        exact_pixel_intrinsics(camera)


def test_view_payload_keeps_sources_separate_and_is_exact_k(tmp_path):
    record = _camera_record()
    path = tmp_path / "view.npz"
    atomic_save_view(path, _payload(record))
    view = load_view(path)
    assert view["metadata"]["pointmap_policy"] == (
        "recomputed_from_depth_with_exact_pixel_K"
    )
    assert not view["metadata"]["model_pointmap_is_geometry_authority"]
    assert view["metadata"]["model_checkpoint_sha256"] == "a" * 64
    assert view["metadata"]["use_fp16"] is True
    assert view["depth_per_step_m"].shape == (4, 5, 5)
    assert view["normal_direct_camera"].shape == (5, 5, 3)
    assert view["normal_depth_exact_k_camera"].shape == (5, 5, 3)
    assert view["refinement_log_depth_std"][2, 2] > 0
    assert view["refinement_final_delta_log_depth"][2, 2] == pytest.approx(0)
    assert view["depth_normal_valid_mask"].sum() == 9


def test_view_loader_rejects_pointmap_not_derived_from_exact_k(tmp_path):
    record = _camera_record()
    payload = _payload(record)
    payload["pointmap_exact_k_camera"] = payload[
        "pointmap_exact_k_camera"
    ].copy()
    payload["pointmap_exact_k_camera"][2, 2, 0] += 0.25
    path = tmp_path / "bad.npz"
    atomic_save_view(path, payload)
    with pytest.raises(RuntimeError, match="not an exact-K unprojection"):
        load_view(path)


def test_index_is_scene_complete_content_addressed_and_detects_mutation(tmp_path):
    record = _camera_record()
    root = tmp_path / "views"
    root.mkdir()
    view_path = root / f"{Path(record['image_name']).stem}.npz"
    atomic_save_view(view_path, _payload(record))
    index_path = tmp_path / "moge3_index.json"
    payload = build_index(
        view_root=root,
        scene_contract={"records": [record]},
        output=index_path,
        model_id="Ruicheng/moge-3-vitl",
        model_revision="immutable-model-revision",
        generator=_generator_contract(),
    )
    assert payload["schema_version"] == MOGE3_INDEX_VERSION
    assert payload["indexed_view_count"] == payload["scene_view_count"] == 1
    assert load_index(index_path)["index_hash"] == payload["index_hash"]

    original_bytes = view_path.read_bytes()
    mutated = bytearray(original_bytes)
    mutated[-1] ^= 1
    view_path.write_bytes(mutated)
    with pytest.raises(RuntimeError, match="view content changed"):
        load_index(index_path, verify_payloads=False)
    view_path.write_bytes(original_bytes)

    raw = json.loads(index_path.read_text(encoding="utf-8"))
    raw["generator"]["checkpoint_sha256"] = "c" * 64
    index_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RuntimeError, match="index hash"):
        load_index(index_path, verify_views=False)


def test_compact_runtime_cache_is_camera_ordered_and_memory_mapped(tmp_path):
    record = _camera_record()
    root = tmp_path / "views"
    root.mkdir()
    atomic_save_view(
        root / f"{Path(record['image_name']).stem}.npz",
        _payload(record),
    )
    index_path = tmp_path / "moge3_index.json"
    build_index(
        view_root=root,
        scene_contract={"records": [record]},
        output=index_path,
        model_id="Ruicheng/moge-3-vitl",
        model_revision="immutable-model-revision",
        generator=_generator_contract(),
    )
    output = tmp_path / "runtime"
    completed = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "scripts/build_moge3_runtime_cache.py"
            ),
            "--moge3-index",
            str(index_path),
            "--output",
            str(output),
            "--width",
            "4",
            "--height",
            "3",
            "--workers",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    runtime = load_runtime_cache(output / "runtime_index.json")
    assert runtime["raster_shape"] == (3, 4)
    assert runtime["camera_order"] == [Path(record["image_name"]).stem]
    assert runtime["arrays"]["depth_m"].shape == (1, 3, 4)
    assert isinstance(runtime["arrays"]["depth_m"], np.memmap)
    depth = np.asarray(runtime["arrays"]["depth_m"][0])
    valid = np.asarray(runtime["arrays"]["valid_mask"][0]).astype(bool)
    assert np.isfinite(depth).all()
    assert np.all(depth[valid] > 0.0)
    manifest = json.loads(
        (output / "runtime_index.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == MOGE3_RUNTIME_CACHE_VERSION


def test_moge3_foliage_uses_one_global_scale_and_rejects_behind_rigid(
    tmp_path,
):
    dataset = tmp_path / "dataset"
    image_root = dataset / "images"
    image_root.mkdir(parents=True)
    view_root = tmp_path / "moge3" / "views"
    view_root.mkdir(parents=True)
    scene_records = []
    images = {}
    cameras = {
        7: {
            "id": 7,
            "model": "PINHOLE",
            "width": 5,
            "height": 5,
            "params": np.asarray([4.0, 5.0, 1.25, 2.25]),
        }
    }
    for frame_index, image_id in enumerate((11, 12)):
        record = _camera_record(image_sha256=(str(frame_index + 1) * 64))
        record["frame_index"] = frame_index
        record["image_name"] = f"seq{frame_index}__frame.png"
        record["source_image_name"] = record["image_name"]
        record["sequence_id"] = f"seq{frame_index}"
        scene_records.append(record)
        rgb = np.full((5, 5, 3), 80 + 20 * frame_index, dtype=np.uint8)
        Image.fromarray(rgb).save(image_root / record["image_name"])
        atomic_save_view(
            view_root / f"{Path(record['image_name']).stem}.npz",
            _payload(record),
        )
        images[image_id] = {
            "id": image_id,
            "name": record["image_name"],
            "camera_id": 7,
            "qvec": np.asarray([1.0, 0.0, 0.0, 0.0]),
            "tvec": np.zeros(3),
        }
    index_path = tmp_path / "moge3" / "moge3_index.json"
    build_index(
        view_root=view_root,
        scene_contract={"records": scene_records},
        output=index_path,
        model_id="Ruicheng/moge-3-vitl",
        model_revision="immutable-model-revision",
        generator=_generator_contract(),
    )

    # Right two columns are rigid scale anchors: Cambridge depth 4 divided
    # by MoGe depth 2 gives one global scale of 2. Left columns are canopy.
    channel0 = torch.ones((5, 5), dtype=torch.bool)
    channel1 = torch.ones((5, 5), dtype=torch.bool)
    channel2 = torch.ones((5, 5), dtype=torch.bool)
    rigid_channel = torch.zeros((5, 5), dtype=torch.bool)
    rigid_channel[:, 3:] = True

    class Masks:
        def __init__(self):
            self.masks = {
                record["image_name"]: (
                    channel0,
                    channel1,
                    channel2,
                    rigid_channel,
                )
                for record in scene_records
            }

        @staticmethod
        def source_name_for(name):
            return name

    rigid_depth_maps = {}
    for image_id in images:
        zbuffer = np.full((5, 5), np.inf, dtype=np.float32)
        zbuffer[:, 3:] = 4.0
        zbuffer[1, 0] = 6.0  # MoGe scaled depth 4 is a valid front hit.
        zbuffer[2, 0] = 3.0  # MoGe scaled depth 4 is behind rigid: reject.
        rigid_depth_maps[image_id] = zbuffer
    store = {
        "dataset": str(dataset),
        "artifacts": [{"name": "moge3_index", "path": str(index_path)}],
    }
    with pytest.raises(RuntimeError, match="requires dense rigid depth maps"):
        _moge3_foliage_samples(
            store,
            images,
            cameras,
            Masks(),
            np.asarray([[0.0, 0.0, 4.0]], dtype=np.float32),
            rigid_depth_maps=None,
            minimum_global_scale_pixels=2,
        )
    samples, audit = _moge3_foliage_samples(
        store,
        images,
        cameras,
        Masks(),
        np.asarray([[0.0, 0.0, 4.0]], dtype=np.float32),
        rigid_depth_maps=rigid_depth_maps,
        samples_per_view=25,
        maximum_samples=100,
        minimum_global_scale_pixels=2,
    )
    assert audit["metric_to_cambridge_scale"] == pytest.approx(2.0)
    assert audit["rigid_behind_rejected_pixels"] == 2
    assert audit["behind_policy"] == "unknown_never_positive_mass"
    assert samples
    assert all(sample["moge3_sequence_local_sample"] for sample in samples)
    rejected_uv = np.asarray([(0.5 / 5.0), (2.5 / 5.0)])
    assert not any(
        np.allclose(sample["observation_uv"][0], rejected_uv)
        for sample in samples
    )


def test_moge3_scene_scale_selects_cross_view_mode_not_mismatch_variance():
    rng = np.random.default_rng(17)
    true_log_scale = math.log(1.375)
    consensus = np.concatenate(
        [
            rng.normal(true_log_scale, 0.025, size=300),
            rng.normal(true_log_scale + 0.01, 0.025, size=300),
        ]
    )
    # More than half of the nominal rigid pixels are genuine surface
    # mismatches spread across a broad range. They must not inflate the gauge
    # uncertainty or move the selected mode.
    mismatches = rng.uniform(-0.8, 1.5, size=800)
    fitted = _robust_moge3_scene_scale(
        np.concatenate((consensus, mismatches)),
        np.concatenate(
            (
                np.repeat([11, 12], 300),
                np.repeat([11, 12], 400),
            )
        ),
        minimum_pixels=256,
    )
    assert fitted["accepted"]
    assert fitted["metric_to_cambridge_scale"] == pytest.approx(
        1.375, rel=0.04
    )
    assert fitted["global_log_scale_sigma"] < 0.08
    assert fitted["scale_support_views"] == 2
    assert fitted["scale_consensus_fraction"] < 0.65


def test_moge3_chart_base_is_continuous_and_preserves_invalid_fallback(tmp_path):
    matcha = np.asarray([[2.0, 2.0], [0.0, 4.0]], dtype=np.float32)
    moge = np.asarray([[2.0, 4.0], [3.0, 8.0]], dtype=np.float32)
    normal = np.zeros((2, 2, 3), dtype=np.float32)
    normal[..., 2] = 1.0
    result = chart_base_variants(
        matcha,
        moge,
        rigid_valid=np.asarray([[1, 1], [1, 0]], dtype=bool),
        refinement_sigma=np.zeros((2, 2), dtype=np.float32),
        normal_direct_camera=normal,
        normal_depth_camera=normal,
        depth_normal_valid=np.ones((2, 2), dtype=bool),
    )
    assert result["depth_moge3"][0, 1] == pytest.approx(4.0)
    assert 2.0 < result["depth_moge3_adaptive"][0, 1] < 4.0
    assert result["depth_moge3_adaptive"][1, 0] == pytest.approx(3.0)
    assert result["depth_moge3"][1, 1] == pytest.approx(4.0)
    assert result["moge3_adaptive_weight"][1, 1] == 0.0

    path = tmp_path / "chart_base.npz"
    np.savez_compressed(
        path,
        schema_version=np.asarray(MOGE3_CHART_BASE_VERSION),
        image_names=np.asarray(["seq0__frame000.png"]),
        depth_moge3=result["depth_moge3"][None],
        depth_moge3_adaptive=result["depth_moge3_adaptive"][None],
        validity=result["validity"][None],
        moge3_precision=result["moge3_precision"][None],
        moge3_adaptive_weight=result["moge3_adaptive_weight"][None],
        metadata_json=np.asarray(json.dumps({"scale": 1.0})),
    )
    loaded = load_chart_base(
        path,
        source="moge3_adaptive",
        expected_names=["seq0__frame000.png"],
    )
    assert np.array_equal(loaded["depth"][0], result["depth_moge3_adaptive"])
    metadata = load_chart_base_metadata(
        path, expected_names=["seq0__frame000.png"]
    )
    assert metadata["metadata"] == {"scale": 1.0}


def test_chart_scale_authority_survives_broad_rigid_residual_mode():
    rigid = {
        "accepted": False,
        "reason": "global_scale_consensus_too_broad",
        "candidate_metric_to_cambridge_scale": 0.84,
        "global_log_scale_sigma": 0.15,
        "scale_support_pixels": 2048,
        "scale_support_views": 6,
    }
    resolved = _resolve_moge3_foliage_scene_scale(
        rigid,
        {
            "metric_to_cambridge_scale": 0.8277335148,
            "global_log_scale_sigma": 0.1009902423,
        },
    )
    assert resolved["accepted"]
    assert resolved["metric_to_cambridge_scale"] == pytest.approx(
        0.8277335148
    )
    assert resolved["scale_authority"] == (
        "moge3_matcha_chart_scene_consensus"
    )
    assert resolved["global_log_scale_sigma"] < 0.12
    assert not resolved["rigid_depth_fit_accepted"]


def test_chart_scale_rejects_supported_inconsistent_rigid_mode():
    with pytest.raises(RuntimeError, match="scene scales disagree"):
        _resolve_moge3_foliage_scene_scale(
            {
                "accepted": False,
                "reason": "global_scale_consensus_too_broad",
                "candidate_metric_to_cambridge_scale": 1.37,
                "scale_support_pixels": 2048,
                "scale_support_views": 6,
            },
            {
                "metric_to_cambridge_scale": 0.8277335148,
                "global_log_scale_sigma": 0.1009902423,
            },
        )


def test_moge3_chart_scale_requires_cross_view_consensus():
    rng = np.random.default_rng(23)
    fitted = robust_chart_scene_scale(
        [
            rng.normal(math.log(1.4), 0.02, size=300),
            rng.normal(math.log(1.4), 0.02, size=300),
            rng.uniform(-1.0, 1.0, size=500),
        ],
        minimum_pixels=256,
    )
    assert fitted["metric_to_cambridge_scale"] == pytest.approx(1.4, rel=0.04)
    assert fitted["scale_support_views"] >= 2
