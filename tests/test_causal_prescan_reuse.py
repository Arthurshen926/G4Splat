import json
import struct

import numpy as np
import pytest
from PIL import Image

from scripts.causal_2dgs_repair import (
    _LazyCameraStore,
    _accepted_causal_target_names,
    _real_camera_envelope_control_order,
    _reuse_full_train_anomaly_prescan,
    _sha256_file,
)


def test_reused_prescan_requires_exact_frozen_metrics_and_camera_coverage(tmp_path):
    model = tmp_path / "frozen_model"
    metrics = model / "evaluation" / "rgb_metrics.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("{}\n", encoding="utf-8")
    report = tmp_path / "prior_prescan.json"
    payload = {
        "status": "completed",
        "expected_full_train_view_count": 2,
        "processed_view_count": 2,
        "metrics_json_sha256": _sha256_file(metrics),
        "records": [
            {"name": "seq1__frame00001", "anomaly_ray_pixel_count": 4, "priority": 0.4},
            {"name": "seq1__frame00002", "anomaly_ray_pixel_count": 8, "priority": 0.5},
        ],
    }
    report.write_text(json.dumps(payload), encoding="utf-8")
    cameras = {"seq1__frame00001": object(), "seq1__frame00002": object()}

    reused = _reuse_full_train_anomaly_prescan(
        report_path=report,
        metrics_path=metrics,
        model_path=model,
        cameras=cameras,
    )

    assert reused["reused_from"] == str(report)
    payload["records"][1]["name"] = "seq1__different"
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly cover"):
        _reuse_full_train_anomaly_prescan(
            report_path=report,
            metrics_path=metrics,
            model_path=model,
            cameras=cameras,
        )


def test_lazy_camera_store_reads_all_pose_metadata_without_decoding_all_images(tmp_path):
    sparse = tmp_path / "sparse" / "0"
    images = tmp_path / "images"
    sparse.mkdir(parents=True)
    images.mkdir()
    Image.fromarray(np.full((60, 80, 3), 127, dtype=np.uint8), mode="RGB").save(images / "frame.png")
    with (sparse / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<iiQQ", 1, 1, 80, 60))  # PINHOLE
        handle.write(struct.pack("<dddd", 60.0, 50.0, 40.0, 30.0))
    with (sparse / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<idddddddi", 1, 1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 1))
        handle.write(b"frame.png\x00")
        handle.write(struct.pack("<Q", 0))

    store = _LazyCameraStore.from_colmap(tmp_path, requested_resolution=2, data_device="cpu")
    geometry = store.geometries["frame"]

    assert len(store) == 1
    assert list(store) == ["frame"]
    assert (geometry.width, geometry.height) == (40, 30)
    assert np.isclose(geometry.fx, 30.0)
    assert np.isclose(geometry.fy, 25.0)
    assert np.allclose(geometry.w2c[:3, 3], [0.1, 0.0, 0.0])


def test_real_camera_envelope_keeps_low_quality_temporal_counterevidence():
    """A bad frozen render cannot exclude an adjacent real camera from a gate."""
    names = [f"seq4__frame{frame:05d}" for frame in range(130, 161)]
    features = {
        name: np.asarray([float(index) / 100.0, 0.0, 0.0], dtype=np.float64)
        for index, name in enumerate(names)
    }
    # Reproduce the old failure mode: quality would rank frames 138--141 far
    # behind unrelated clean ones.  The new envelope can retain the value as a
    # tie-breaker but must still include them by temporal adjacency.
    quality = {name: (25.0 if name.endswith(("00138", "00139", "00140", "00141")) else 30.0) for name in names}
    targets = ["seq4__frame00148", "seq4__frame00149", "seq4__frame00150"]

    ordered = _real_camera_envelope_control_order(
        targets,
        features,
        quality,
        control_count=32,
        temporal_radius=16,
    )
    local_names = [
        row["name"]
        for row in ordered
        if row["selection_stage"] == "same_sequence_temporal_envelope"
    ]

    assert {"seq4__frame00138", "seq4__frame00139", "seq4__frame00140", "seq4__frame00141"}.issubset(local_names)
    assert all(row["selection_stage"] != "legacy_metric_nearby" for row in ordered)
    assert all(row["temporal_offset_frames"] is not None for row in ordered[:24])


def test_post_edit_gate_scores_only_accepted_causal_targets():
    contexts = {name: object() for name in ("target_a", "target_b", "unrelated_triage")}
    counterfactuals = [
        {
            "accepted": True,
            "targets": [{"name": "target_b"}, {"name": "target_a"}],
        },
        {
            "accepted": False,
            "targets": [{"name": "unrelated_triage"}],
        },
    ]

    selected = _accepted_causal_target_names(counterfactuals, contexts)

    assert selected == ["target_a", "target_b"]
