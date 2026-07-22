import json

import pytest

from scripts.audit_camera_contract import build_camera_contract_audit


def _camera(name, *, fx=10.0, offset=0.0):
    return {
        "id": 3,
        "img_name": name,
        "width": 640,
        "height": 360,
        "fx": fx,
        "fy": 11.0,
        "position": [offset, 2.0, 3.0],
        "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }


def test_camera_contract_normalizes_extension_and_path_before_comparing(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps([_camera("seq1__frame00001")]))
    candidate.write_text(json.dumps([_camera("seq1/frame00001.png")]))

    report = build_camera_contract_audit(baseline, candidate)

    assert report["passed"]
    assert report["shared_camera_count"] == 1
    assert report["field_max_abs_difference"]["rotation"] == 0.0


def test_camera_contract_reports_an_image_aligned_geometric_difference(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps([_camera("frame00001.png")]))
    candidate.write_text(json.dumps([_camera("frame00001.png", fx=10.01)]))

    report = build_camera_contract_audit(baseline, candidate, tolerance=1e-6)

    assert not report["passed"]
    assert report["mismatch_count"] == 1
    mismatch = report["mismatches"][0]
    assert mismatch["image"] == "frame00001"
    assert mismatch["field"] == "fx"
    assert mismatch["max_abs_difference"] == pytest.approx(0.01)
