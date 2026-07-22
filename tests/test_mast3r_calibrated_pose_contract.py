import numpy as np

from matcha.pointmap.pose_contract import calibrated_pose_contract


def _cameras():
    cameras = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    cameras[1, 0, 3] = 2.0
    return cameras


def test_calibrated_pose_contract_accepts_identical_effective_cameras():
    cameras = _cameras()
    report = calibrated_pose_contract(cameras, cameras, ["a.png", "b.png"])

    assert report["passed"]
    assert report["relative_center_error"]["max"] == 0.0
    assert report["rotation_error_deg"]["max"] == 0.0


def test_calibrated_pose_contract_rejects_hidden_center_shift():
    calibrated = _cameras()
    estimated = calibrated.copy()
    estimated[1, 2, 3] += 0.02

    report = calibrated_pose_contract(
        estimated,
        calibrated,
        ["a.png", "b.png"],
        max_relative_center_error=1e-3,
    )

    assert not report["passed"]
    assert report["failing_image_names"] == ["b.png"]
