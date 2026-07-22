from argparse import Namespace
from pathlib import Path

from scripts.run_sfm import build_command


def test_run_sfm_passes_explicit_indices_as_separate_arguments():
    args = Namespace(
        source_path="dataset",
        output_path="output",
        n_images=3,
        image_idx=[2, 5, 9],
        use_all_images=False,
        randomize_images=False,
        sparse_export_stride=1,
    )
    config = {
        "weights_path": "weights.pth",
        "retrieval_model": "retrieval.pth",
        "min_conf_thr": 0.0,
        "matching_conf_thr": 0.0,
        "n_coarse_iterations": 1,
        "n_refinement_iterations": 1,
        "TSDF_thresh": 0.0,
        "fix_focal": True,
        "fix_principal_point": True,
        "fix_rotation": True,
        "fix_translation": True,
        "image_size": 512,
        "max_window_size": 20,
        "max_refid": 10,
        "output_conf_thr": 0.1,
        "use_calibrated_poses": True,
        "save_glb": False,
        "align_camera_locations": True,
    }

    command = build_command(args, config)
    start = command.index("--image_idx")

    assert command[start + 1 : start + 4] == ["2", "5", "9"]
    assert command[command.index("--sparse_export_stride") + 1] == "1"
    assert "" not in command


def test_run_sfm_exposes_strict_calibrated_pose_ablation_only_when_requested():
    args = Namespace(
        source_path="dataset",
        output_path="output",
        n_images=1,
        image_idx=None,
        use_all_images=False,
        randomize_images=False,
        strict_calibrated_poses=True,
        sparse_export_stride=1,
    )
    config = {
        "weights_path": "weights.pth",
        "retrieval_model": "retrieval.pth",
        "min_conf_thr": 0.0,
        "matching_conf_thr": 0.0,
        "n_coarse_iterations": 1,
        "n_refinement_iterations": 1,
        "TSDF_thresh": 0.0,
        "fix_focal": True,
        "fix_principal_point": True,
        "fix_rotation": True,
        "fix_translation": True,
        "image_size": 512,
        "max_window_size": 20,
        "max_refid": 10,
        "output_conf_thr": 0.1,
        "use_calibrated_poses": True,
        "save_glb": False,
        "align_camera_locations": True,
    }

    command = build_command(args, config)

    assert command[-1] == "--strict_calibrated_poses"


def test_run_sfm_exposes_per_view_calibrated_intrinsics_as_a_separate_variable():
    args = Namespace(
        source_path="dataset",
        output_path="output",
        n_images=1,
        image_idx=None,
        use_all_images=False,
        randomize_images=False,
        strict_calibrated_poses=True,
        per_view_calibrated_intrinsics=True,
        sparse_export_stride=1,
    )
    config = {
        "weights_path": "weights.pth",
        "retrieval_model": "retrieval.pth",
        "min_conf_thr": 0.0,
        "matching_conf_thr": 0.0,
        "n_coarse_iterations": 1,
        "n_refinement_iterations": 1,
        "TSDF_thresh": 0.0,
        "fix_focal": True,
        "fix_principal_point": True,
        "fix_rotation": True,
        "fix_translation": True,
        "image_size": 512,
        "max_window_size": 20,
        "max_refid": 10,
        "output_conf_thr": 0.1,
        "use_calibrated_poses": True,
        "save_glb": False,
        "align_camera_locations": True,
    }

    command = build_command(args, config)

    assert command[-2:] == [
        "--strict_calibrated_poses",
        "--per_view_calibrated_intrinsics",
    ]


def test_strict_calibrated_pose_mode_really_freezes_ma_st3r_translation_and_scale():
    source = Path("mast3r/run_mast3r.py").read_text()

    assert "'opt_tran': not strict_calibrated_poses" in source
    assert "'opt_size': not strict_calibrated_poses" in source
    assert "--strict_calibrated_poses requires --use_calibrated_poses" in source
    assert "shared_intrinsics = not per_view_calibrated_intrinsics" in source
    assert "--per_view_calibrated_intrinsics requires --use_calibrated_poses" in source
