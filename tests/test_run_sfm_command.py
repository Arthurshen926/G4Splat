from argparse import Namespace

from scripts.run_sfm import build_command


def test_run_sfm_passes_explicit_indices_as_separate_arguments():
    args = Namespace(
        source_path="dataset",
        output_path="output",
        n_images=3,
        image_idx=[2, 5, 9],
        use_all_images=False,
        randomize_images=False,
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
    assert "" not in command
