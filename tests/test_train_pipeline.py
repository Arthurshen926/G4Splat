from pathlib import Path


def test_chart_plane_frontend_uses_requested_resolution_and_device():
    source = Path("train.py").read_text()

    render_command = source.split('render_charts_command = " ".join([', 1)[1]
    render_command = render_command.split("])\n", 1)[0]
    assert '"--resolution", str(args.resolution)' in render_command
    assert '"--data_device", args.data_device' in render_command


def test_camera_config_defaults_do_not_overwrite_explicit_resolution():
    source = Path("2d-gaussian-splatting/scene/dataset_readers.py").read_text()

    fill_config = source.split("def fill_config_args(args):", 1)[1]
    fill_config = fill_config.split("from utils.camera_utils", 1)[0]
    assert "if not hasattr(args, name) or getattr(args, name) is None" in fill_config
    assert "args.resolution = -1" not in fill_config


def test_see3d_iteration_is_configurable_and_not_hardcoded():
    source = Path("train.py").read_text()
    command = source.split("def get_see3d_inpaint_command", 1)[1]
    command = command.split("eval_command", 1)[0]
    assert '"--iteration", str(args.see3d_iteration)' in command
    assert '"--iteration", \'7000\'' not in command


def test_warmstart_refinement_exposes_ply_and_freeze_flags():
    source = Path("scripts/refine_free_gaussians.py").read_text()
    assert "--init_ply" in source
    assert "--freeze_init_ply" in source
    assert "warmstart_reseed_pixel_stride" in source


def test_no_see3d_final_pass_warmstarts_from_initial_7k_ply():
    source = Path("train.py").read_text()
    branch = source.split("if args.disable_see3d:", 1)[1]
    branch = branch.split("# see3d inpainting stage 1", 1)[0]
    assert "iteration_7000" in branch
    assert "init_ply=initial_ply" in branch
    assert "point_cloud-initial-7k" not in branch


def test_render_output_dir_is_backward_compatible_with_old_cfg_args():
    source = Path("2d-gaussian-splatting/render.py").read_text()
    assert 'getattr(args, "output_dir", None)' in source
