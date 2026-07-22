from pathlib import Path

import yaml


def test_chart_plane_frontend_uses_requested_resolution_and_device():
    source = Path("train.py").read_text()

    render_command = source.split('render_charts_command = " ".join([', 1)[1]
    render_command = render_command.split("])\n", 1)[0]
    assert '"--resolution", str(args.resolution)' in render_command
    assert '"--data_device", args.data_device' in render_command
    assert '"--max_chart_abs_depth", str(args.max_chart_abs_depth)' in render_command
    assert '"--max_chart_abs_point", str(args.max_chart_abs_point)' in render_command


def test_train_entrypoint_accepts_mainline_chart_cap_flag_spelling():
    source = Path("train.py").read_text()

    assert "'--max-chart-abs-depth'" in source
    assert "'--max-chart-abs-point'" in source


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


def test_cfg_args_loader_accepts_legacy_pathlib_serialization():
    source = Path("2d-gaussian-splatting/arguments/__init__.py").read_text()
    assert '"PosixPath": PosixPath' in source
    assert '"WindowsPath": WindowsPath' in source


def test_standard_control_serializes_path_arguments_as_plain_strings():
    source = Path("scripts/train_standard_full_2dgs.py").read_text()
    assert "cfg_values = {" in source
    assert "str(value) if isinstance(value, Path) else value" in source


def test_standard_control_audits_mask_resolution_before_claiming_ulfloc_parity():
    source = Path("scripts/train_standard_full_2dgs.py").read_text()
    assert "def _mask_resolution_audit(" in source
    assert '"native_pixel_geometry"' in source
    assert "--require-native-ulfloc-mask-resolution" in source
    assert "nearest_resampled_mask_pixels" in source


def test_refinement_records_its_effective_single_variable_ablation_inputs():
    source = Path("scripts/refine_free_gaussians.py").read_text()
    assert '"refinement_manifest.json"' in source
    assert '"effective_iterations": int(args.iterations or config["iterations"])' in source
    assert "write_refinement_manifest(args, config, command)" in source


def test_affine_color_correction_checkpoint_is_saved_and_used_by_rgb_export():
    trainer = Path("2d-gaussian-splatting/train_with_refine_depth.py").read_text()
    renderer = Path("2d-gaussian-splatting/render.py").read_text()
    assert '"color_correction.pth"' in trainer
    assert "save_per_image_affine_color_correction(" in trainer
    assert "load_checkpoint_color_correction(" in renderer
    assert "apply_per_image_affine_color_correction(" in renderer
    assert "faithful train-objective evaluation" in renderer


def test_capacity_ablation_changes_only_the_densification_horizon():
    default = yaml.safe_load(
        Path("configs/free_gaussians_refinement/default.yaml").read_text()
    )
    capacity = yaml.safe_load(
        Path("configs/free_gaussians_refinement/default_densify15k.yaml").read_text()
    )
    assert {
        key: value for key, value in capacity.items() if key != "densify_until_iter"
    } == {
        key: value for key, value in default.items() if key != "densify_until_iter"
    }
    assert default["densify_until_iter"] == 3500
    assert capacity["densify_until_iter"] == 15000


def test_standard_schedule_controls_each_change_one_default_field():
    default = yaml.safe_load(
        Path("configs/free_gaussians_refinement/default.yaml").read_text()
    )
    controls = {
        "default_opacity3k.yaml": ("opacity_reset_interval", 3000),
        "default_normal7k.yaml": ("normal_consistency_from", 7000),
        "default_no_mip.yaml": ("use_mip_filter", False),
    }
    for filename, (changed_key, changed_value) in controls.items():
        candidate = yaml.safe_load(
            Path("configs/free_gaussians_refinement", filename).read_text()
        )
        assert candidate[changed_key] == changed_value
        assert {
            key: value for key, value in candidate.items() if key != changed_key
        } == {
            key: value for key, value in default.items() if key != changed_key
        }


def test_train_pipeline_forwards_explicit_strict_calibrated_pose_mode():
    source = Path("train.py").read_text()
    sfm_command = source.split('sfm_command = " ".join([', 1)[1]
    sfm_command = sfm_command.split("])\n", 1)[0]
    assert '"--strict_calibrated_poses" if args.strict_calibrated_poses else ""' in sfm_command


def test_train_pipeline_forwards_bounded_sparse_export_without_downsampling_pointmaps():
    source = Path("train.py").read_text()
    sfm_command = source.split('sfm_command = " ".join([', 1)[1]
    sfm_command = sfm_command.split("])\n", 1)[0]
    assert '"--sparse-export-stride", str(args.mast3r_sparse_export_stride)' in sfm_command
    assert "full pointmaps used by Chart alignment are never downsampled" in source


def test_train_pipeline_forwards_a_fixed_chart_alignment_seed():
    source = Path("train.py").read_text()
    align_command = source.split('align_charts_command = " ".join([', 1)[1]
    align_command = align_command.split("])\n", 1)[0]
    assert '"--seed", str(args.chart_alignment_seed)' in align_command


def test_train_pipeline_forwards_the_explicit_rgb_supervision_protocol():
    source = Path("train.py").read_text()
    refine_command = source.split("def get_refine_free_gaussians_command", 1)[1]
    refine_command = refine_command.split("render_all_img_command", 1)[0]
    assert '"--rgb-supervision-profile", args.rgb_supervision_profile' in refine_command


def test_train_pipeline_forwards_the_explicit_rgb_sampling_policy():
    source = Path("train.py").read_text()
    refine_command = source.split("def get_refine_free_gaussians_command", 1)[1]
    refine_command = refine_command.split("render_all_img_command", 1)[0]
    assert '"--rgb-sampling-policy", args.rgb_sampling_policy' in refine_command


def test_train_pipeline_forwards_active_chart_geometry_sampling_policy():
    source = Path("train.py").read_text()
    refine_command = source.split("def get_refine_free_gaussians_command", 1)[1]
    refine_command = refine_command.split("render_all_img_command", 1)[0]
    assert '"--chart-geometry-sampling-policy", args.chart_geometry_sampling_policy' in refine_command


def test_train_pipeline_forwards_densification_statistics_policy_and_trainer_gates_both_inputs():
    pipeline = Path("train.py").read_text()
    refine_command = pipeline.split("def get_refine_free_gaussians_command", 1)[1]
    refine_command = refine_command.split("render_all_img_command", 1)[0]
    trainer = Path("2d-gaussian-splatting/train_with_refine_depth.py").read_text()

    assert '"--densification-view-policy", args.densification_view_policy' in refine_command
    assert 'policy=densification_view_policy' in trainer
    # Both statistics drive topology: radii affect pruning and the gradient
    # accumulator controls splitting.  A one-sided gate would leave the
    # Chart-capacity confound in place.
    guarded = trainer.split("# Densification", 1)[1].split("# Optimizer step", 1)[0]
    assert "if update_densification_stats:" in guarded
    assert "gaussians.max_radii2D" in guarded
    assert "gaussians.add_densification_stats" in guarded
    assert '"densification_stats"' in trainer


def test_train_pipeline_forwards_strict_chart_prior_weight():
    pipeline = Path("train.py").read_text()
    refine_command = pipeline.split("def get_refine_free_gaussians_command", 1)[1]
    refine_command = refine_command.split("render_all_img_command", 1)[0]
    wrapper = Path("scripts/refine_free_gaussians.py").read_text()
    trainer = Path("2d-gaussian-splatting/train_with_refine_depth.py").read_text()

    assert '"--chart-geometry-prior-weight", str(args.chart_geometry_prior_weight)' in refine_command
    assert '"--chart-geometry-prior-weight", str(args.chart_geometry_prior_weight)' in wrapper
    assert "scale_chart_geometry_priors(" in trainer
    assert "'--chart-geometry-prior-weight'," in pipeline
    assert "'--chart_geometry_prior_weight'," in pipeline


def test_nonwarp_initialization_keeps_depth_image_and_mask_subset_in_lockstep():
    source = Path("2d-gaussian-splatting/train_with_refine_depth.py").read_text()
    initialization = source.split("input_view_gaussian_params = get_gaussian_parameters_from_pa_data(", 1)[1]
    initialization = initialization.split("        if use_pseudo_initialization:", 1)[0]
    assert "visibility_masks=[initialization_valid_masks[i] for i in init_view_ids]" in initialization
    assert "else initialization_valid_masks" not in initialization


def test_plane_frontend_validates_chart_tensor_camera_order_before_rendering():
    source = Path("2d-gaussian-splatting/render_chart_views.py").read_text()
    assert "validate_chart_camera_order(" in source
    assert "[view.image_name for view in train_viewpoints]" in source


def test_refinement_persists_the_actual_chart_scheduler_audit():
    source = Path("2d-gaussian-splatting/train_with_refine_depth.py").read_text()
    assert '"chart_geometry_scheduler.json"' in source
    assert '"active_chart_indices": active_input_chart_indices' in source
    assert '"scheduled_chart_indices": geometry_input_chart_indices' in source
    assert '"chart_importance_weight": chart_rgb_sampling_weight' in source
    assert '"all_scheduled_chart_names_present_in_dense_set"' in source


def test_dense_refinement_does_not_label_chart_only_samples_as_all_train_metrics():
    source = Path("2d-gaussian-splatting/train_with_refine_depth.py").read_text()
    assert "report_train_cameras=(" in source
    assert '"dense_train_samples" if use_dense_supervision' in source
    assert 'report_train_name="chart_train_samples"' in source
