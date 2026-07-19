import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import shlex
import subprocess

def run_command_safe(command):
    print(f"Running command: {shlex.join(command)}")
    subprocess.run(command, check=True)
    print("Command succeeded!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--plane_root_dir", type=str, required=True)
    parser.add_argument("--iteration", required=True, type=str)
    parser.add_argument("--see3d_stage", required=True, type=int)
    parser.add_argument("--select_inpaint_num", required=True, type=str)
    parser.add_argument(
        "--depthanythingv2_checkpoint_dir",
        type=str,
        default="./Depth-Anything-V2/checkpoints/",
    )
    parser.add_argument("--depthanything_encoder", type=str, default="vitl")
    parser.add_argument(
        "--scene_aligned_cameras",
        action="store_true",
        help="Keep generated views on the calibrated camera pose manifold.",
    )
    parser.add_argument(
        "--preserve_visible_render",
        action="store_true",
        help="Keep the current GS render where the novel view is already observed.",
    )
    parser.add_argument(
        "--skip_plane_views",
        action="store_true",
        help="Select See3D candidates without globally refined plane point clouds.",
    )
    parser.add_argument(
        "--quality_filter",
        action="store_true",
        help="Reject risky pseudo-views before aggregating them into refinement data.",
    )
    parser.add_argument("--max_synthesized_fraction", type=float, default=0.35)
    parser.add_argument("--min_visible_sharpness", type=float, default=100.0)
    parser.add_argument("--max_alignment_rmse", type=float, default=0.12)
    parser.add_argument("--min_alignment_support_pixels", type=int, default=20_000)
    parser.add_argument(
        "--skip_plane_extraction",
        action="store_true",
        help="Do not run SAM plane extraction for conservative warm-start refinement.",
    )
    parser.add_argument(
        "--artifact_mask_detector",
        action="store_true",
        help="Augment visibility masks with conservative no-reference artifact masks.",
    )
    parser.add_argument("--artifact_detector_repo", type=str, default="/root/STDLoc")
    parser.add_argument("--artifact_max_edit_fraction", type=float, default=0.35)
    parser.add_argument("--max_reference_views", type=int, default=6)
    args = parser.parse_args()

    # 1. render novel views
    command = [
        sys.executable,
        "2d-gaussian-splatting/render_novel_views.py",
        "--source_path", args.source_path,
        "--model_path", args.model_path,
        "--iteration", args.iteration,
        "--see3d_stage", str(args.see3d_stage),
        "--select_inpaint_num", args.select_inpaint_num,
        "--max_reference_views", str(args.max_reference_views),
    ]
    if args.scene_aligned_cameras:
        command.append("--scene_aligned_cameras")
    if args.skip_plane_views:
        command.append("--skip_plane_views")
    run_command_safe(command)

    ref_image_path = os.path.join(args.source_path, 'see3d_render', 'ref-views')
    warp_image_path = os.path.join(args.source_path, 'see3d_render', f'stage{args.see3d_stage}', 'select-gs')
    if args.artifact_mask_detector:
        repair_warp_path = os.path.join(
            args.source_path, 'see3d_render', f'stage{args.see3d_stage}', 'select-gs-repair'
        )
        command = [
            sys.executable,
            "scripts/augment_see3d_masks.py",
            "--input", warp_image_path,
            "--output", repair_warp_path,
            "--detector_repo", args.artifact_detector_repo,
            "--max_edit_fraction", str(args.artifact_max_edit_fraction),
            "--replace",
        ]
        run_command_safe(command)
        warp_image_path = repair_warp_path
    output_root_dir = os.path.join(args.source_path, 'see3d_render', f'stage{args.see3d_stage}', 'select-gs-inpainted')

    # 2. inpaint rgb
    command = [
        sys.executable,
        "2d-gaussian-splatting/guidance/see3d_util.py",
        "--ref_imgs_dir", ref_image_path,
        "--warp_root_dir", warp_image_path,
        "--output_root_dir", output_root_dir,
    ]
    run_command_safe(command)

    inpaint_dir_name = 'select-gs-inpainted'
    if args.preserve_visible_render:
        command = [
            sys.executable,
            "2d-gaussian-splatting/guidance/merge_util.py",
            "--source_path", args.source_path,
            "--see3d_stage", str(args.see3d_stage),
            "--plane_root_dir", args.plane_root_dir,
            "--prepare_only",
            "--warp_root_dir", warp_image_path,
        ]
        run_command_safe(command)
        inpaint_dir_name = 'select-gs-inpainted-merged'

    # 3. generate depth and normal
    command = [
        sys.executable,
        "2d-gaussian-splatting/guidance/see3d_dn_util.py",
        "--source_path", args.source_path,
        "--see3d_stage", str(args.see3d_stage),
        "--inpaint_dir_name", inpaint_dir_name,
        "--depthanythingv2_checkpoint_dir", args.depthanythingv2_checkpoint_dir,
        "--depthanything_encoder", args.depthanything_encoder,
    ]
    run_command_safe(command)

    accepted_indices_path = None
    if args.quality_filter:
        stage_root = os.path.join(
            args.source_path, 'see3d_render', f'stage{args.see3d_stage}'
        )
        command = [
            sys.executable,
            "scripts/filter_see3d_views.py",
            "--stage_root", stage_root,
            "--max_synthesized_fraction", str(args.max_synthesized_fraction),
            "--min_visible_sharpness", str(args.min_visible_sharpness),
            "--max_alignment_rmse", str(args.max_alignment_rmse),
            "--min_alignment_support_pixels", str(args.min_alignment_support_pixels),
        ]
        run_command_safe(command)
        accepted_indices_path = os.path.join(stage_root, 'accepted_view_indices.json')

    # 4. generate 2D planes
    cur_plane_root_dir = os.path.join(args.source_path, 'see3d_render', f'stage{args.see3d_stage}', 'select-gs-planes')
    if not args.skip_plane_extraction:
        command = [
            sys.executable,
            "2d-gaussian-splatting/planes/plane_excavator.py",
            "--plane_root_path", cur_plane_root_dir,
        ]
        run_command_safe(command)

    # 5. merge results
    anchor_view_id_json_path = os.path.join(args.source_path, 'see3d_render', f'stage{args.see3d_stage}', 'anchor_view_id.json')
    command = [
        sys.executable,
        "2d-gaussian-splatting/guidance/merge_util.py",
        "--source_path", args.source_path,
        "--see3d_stage", str(args.see3d_stage),
        "--plane_root_dir", args.plane_root_dir,
        "--anchor_view_id_json_path", anchor_view_id_json_path,
    ]
    if args.preserve_visible_render:
        command.append("--premerged")
    else:
        command.append("--none_replace")
    if accepted_indices_path is not None:
        command.extend(["--accepted_indices_json", accepted_indices_path])
    if args.skip_plane_extraction:
        command.append("--skip_plane_files")
    command.extend(["--warp_root_dir", warp_image_path])
    run_command_safe(command)

    print(f'See3D stage {args.see3d_stage} inpaint done!')
