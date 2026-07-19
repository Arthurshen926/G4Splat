import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '2d-gaussian-splatting'))
from utils.general_utils import seed_everything
from scene.dataset_readers import load_see3d_cameras
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import shutil
import json


def run_command_safe(command):
    print(f"Running command: {command}")
    exit_code = os.system(command)
    if exit_code != 0:
        print("Command failed!")
        sys.exit(1)
    else:
        print("Command succeeded!")

def replace_inpaint_results(warp_root_dir, inpaint_root_dir, save_root_dir):
    os.makedirs(save_root_dir, exist_ok=True)
    inpaint_img_list = sorted(
        img for img in os.listdir(inpaint_root_dir) if img.endswith('.png')
    )
    img_num = len(inpaint_img_list)
    for idx in range(img_num):
        gs_render_img_path = os.path.join(warp_root_dir, f'warp_frame{idx:06d}.png')
        mask_img_path = os.path.join(warp_root_dir, f'mask_frame{idx:06d}.png')
        inpaint_img_path = os.path.join(inpaint_root_dir, f'predict_warp_frame{idx:06d}.png')

        gs_render_img = Image.open(gs_render_img_path)
        mask_img = Image.open(mask_img_path)
        inpaint_img = Image.open(inpaint_img_path)

        mask_map = np.array(mask_img) / 255
        gs_render_img = np.array(gs_render_img)
        inpaint_img = np.array(inpaint_img)

        save_img = inpaint_img.copy()
        save_img[mask_map == 1] = gs_render_img[mask_map == 1]          # NOTE: visible part is gs_render_img, invisible part is inpaint_img
        save_img = Image.fromarray(save_img)
        save_img.save(os.path.join(save_root_dir, f'predict_warp_frame{idx:06d}.png'))

        print(f'Inpaint {idx} replace done!')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--source_path', type=str)
    parser.add_argument('--plane_root_dir', type=str)
    parser.add_argument("--see3d_stage", required=True, type=int)
    parser.add_argument(
        "--warp_root_dir",
        type=str,
        default=None,
        help="Override the stage select-gs condition/mask directory.",
    )
    parser.add_argument("--inpaint_root_dir", type=str, default=None)
    parser.add_argument("--save_root_dir", type=str, default=None)
    parser.add_argument("--none_replace", action='store_true')
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Merge trusted GS pixels into See3D RGB and stop before geometry aggregation.",
    )
    parser.add_argument(
        "--premerged",
        action="store_true",
        help="Reuse select-gs-inpainted-merged prepared before depth/normal inference.",
    )
    parser.add_argument("--anchor_view_id_json_path", type=str, default=None)
    parser.add_argument(
        "--accepted_indices_json",
        type=str,
        default=None,
        help="Optional stage-local See3D view allowlist produced by filter_see3d_views.py.",
    )
    parser.add_argument(
        "--skip_plane_files",
        action="store_true",
        help="Aggregate RGB/depth/visibility cues without SAM plane masks.",
    )
    args = parser.parse_args()

    seed_everything()

    see3d_root_dir = os.path.join(args.source_path, 'see3d_render')
    cur_see3d_root_dir = os.path.join(see3d_root_dir, f'stage{args.see3d_stage}')
    inpaint_root_dir = args.inpaint_root_dir or os.path.join(
        cur_see3d_root_dir, 'select-gs-inpainted'
    )
    save_root_dir = args.save_root_dir or os.path.join(
        cur_see3d_root_dir, 'select-gs-inpainted-merged'
    )

    warp_root_dir = args.warp_root_dir or os.path.join(cur_see3d_root_dir, 'select-gs')
    accepted_local_indices = None
    if args.accepted_indices_json is not None:
        with open(args.accepted_indices_json, 'r') as f:
            accepted_local_indices = json.load(f)
        if not isinstance(accepted_local_indices, list) or not all(
            isinstance(index, int) for index in accepted_local_indices
        ):
            raise ValueError("--accepted_indices_json must contain a JSON list of integers")
        if len(set(accepted_local_indices)) != len(accepted_local_indices):
            raise ValueError("--accepted_indices_json contains duplicate indices")
    # 1. replace inpaint results
    if args.prepare_only and args.none_replace:
        raise ValueError("--prepare_only cannot be combined with --none_replace")
    if args.prepare_only and args.premerged:
        raise ValueError("--prepare_only cannot be combined with --premerged")

    if args.prepare_only:
        replace_inpaint_results(warp_root_dir, inpaint_root_dir, save_root_dir)
        print(f'See3D stage {args.see3d_stage} prepared protected RGB inputs.')
        sys.exit(0)

    if not args.none_replace and not args.premerged:
        replace_inpaint_results(warp_root_dir, inpaint_root_dir, save_root_dir)
        print(f'See3D stage {args.see3d_stage} replace inpaint results done!')
    if args.premerged and not os.path.isdir(save_root_dir):
        raise FileNotFoundError(
            f"Missing protected See3D RGB directory: {save_root_dir}. "
            "Run merge_util.py --prepare_only first."
        )

    # 2. copy inpaint results to all inpaint folder (NOTE: begin_idx is id in all inpaint images)
    all_inpaint_image_dir = os.path.join(see3d_root_dir, 'inpainted_images')
    all_visible_mask_dir = os.path.join(see3d_root_dir, 'visible_masks')
    os.makedirs(all_visible_mask_dir, exist_ok=True)
    if not os.path.exists(all_inpaint_image_dir):
        os.makedirs(all_inpaint_image_dir, exist_ok=True)
        begin_idx = 0
    else:
        begin_idx = len(os.listdir(all_inpaint_image_dir))

    cur_result_dir = save_root_dir if not args.none_replace else inpaint_root_dir
    inpaint_img_list = sorted(
        image_name
        for image_name in os.listdir(cur_result_dir)
        if image_name.endswith('.png')
    )
    copy_local_indices = (
        list(range(len(inpaint_img_list)))
        if accepted_local_indices is None
        else accepted_local_indices
    )
    for local_idx in copy_local_indices:
        result_img_name = f'predict_warp_frame{local_idx:06d}.png'
        if result_img_name not in inpaint_img_list:
            raise FileNotFoundError(os.path.join(cur_result_dir, result_img_name))
        inpaint_img_path = os.path.join(cur_result_dir, result_img_name)
        shutil.copy(
            inpaint_img_path,
            os.path.join(all_inpaint_image_dir, f'predict_warp_frame{begin_idx:06d}.png'),
        )
        shutil.copy(
            os.path.join(warp_root_dir, f'mask_frame{local_idx:06d}.png'),
            os.path.join(all_visible_mask_dir, f'visible_mask_frame{begin_idx:06d}.png'),
        )
        begin_idx += 1
    print(f'See3D stage {args.see3d_stage} copy inpaint results to all inpaint folder done!')

    # 3. merge novel cameras
    see3d_cam_path = os.path.join(see3d_root_dir, 'see3d_cameras.npz')
    if os.path.exists(see3d_cam_path):
        pre_see3d_cameras = np.load(see3d_cam_path)
        pre_see3d_cameras = dict(pre_see3d_cameras)                 # NOTE: convert npz to dict
        pre_see3d_views = pre_see3d_cameras['n_views']

        os.remove(see3d_cam_path)
    else:
        pre_see3d_cameras = {}
        pre_see3d_views = 0

    cur_see3d_cam_path = os.path.join(cur_see3d_root_dir, f'stage{args.see3d_stage}_see3d_cameras.npz')
    cur_see3d_cameras = np.load(cur_see3d_cam_path)
    original_cur_see3d_views = int(cur_see3d_cameras['n_views'])
    camera_local_indices = (
        list(range(original_cur_see3d_views))
        if accepted_local_indices is None
        else accepted_local_indices
    )
    if any(index < 0 or index >= original_cur_see3d_views for index in camera_local_indices):
        raise ValueError(
            f"Accepted See3D indices must be in [0, {original_cur_see3d_views})"
        )
    cur_see3d_views = len(camera_local_indices)

    for output_index, local_index in enumerate(camera_local_indices):
        cur_id = output_index + pre_see3d_views
        pre_see3d_cameras[f'R_{cur_id:06d}'] = cur_see3d_cameras[f'R_{local_index:06d}']
        pre_see3d_cameras[f'T_{cur_id:06d}'] = cur_see3d_cameras[f'T_{local_index:06d}']
        pre_see3d_cameras[f'FoVx_{cur_id:06d}'] = cur_see3d_cameras[f'FoVx_{local_index:06d}']
        pre_see3d_cameras[f'FoVy_{cur_id:06d}'] = cur_see3d_cameras[f'FoVy_{local_index:06d}']
        pre_see3d_cameras[f'image_width_{cur_id:06d}'] = cur_see3d_cameras[f'image_width_{local_index:06d}']
        pre_see3d_cameras[f'image_height_{cur_id:06d}'] = cur_see3d_cameras[f'image_height_{local_index:06d}']

    pre_see3d_cameras['n_views'] = cur_see3d_views + pre_see3d_views
    if 'train_views' not in pre_see3d_cameras:
        pre_see3d_cameras['train_views'] = cur_see3d_cameras['train_views']
    np.savez(see3d_cam_path, **pre_see3d_cameras)
    print(f'See3D stage {args.see3d_stage} merge novel cameras done!')

    # 4. merge geometry cues (NOTE: begin_plane_idx is id in all plane images, input views + novel views)
    plane_root_dir = args.plane_root_dir
    cur_plane_root_dir = os.path.join(cur_see3d_root_dir, 'select-gs-planes')
    if os.path.exists(plane_root_dir):
        plane_file_list = os.listdir(plane_root_dir)
        plane_rgb_list = [file for file in plane_file_list if 'rgb_frame' in file]
        begin_plane_idx = len(plane_rgb_list)
    else:
        os.makedirs(plane_root_dir, exist_ok=True)
        begin_plane_idx = 0

    anchor_view_id_list = []
    for i in camera_local_indices:
        # rgb
        shutil.copy(os.path.join(cur_plane_root_dir, f'rgb_frame{i:06d}.png'), os.path.join(plane_root_dir, f'rgb_frame{begin_plane_idx:06d}.png'))

        # depth
        shutil.copy(os.path.join(cur_plane_root_dir, f'depth_frame{i:06d}.tiff'), os.path.join(plane_root_dir, f'depth_frame{begin_plane_idx:06d}.tiff'))
        shutil.copy(os.path.join(cur_plane_root_dir, f'depth_frame{i:06d}.png'), os.path.join(plane_root_dir, f'depth_frame{begin_plane_idx:06d}.png'))

        shutil.copy(os.path.join(cur_plane_root_dir, f'mono_depth_frame{i:06d}.tiff'), os.path.join(plane_root_dir, f'mono_depth_frame{begin_plane_idx:06d}.tiff'))
        shutil.copy(os.path.join(cur_plane_root_dir, f'mono_depth_frame{i:06d}.png'), os.path.join(plane_root_dir, f'mono_depth_frame{begin_plane_idx:06d}.png'))

        # normal
        shutil.copy(os.path.join(cur_plane_root_dir, f'depth_normal_world_frame{i:06d}.npy'), os.path.join(plane_root_dir, f'depth_normal_world_frame{begin_plane_idx:06d}.npy'))
        shutil.copy(os.path.join(cur_plane_root_dir, f'depth_normal_world_frame{i:06d}.png'), os.path.join(plane_root_dir, f'depth_normal_world_frame{begin_plane_idx:06d}.png'))

        shutil.copy(os.path.join(cur_plane_root_dir, f'mono_normal_world_frame{i:06d}.npy'), os.path.join(plane_root_dir, f'mono_normal_world_frame{begin_plane_idx:06d}.npy'))
        shutil.copy(os.path.join(cur_plane_root_dir, f'mono_normal_world_frame{i:06d}.png'), os.path.join(plane_root_dir, f'mono_normal_world_frame{begin_plane_idx:06d}.png'))

        shutil.copy(os.path.join(cur_plane_root_dir, f'mono_normal_frame{i:06d}.npy'), os.path.join(plane_root_dir, f'mono_normal_frame{begin_plane_idx:06d}.npy'))
        shutil.copy(os.path.join(cur_plane_root_dir, f'mono_normal_frame{i:06d}.png'), os.path.join(plane_root_dir, f'mono_normal_frame{begin_plane_idx:06d}.png'))

        # visibility
        shutil.copy(os.path.join(cur_plane_root_dir, f'visibility_frame{i:06d}.npy'), os.path.join(plane_root_dir, f'visibility_frame{begin_plane_idx:06d}.npy'))
        shutil.copy(os.path.join(cur_plane_root_dir, f'visibility_frame{i:06d}.png'), os.path.join(plane_root_dir, f'visibility_frame{begin_plane_idx:06d}.png'))

        # 2D planes are optional for the conservative MAtCha warm-start path.
        if not args.skip_plane_files:
            shutil.copy(os.path.join(cur_plane_root_dir, f'plane_mask_frame{i:06d}.npy'), os.path.join(plane_root_dir, f'plane_mask_frame{begin_plane_idx:06d}.npy'))
            shutil.copy(os.path.join(cur_plane_root_dir, f'plane_vis_frame{i:06d}.png'), os.path.join(plane_root_dir, f'plane_vis_frame{begin_plane_idx:06d}.png'))

        anchor_view_id_list.append(begin_plane_idx)

        begin_plane_idx += 1

    # copy need inpaint views points
    shutil.copy(os.path.join(cur_see3d_root_dir, f'stage{args.see3d_stage}_need_inpaint_views_points.ply'), os.path.join(plane_root_dir, f'stage{args.see3d_stage}_need_inpaint_views_points.ply'))

    if args.anchor_view_id_json_path is None:
        raise ValueError("--anchor_view_id_json_path is required when aggregating a See3D stage")
    with open(args.anchor_view_id_json_path, 'w') as f:
        json.dump(anchor_view_id_list, f)

    print(f'See3D stage {args.see3d_stage} merge geometry cues done!')
