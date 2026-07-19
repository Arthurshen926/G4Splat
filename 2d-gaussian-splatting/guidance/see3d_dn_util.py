'''
This file is used to generate depth and normal maps for the see3d model.
'''
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '2d-gaussian-splatting'))
from scene.dataset_readers import load_see3d_cameras
import cv2
import numpy as np
import shutil
from argparse import ArgumentParser
from matcha.pointmap.depthanythingv2 import load_model, apply_depthanything, depth_linear_align
from matcha.dm_scene.cameras import GSCamera
from PIL import Image
import torch
import matplotlib.pyplot as plt
import json
from utils.point_utils import depth_to_normal
from utils.render_utils import save_img_f32, save_img_u8
from utils.general_utils import seed_everything
from matcha.see3d_geometry import build_render_support_mask, robust_align_inverse_depth

def get_surf_cam_normal(view, depth):
    world_normal_map = depth_to_normal(view, depth)
    surf_normal = world_normal_map.permute(2,0,1)
    surf_normal_cam = (surf_normal.permute(1,2,0) @ (view.world_view_transform[:3,:3])).permute(2,0,1)
    return surf_normal, surf_normal_cam


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--source_path', type=str)
    parser.add_argument("--see3d_stage", required=True, type=int)
    parser.add_argument(
        "--depthanythingv2_checkpoint_dir",
        type=str,
        default="./Depth-Anything-V2/checkpoints/",
    )
    parser.add_argument("--depthanything_encoder", type=str, default="vitl")
    parser.add_argument("--inpaint_dir_name", type=str, default="select-gs-inpainted")
    parser.add_argument("--support_mask_erosion", type=int, default=1)
    parser.add_argument("--min_depth_fit_samples", type=int, default=512)
    parser.add_argument("--max_depth_fit_relative_rmse", type=float, default=0.2)
    args = parser.parse_args()

    seed_everything()

    cur_see3d_root_dir = os.path.join(args.source_path, 'see3d_render', f'stage{args.see3d_stage}')
    warp_root_dir = os.path.join(cur_see3d_root_dir, 'select-gs')
    inpaint_root_dir = os.path.join(cur_see3d_root_dir, args.inpaint_dir_name)
    save_root_dir = os.path.join(cur_see3d_root_dir, 'select-gs-planes')                # save depth, normal and planes
    if os.path.exists(save_root_dir):
        shutil.rmtree(save_root_dir)
    os.makedirs(save_root_dir, exist_ok=True)

    device = 'cuda'
    visible_threshold = 0.9
    alignment_diagnostics = []

    # Load See3D cameras
    see3d_cameras_path = os.path.join(cur_see3d_root_dir, f'stage{args.see3d_stage}_see3d_cameras.npz')
    see3d_gs_cameras_list, _ = load_see3d_cameras(see3d_cameras_path, inpaint_root_dir)
    n_views = len(see3d_gs_cameras_list)

    # 1. rgb
    for i in range(n_views):
        inpaint_img_path = os.path.join(inpaint_root_dir, f'predict_warp_frame{i:06d}.png')
        save_img_path = os.path.join(save_root_dir, f'rgb_frame{i:06d}.png')
        shutil.copy(inpaint_img_path, save_img_path)

    # 2. depth and normal
    # generate mono depth by depth anything v2
    model = load_model(
        checkpoint_dir=args.depthanythingv2_checkpoint_dir,
        encoder=args.depthanything_encoder,
        device=device,
    )
    for i in range(n_views):
        rgb_path = os.path.join(save_root_dir, f'rgb_frame{i:06d}.png')
        rgb = np.array(Image.open(rgb_path)) / 255.0
        rgb = torch.from_numpy(rgb).to(device).float()
        mono_disp = apply_depthanything(model, rgb)                         # Compute inverse depths with DepthAnything

        # save mono depth as png
        mono_depth = 1. / mono_disp
        mono_depth = mono_depth.cpu().numpy()
        mono_depth_tiff_path = os.path.join(save_root_dir, f'mono_depth_frame{i:06d}.tiff')
        save_img_f32(mono_depth, mono_depth_tiff_path)
        mono_depth_path = os.path.join(save_root_dir, f'mono_depth_frame{i:06d}.png')
        plt.imsave(mono_depth_path, mono_depth, cmap='viridis')

        # linear align depth and generate normal
        warp_depth_path = os.path.join(warp_root_dir, f'depth_frame{i:06d}.tiff')
        warp_depth = np.array(Image.open(warp_depth_path))
        warp_depth = torch.from_numpy(warp_depth).to(device)

        alpha_path = os.path.join(warp_root_dir, f'alpha_{i:06d}.npy')
        alpha = torch.from_numpy(np.load(alpha_path)).to(device)
        visibility_path = os.path.join(warp_root_dir, f'mask_frame{i:06d}.png')
        visibility = torch.from_numpy(
            np.asarray(Image.open(visibility_path).convert('L')).copy()
        ).to(device).float() / 255.0
        visible_mask = build_render_support_mask(
            alpha,
            visibility,
            warp_depth,
            alpha_threshold=visible_threshold,
            erosion_radius=args.support_mask_erosion,
        )
        np.save(os.path.join(save_root_dir, f'visibility_frame{i:06d}.npy'), visible_mask.cpu().numpy())
        Image.fromarray((visible_mask.cpu().numpy() * 255.).astype(np.uint8)).save(os.path.join(save_root_dir, f'visibility_frame{i:06d}.png'))

        aligned_depth, fit_diagnostics = robust_align_inverse_depth(
            mono_disp,
            warp_depth,
            visible_mask,
            min_samples=args.min_depth_fit_samples,
            max_relative_rmse=args.max_depth_fit_relative_rmse,
        )
        frame_diagnostics = {"frame": i, **fit_diagnostics.to_dict()}
        alignment_diagnostics.append(frame_diagnostics)
        print(f"[Depth alignment] {frame_diagnostics}")
        surf_normal, surf_normal_cam = get_surf_cam_normal(see3d_gs_cameras_list[i], aligned_depth.unsqueeze(0))
        surf_normal = surf_normal.permute(1,2,0)
        surf_normal_cam = surf_normal_cam.permute(1,2,0)
        
        # NOTE: For See3D views, rendered depth is not available, so we use aligned depth to compute normals
        surf_normal_path = os.path.join(save_root_dir, f'depth_normal_world_frame{i:06d}.npy')
        np.save(surf_normal_path, surf_normal.cpu().numpy())
        save_img_u8(surf_normal.cpu().numpy() * 0.5 + 0.5, os.path.join(save_root_dir, f'depth_normal_world_frame{i:06d}.png'))

        mono_normal_world_path = os.path.join(save_root_dir, f'mono_normal_world_frame{i:06d}.npy')
        np.save(mono_normal_world_path, surf_normal.cpu().numpy())
        save_img_u8(surf_normal.cpu().numpy() * 0.5 + 0.5, os.path.join(save_root_dir, f'mono_normal_world_frame{i:06d}.png'))

        surf_normal_cam_path = os.path.join(save_root_dir, f'mono_normal_frame{i:06d}.npy')
        np.save(surf_normal_cam_path, surf_normal_cam.cpu().numpy())
        save_img_u8(surf_normal_cam.cpu().numpy() * 0.5 + 0.5, os.path.join(save_root_dir, f'mono_normal_frame{i:06d}.png'))

        # merge aligned depth and warp depth
        merge_depth = aligned_depth.clone()
        merge_depth[visible_mask] = warp_depth[visible_mask]            # use warp depth when visible

        save_img_f32(merge_depth.cpu().numpy(), os.path.join(save_root_dir, f'depth_frame{i:06d}.tiff'))
        merge_depth_path = os.path.join(save_root_dir, f'depth_frame{i:06d}.png')
        plt.imsave(merge_depth_path, merge_depth.cpu().numpy(), cmap='viridis')

        print(f'frame {i:06d} done!')

    with open(os.path.join(save_root_dir, 'depth_alignment_diagnostics.json'), 'w') as file:
        json.dump(alignment_diagnostics, file, indent=2)
    print('All frames done!')
