import torch
from scene import Scene, GaussianModel
from scene.dataset_readers import load_see3d_cameras
import os
import sys
sys.path.append(os.getcwd())
import json
import numpy as np
import shutil
import cv2
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

from utils.render_utils import save_img_f32, save_img_u8
from tqdm import tqdm
from PIL import Image
import trimesh

from utils.general_utils import safe_state

from guidance.cam_utils import (
    generate_see3d_camera_by_lookat, 
    select_need_inpaint_views, 
    vis_camera_pose, 
    generate_see3d_camera_by_lookat_object_centric, 
    generate_random_perturbed_camera_poses, 
    generate_interpolated_camera_poses, 
    generate_look_around_camera_poses, 
    generate_see3d_camera_by_view_angle,
    generate_see3d_camera_by_lookat_none_vis_plane,
    generate_see3d_camera_by_lookat_all_plane,
    generate_pose_graph_interpolated_camera_poses,
)

from matcha.dm_scene.charts import depths_to_points_parallel

from guidance.vis_grid import VisibilityGrid
from planes.get_global_3Dpnts import get_none_vis_global_3Dpnts, get_visible_mask_for_input_views, get_all_global_3Dpnts


def select_reference_viewpoints(reference_viewpoints, target_viewpoints, count):
    if count <= 0 or count >= len(reference_viewpoints):
        return list(reference_viewpoints)
    if not target_viewpoints:
        return list(reference_viewpoints[:count])

    reference_centers = np.stack([
        camera.camera_center.detach().cpu().numpy() for camera in reference_viewpoints
    ])
    target_centers = np.stack([
        camera.camera_center.detach().cpu().numpy() for camera in target_viewpoints
    ])
    reference_forwards = np.stack([
        torch.linalg.inv(camera.world_view_transform)[2, :3].detach().cpu().numpy()
        for camera in reference_viewpoints
    ])
    target_forwards = np.stack([
        torch.linalg.inv(camera.world_view_transform)[2, :3].detach().cpu().numpy()
        for camera in target_viewpoints
    ])
    reference_forwards /= np.maximum(np.linalg.norm(reference_forwards, axis=1, keepdims=True), 1e-8)
    target_forwards /= np.maximum(np.linalg.norm(target_forwards, axis=1, keepdims=True), 1e-8)
    scene_scale = max(float(np.linalg.norm(reference_centers.max(0) - reference_centers.min(0))), 1e-6)
    center_distance = np.linalg.norm(
        reference_centers[:, None] - target_centers[None], axis=2
    ) / scene_scale
    direction_distance = 1.0 - np.clip(reference_forwards @ target_forwards.T, -1.0, 1.0)
    target_score = center_distance + 0.25 * direction_distance

    selected = []
    uncovered = set(range(len(target_viewpoints)))
    while len(selected) < count:
        best_index = None
        best_score = float("inf")
        target_indices = sorted(uncovered) if uncovered else list(range(len(target_viewpoints)))
        for index in range(len(reference_viewpoints)):
            if index in selected:
                continue
            score = float(target_score[index, target_indices].min())
            if selected:
                separation = np.min(
                    np.linalg.norm(
                        reference_centers[selected] - reference_centers[index], axis=1
                    )
                ) / scene_scale
                score -= 0.10 * min(float(separation), 0.25)
            if score < best_score:
                best_score = score
                best_index = index
        if best_index is None:
            break
        selected.append(best_index)
        covered_target = min(
            target_indices, key=lambda target: target_score[best_index, target]
        )
        uncovered.discard(covered_target)
    return [reference_viewpoints[index] for index in selected]


def novel_view_quality(rgb_path, mask_path):
    rgb = np.asarray(Image.open(rgb_path).convert('RGB'), dtype=np.uint8)
    known = np.asarray(Image.open(mask_path).convert('L'), dtype=np.uint8) > 127
    edit_fraction = float((~known).mean())
    if not np.any(known):
        return {
            'accepted': False,
            'edit_fraction': edit_fraction,
            'rgb_std': 0.0,
            'known_laplacian_variance': 0.0,
        }
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    rgb_std = float(rgb[known].astype(np.float32).std() / 255.0)
    laplacian_variance = float(laplacian[known].var())
    return {
        'accepted': bool(
            edit_fraction >= 0.002 and rgb_std >= 0.04 and laplacian_variance >= 0.5
        ),
        'edit_fraction': edit_fraction,
        'rgb_std': rgb_std,
        'known_laplacian_variance': laplacian_variance,
    }

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", required=True, type=str)
    parser.add_argument("--see3d_stage", required=True, type=int)
    parser.add_argument("--select_inpaint_num", required=True, type=str)
    parser.add_argument(
        "--scene_aligned_cameras",
        action="store_true",
        help="Generate See3D views on a pose-neighbor graph instead of fixed world axes.",
    )
    parser.add_argument(
        "--skip_plane_views",
        action="store_true",
        help=(
            "Do not generate candidates from globally refined planes. This supports "
            "warm-start runs where plane rewrites are intentionally disabled."
        ),
    )
    parser.add_argument(
        "--max_reference_views",
        type=int,
        default=6,
        help="Keep only pose-near, direction-compatible real reference views; zero keeps all.",
    )
    parser.add_argument(
        "--no_prefilter_novel_views",
        action="store_true",
        help="Disable blank/flat/no-edit candidate rejection before See3D.",
    )
    args = get_combined_args(parser)

    # Initialize system state (RNG)
    safe_state(False)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    train_viewpoints = scene.getTrainCameras().copy()
    reference_viewpoints = train_viewpoints.copy()
    input_view_num = len(train_viewpoints)

    see3d_render_path = os.path.join(args.source_path, 'see3d_render')
    os.makedirs(see3d_render_path, exist_ok=True)

    # Reference images are selected after the target novel cameras are known.
    ref_views_save_root_path = os.path.join(see3d_render_path, 'ref-views')
    os.makedirs(ref_views_save_root_path, exist_ok=True)

    # load see3d cameras
    see3d_cam_path = os.path.join(see3d_render_path, 'see3d_cameras.npz')
    if os.path.exists(see3d_cam_path):
        see3d_viewpoints, _ = load_see3d_cameras(see3d_cam_path, os.path.join(see3d_render_path, 'inpainted_images'))
        train_viewpoints.extend(see3d_viewpoints)
        print(f'Stage {args.see3d_stage} See3D cameras loaded from {see3d_cam_path}')
    else:
        if args.see3d_stage > 1:
            assert False, 'See3D cameras not found, but see3d_stage > 1'

    # save this stage see3d_render
    novel_views_save_root_path = os.path.join(see3d_render_path, f'stage{args.see3d_stage}')
    os.makedirs(novel_views_save_root_path, exist_ok=True)

    # render train views
    alpha_vis_thresh = 0.99
    train_save_root_path = os.path.join(novel_views_save_root_path, 'render-train-views')
    os.makedirs(train_save_root_path, exist_ok=True)
    train_view_depths = []
    for train_viewpoint in train_viewpoints:
        idx = train_viewpoint.colmap_id                                         # colmap_id is the unique index of the train viewpoint and see3d viewpoint
        render_pkg = render(train_viewpoint, gaussians, pipe, background)
        rgb = render_pkg['render']
        alpha = render_pkg['rend_alpha']
        depth = render_pkg['surf_depth']
        train_view_depths.append(depth[0].detach().cpu().numpy())

        save_img_u8(rgb.permute(1,2,0).detach().cpu().numpy(), os.path.join(train_save_root_path, f'{idx:05d}.png'))
        save_img_f32(depth[0].detach().cpu().numpy(), os.path.join(train_save_root_path, f'depth_{idx:05d}.tiff'))

    print(f'Train views render done!')

    input_view_depths = train_view_depths[:input_view_num]
    input_viewpoints = train_viewpoints[:input_view_num]
    gs_input_view_depths = np.array(input_view_depths)
    gs_input_view_depths = torch.from_numpy(gs_input_view_depths).cuda()
    gs_input_view_depths = gs_input_view_depths.unsqueeze(1)
    gs_input_view_points = depths_to_points_parallel(gs_input_view_depths, input_viewpoints)

    # init visibility grid
    bbox_min = torch.min(gaussians.get_xyz, dim=0).values
    bbox_max = torch.max(gaussians.get_xyz, dim=0).values
    grid_resolution = 256

    visibility_grid = VisibilityGrid(bbox_min, bbox_max, grid_resolution, train_viewpoints, train_view_depths)
    visibility_grid.vis_invisible_pnts(os.path.join(novel_views_save_root_path, 'invisible_points.ply'))

    # generate novel cameras
    novel_poses, novel_cams = [], []
    plane_root_path = os.path.join(args.source_path, 'plane-refine-depths')
    vis_plane_pnts_path = os.path.join(novel_views_save_root_path, f'stage{args.see3d_stage}_vis_global_3Dplane_points')
    if args.scene_aligned_cameras:
        chart_fovy_deg = float(np.rad2deg(np.median([float(cam.FoVy) for cam in input_viewpoints])))
        chart_fovx_deg = float(np.rad2deg(np.median([float(cam.FoVx) for cam in input_viewpoints])))

    if args.see3d_stage == 1:
        only_warp_input_views = False
        select_view_method = 'covisibility_rate'
        used_top_k = 5
        if args.scene_aligned_cameras:
            used_fovy_deg = chart_fovy_deg
            used_fovx_deg = chart_fovx_deg
            novel_poses_1, novel_cams_1 = generate_pose_graph_interpolated_camera_poses(
                input_viewpoints,
                visibility_grid,
                n_frames=80,
                neighbor_count=3,
                fovy_deg=used_fovy_deg,
                fovx_deg=used_fovx_deg,
            )
            novel_poses.extend(novel_poses_1)
            novel_cams.extend(novel_cams_1)
        else:
            used_fovy_deg = 80
            used_fovx_deg = 80

            # look at scene center
            novel_poses_1, novel_cams_1 = generate_see3d_camera_by_lookat_object_centric(train_viewpoints, visibility_grid, n_frames=40, fovy_deg=used_fovy_deg)
            novel_poses.extend(novel_poses_1)
            novel_cams.extend(novel_cams_1)

            # look at scene around
            novel_poses_2, novel_cams_2 = generate_see3d_camera_by_lookat(input_viewpoints, visibility_grid, gs_input_view_depths.squeeze(1), gs_input_view_points, n_frames=40, fovy_deg=used_fovy_deg)
            novel_poses.extend(novel_poses_2)
            novel_cams.extend(novel_cams_2)

    elif args.see3d_stage == 2:
        only_warp_input_views = False
        select_view_method = 'covisibility_rate'
        used_top_k = 5
        if args.scene_aligned_cameras:
            used_fovy_deg = chart_fovy_deg
            used_fovx_deg = chart_fovx_deg
            novel_poses_1, novel_cams_1 = generate_pose_graph_interpolated_camera_poses(
                input_viewpoints,
                visibility_grid,
                n_frames=60,
                neighbor_count=4,
                fovy_deg=used_fovy_deg,
                fovx_deg=used_fovx_deg,
            )
        else:
            used_fovy_deg = 80
            used_fovx_deg = 80
            # look around in input views position
            novel_poses_1, novel_cams_1 = generate_see3d_camera_by_view_angle(input_viewpoints, visibility_grid, fovy_deg=used_fovy_deg, n_frames=60)
        novel_poses.extend(novel_poses_1)
        novel_cams.extend(novel_cams_1)

    elif args.see3d_stage == 3:
        used_fovy_deg = chart_fovy_deg if args.scene_aligned_cameras else 100
        used_fovx_deg = chart_fovx_deg if args.scene_aligned_cameras else 100
        used_top_k = 10
        only_warp_input_views = True
        select_view_method = 'none_visible_rate'

    else:
        raise ValueError(f'Invalid see3d_stage: {args.see3d_stage}')
    
    if args.skip_plane_views:
        print("[INFO] Skipping global-plane camera proposals.")
    else:
        plane_all_points_dict = get_all_global_3Dpnts(
            args.source_path,
            plane_root_path,
            see3d_render_path,
            vis_plane_pnts_path,
            top_k=used_top_k,
        )
        novel_poses_3, novel_cams_3 = generate_see3d_camera_by_lookat_all_plane(
            train_viewpoints,
            visibility_grid,
            plane_all_points_dict,
            fovy_deg=used_fovy_deg,
            fovx_deg=used_fovx_deg,
            use_camera_vertical=args.scene_aligned_cameras,
        )
        novel_poses.extend(novel_poses_3)
        novel_cams.extend(novel_cams_3)

    # # vis train camera
    # train_c2ws = []
    # for train_cam in train_viewpoints:
    #     w2c = (train_cam.world_view_transform).transpose(0, 1).cpu().numpy()
    #     c2w = np.linalg.inv(w2c)
    #     train_c2ws.append(c2w)
    # train_c2ws = np.array(train_c2ws)
    # temp_mesh_path = './data/replica/scan5/gt_mesh/scene_mesh.ply'
    # vis_camera_pose(novel_poses, mesh_path=temp_mesh_path)
    # # vis_camera_pose(train_c2ws, mesh_path=temp_mesh_path)
    # exit()


    # render gs
    gs_output_dir = os.path.join(novel_views_save_root_path, 'raw-gs')
    os.makedirs(gs_output_dir, exist_ok=True)

    gs_depths = []
    none_visible_rate_list = []
    alpha_list = []
    input_view_depths_cuda = [torch.from_numpy(depth).float().cuda() for depth in input_view_depths]
    for idx, novel_cam in enumerate(novel_cams):

        render_pkg = render(novel_cam, gaussians, pipe, background)
        rgb = render_pkg['render']
        alpha = render_pkg['rend_alpha']
        depth = render_pkg['surf_depth']

        gs_depths.append(depth[0].detach())

        save_img_u8(rgb.permute(1,2,0).detach().cpu().numpy(), os.path.join(gs_output_dir, f'ori_warp_frame{idx:06d}.png'))
        save_img_f32(depth[0].detach().cpu().numpy(), os.path.join(gs_output_dir, f'depth_frame{idx:06d}.tiff'))
        # save .npy
        np.save(os.path.join(gs_output_dir, f'alpha_{idx:06d}.npy'), alpha[0].detach().cpu().numpy())
        alpha_vis_mask = alpha[0].detach().cpu().numpy() > alpha_vis_thresh
        alpha_list.append(alpha_vis_mask)

        save_img_u8(alpha_vis_mask, os.path.join(gs_output_dir, f'alpha_mask_frame{idx:06d}.png'))

        # filter rgb use alpha
        rgb_filtered = rgb.permute(1,2,0).detach().cpu().numpy() * alpha_vis_mask[:,:,None]
        save_img_u8(rgb_filtered, os.path.join(gs_output_dir, f'alpha_warp_frame{idx:06d}.png'))

        if only_warp_input_views:
            # get visible mask in input views
            gs_points = depths_to_points_parallel(depth.unsqueeze(0), [novel_cam]).squeeze(0)
            visible_mask = get_visible_mask_for_input_views(input_viewpoints, input_view_depths_cuda, gs_points)
            visible_mask = visible_mask.reshape(rgb.shape[1], rgb.shape[2]).detach().cpu().numpy()
            save_img_u8(visible_mask, os.path.join(gs_output_dir, f'mask_frame{idx:06d}.png'))

            # filter rgb use visible mask
            rgb_filtered = rgb.permute(1,2,0).detach().cpu().numpy() * visible_mask[:,:,None]
            save_img_u8(rgb_filtered, os.path.join(gs_output_dir, f'warp_frame{idx:06d}.png'))

            none_visible_rate = 1 - visible_mask.sum() / (visible_mask.shape[0] * visible_mask.shape[1])
            none_visible_rate_list.append(none_visible_rate)

        print(f'Novel view {idx} save done!')

    if not only_warp_input_views:
        # render visibility map
        visibility_maps = visibility_grid.render_visibility_map(novel_cams, gs_depths)
        for idx in range(len(visibility_maps)):
            rgb_path = os.path.join(gs_output_dir, f'ori_warp_frame{idx:06d}.png')
            rgb_image = Image.open(rgb_path)
            rgb_image = np.array(rgb_image)

            vis_map = visibility_maps[idx].detach().cpu().numpy() > 0.5
            alpha_map = alpha_list[idx]
            vis_map = vis_map & alpha_map                            # filter vis_map by alpha_map
            vis_map = vis_map.astype(np.float32)

            none_visible_rate = 1 - vis_map.sum() / (vis_map.shape[0] * vis_map.shape[1])
            none_visible_rate_list.append(none_visible_rate)

            rgb_image = rgb_image * vis_map[:,:,None]
            rgb_image = Image.fromarray(rgb_image.astype(np.uint8))
            rgb_image.save(os.path.join(gs_output_dir, f'warp_frame{idx:06d}.png'))

            # save vis_map
            vis_map = vis_map.astype(np.uint8) * 255
            vis_map = Image.fromarray(vis_map)
            vis_map.save(os.path.join(gs_output_dir, f'mask_frame{idx:06d}.png'))

        print(f'Render visibility map done!')

    max_none_visible_thresh = 0.6
    if select_view_method == 'none_visible_rate':
        need_inpaint_views = [i for i in range(len(novel_cams)) if none_visible_rate_list[i] < max_none_visible_thresh]         # delete views with large none visible regions
    elif select_view_method == 'covisibility_rate':
        need_inpaint_views = select_need_inpaint_views(novel_cams, none_visible_rate_list, gaussians, int(args.select_inpaint_num), none_visible_rate_high_bound=max_none_visible_thresh, covisible_rate_high_bound=0.9)
    else:
        raise ValueError(f'Invalid select_view_method: {select_view_method}')

    quality_report = {}
    if not args.no_prefilter_novel_views:
        for index in range(len(novel_cams)):
            quality_report[index] = novel_view_quality(
                os.path.join(gs_output_dir, f'ori_warp_frame{index:06d}.png'),
                os.path.join(gs_output_dir, f'mask_frame{index:06d}.png'),
            )
        requested = int(args.select_inpaint_num)
        filtered = [index for index in need_inpaint_views if quality_report[index]['accepted']]
        remaining = sorted(
            (
                index for index in range(len(novel_cams))
                if index not in filtered
                and quality_report[index]['accepted']
                and none_visible_rate_list[index] < max_none_visible_thresh
            ),
            key=lambda index: abs(none_visible_rate_list[index] - 0.20),
        )
        need_inpaint_views = (filtered + remaining)[:requested]
        if not need_inpaint_views:
            raise RuntimeError('Novel-view quality prefilter rejected every See3D candidate')
        with open(os.path.join(novel_views_save_root_path, 'novel_view_quality.json'), 'w') as handle:
            json.dump(
                {
                    'selected_indices': need_inpaint_views,
                    'views': {str(index): value for index, value in quality_report.items()},
                },
                handle,
                indent=2,
            )

    print(f'Need inpaint views: {need_inpaint_views}')

    select_gs_output_dir = os.path.join(novel_views_save_root_path, 'select-gs')
    os.makedirs(select_gs_output_dir, exist_ok=True)
    need_inpaint_views_cams = [novel_cams[i] for i in need_inpaint_views]
    selected_references = select_reference_viewpoints(
        reference_viewpoints, need_inpaint_views_cams, int(args.max_reference_views)
    )
    for path in os.listdir(ref_views_save_root_path):
        full_path = os.path.join(ref_views_save_root_path, path)
        if os.path.isfile(full_path) or os.path.islink(full_path):
            os.remove(full_path)
    src_image_root_path = os.path.join(args.source_path, 'images')
    image_files = {
        os.path.splitext(name)[0]: name for name in os.listdir(src_image_root_path)
    }
    for viewpoint in selected_references:
        source_name = image_files.get(viewpoint.image_name)
        if source_name is None:
            raise FileNotFoundError(f'Missing reference image for {viewpoint.image_name}')
        shutil.copy(
            os.path.join(src_image_root_path, source_name),
            os.path.join(ref_views_save_root_path, source_name),
        )
    with open(os.path.join(novel_views_save_root_path, 'selected_reference_views.json'), 'w') as handle:
        json.dump([viewpoint.image_name for viewpoint in selected_references], handle, indent=2)
    print(f'Selected {len(selected_references)} pose-aware real reference views.')
    need_inpaint_views_depths = [gs_depths[i] for i in need_inpaint_views]
    need_inpaint_views_depths = torch.stack(need_inpaint_views_depths, dim=0)

    # get need inpaint views points
    invalid_depth_mask = need_inpaint_views_depths <= 1e-6
    need_inpaint_views_depths[invalid_depth_mask] = 1e-3
    need_inpaint_views_points = depths_to_points_parallel(need_inpaint_views_depths, need_inpaint_views_cams)

    # # vis each view need inpaint views points
    # for idx in range(len(need_inpaint_views_cams)):
    #     vis_points = (need_inpaint_views_points[idx].cpu().numpy()).reshape(-1, 3)
    #     invalid_points_mask = (invalid_depth_mask[idx].cpu().numpy()).reshape(-1)
    #     vis_points = vis_points[~invalid_points_mask]
    #     trimesh.PointCloud(vis_points).export(os.path.join(novel_views_save_root_path, f'stage{args.see3d_stage}_need_inpaint_views_points_frame{idx:06d}.ply'))

    need_inpaint_views_points = need_inpaint_views_points.reshape(-1, 3)
    invalid_depth_mask_flatten = invalid_depth_mask.reshape(-1)
    need_inpaint_views_points = need_inpaint_views_points[~invalid_depth_mask_flatten]
    
    # save need inpaint views points
    need_inpaint_views_points_path = os.path.join(novel_views_save_root_path, f'stage{args.see3d_stage}_need_inpaint_views_points.ply')
    trimesh.PointCloud(need_inpaint_views_points.cpu().numpy()).export(need_inpaint_views_points_path)
    print(f'Saved need inpaint views points to {need_inpaint_views_points_path}')

    # save need inpaint views cameras
    save_cameras = {}
    save_cameras['train_views'] = len(train_viewpoints)
    for idx, need_inpaint_view_cam in enumerate(need_inpaint_views_cams):
        save_cameras[f'R_{idx:06d}'] = need_inpaint_view_cam.R
        save_cameras[f'T_{idx:06d}'] = need_inpaint_view_cam.T
        save_cameras[f'FoVx_{idx:06d}'] = need_inpaint_view_cam.FoVx
        save_cameras[f'FoVy_{idx:06d}'] = need_inpaint_view_cam.FoVy
        save_cameras[f'image_width_{idx:06d}'] = need_inpaint_view_cam.image_width
        save_cameras[f'image_height_{idx:06d}'] = need_inpaint_view_cam.image_height

        ori_id = need_inpaint_views[idx]
        # copy ori_id warp_frame into select_gs_output_dir
        shutil.copy(os.path.join(gs_output_dir, f'ori_warp_frame{ori_id:06d}.png'), os.path.join(select_gs_output_dir, f'ori_warp_frame{idx:06d}.png'))
        shutil.copy(os.path.join(gs_output_dir, f'depth_frame{ori_id:06d}.tiff'), os.path.join(select_gs_output_dir, f'depth_frame{idx:06d}.tiff'))
        shutil.copy(os.path.join(gs_output_dir, f'alpha_{ori_id:06d}.npy'), os.path.join(select_gs_output_dir, f'alpha_{idx:06d}.npy'))
        shutil.copy(os.path.join(gs_output_dir, f'alpha_mask_frame{ori_id:06d}.png'), os.path.join(select_gs_output_dir, f'alpha_mask_frame{idx:06d}.png'))
        shutil.copy(os.path.join(gs_output_dir, f'alpha_warp_frame{ori_id:06d}.png'), os.path.join(select_gs_output_dir, f'alpha_warp_frame{idx:06d}.png'))
        shutil.copy(os.path.join(gs_output_dir, f'warp_frame{ori_id:06d}.png'), os.path.join(select_gs_output_dir, f'warp_frame{idx:06d}.png'))
        shutil.copy(os.path.join(gs_output_dir, f'mask_frame{ori_id:06d}.png'), os.path.join(select_gs_output_dir, f'mask_frame{idx:06d}.png'))

    # save need inpaint views cameras
    save_cameras['n_views'] = len(need_inpaint_views_cams)
    np.savez(os.path.join(novel_views_save_root_path, f'stage{args.see3d_stage}_see3d_cameras.npz'), **save_cameras)

    print(f'See3D stage {args.see3d_stage} save done!')
