import os
import shutil
import sys
import time
from argparse import ArgumentParser

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "2d-gaussian-splatting"))

import numpy as np
from PIL import Image
import torch
import trimesh

from arguments import ModelParams
from guidance.cam_utils import project_points_to_image
from scene.dataset_readers import load_cameras, load_see3d_cameras
from utils.general_utils import safe_state


def project_visible_points(viewpoint, points, refine_depth, depth_threshold):
    """Project one view without retaining an all-points/all-views matrix."""
    points_depth, points_2d, in_image = project_points_to_image(viewpoint, points)
    point_indices = torch.nonzero(in_image).squeeze(-1)
    if point_indices.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=points.device)
        return empty, empty, empty

    height, width = refine_depth.shape
    valid_points_2d = points_2d[in_image]
    u = torch.clamp(valid_points_2d[:, 0].long(), 0, width - 1)
    v = torch.clamp(valid_points_2d[:, 1].long(), 0, height - 1)
    valid_points_depth = points_depth[in_image]
    depth_at_pixels = refine_depth[v, u]
    relative_diff = torch.abs(valid_points_depth - depth_at_pixels) / (
        valid_points_depth.abs() + 1e-6
    )
    depth_valid = (relative_diff < depth_threshold) & (valid_points_depth > 0)
    return point_indices[depth_valid], u[depth_valid], v[depth_valid]


def load_refine_points(plane_root_path):
    point_files = sorted(
        file_name
        for file_name in os.listdir(plane_root_path)
        if "refine_points_frame" in file_name
    )
    points = []
    for file_name in point_files:
        point_path = os.path.join(plane_root_path, file_name)
        points.append(
            torch.tensor(
                trimesh.load(point_path).vertices,
                dtype=torch.float32,
                device="cuda",
            )
        )
    return points


def load_refine_depths(plane_root_path):
    depth_files = sorted(
        file_name
        for file_name in os.listdir(plane_root_path)
        if "refine_depth_frame" in file_name
    )
    return [
        torch.from_numpy(
            np.array(Image.open(os.path.join(plane_root_path, file_name)))
        ).cuda()
        for file_name in depth_files
    ]


def save_confident_map(plane_root_path, view_idx, confident_map):
    confident_map_vis = (confident_map * 255).astype(np.uint8)
    confident_map_path = os.path.join(
        plane_root_path,
        f"confident_map_frame{view_idx:06d}.png",
    )
    Image.fromarray(confident_map_vis).save(confident_map_path)

    rgb_path = os.path.join(plane_root_path, f"rgb_frame{view_idx:06d}.png")
    rgb_image = np.array(Image.open(rgb_path))
    rgb_image = rgb_image * confident_map[:, :, None]
    Image.fromarray(rgb_image.astype(np.uint8)).save(
        os.path.join(plane_root_path, f"confident_masked_frame{view_idx:06d}.png")
    )
    print(f"Saved confident map for view {view_idx} to {confident_map_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate See3D consistency masks")
    parser.add_argument("--plane_root_path", type=str, required=True)
    parser.add_argument("--see3d_root_path", type=str, default=None)
    model = ModelParams(parser, sentinel=True)
    args = parser.parse_args()

    print("NOTE: Using training views from data_path")
    safe_state(False)
    start_time = time.time()
    input_viewpoints, _ = load_cameras(model.extract(args))

    if args.see3d_root_path is None:
        print("Not need to generate confident maps for see3d views!")
        first_depth_path = os.path.join(
            args.plane_root_path,
            "refine_depth_frame000000.tiff",
        )
        height, width = np.array(Image.open(first_depth_path)).shape
        confident_map = np.ones((height, width), dtype=np.uint8)
        for view_idx in range(len(input_viewpoints)):
            save_confident_map(args.plane_root_path, view_idx, confident_map)
        sys.exit(0)

    see3d_root_path = args.see3d_root_path
    plane_root_path = args.plane_root_path
    depth_threshold = 0.1
    camera_path = os.path.join(see3d_root_path, "see3d_cameras.npz")
    inpaint_root_dir = os.path.join(see3d_root_path, "inpainted_images")
    print(f"NOTE: Using training views from camera_path {camera_path}")
    see3d_viewpoints, _ = load_see3d_cameras(camera_path, inpaint_root_dir)
    train_viewpoints = input_viewpoints + see3d_viewpoints
    input_view_num = len(input_viewpoints)

    print("********** load refine points **********")
    point_lists = load_refine_points(plane_root_path)
    if len(point_lists) != len(train_viewpoints):
        raise RuntimeError(
            f"Loaded {len(point_lists)} point sets for {len(train_viewpoints)} views"
        )
    points = torch.cat(point_lists, dim=0)
    del point_lists

    print("********** load refine depth **********")
    refine_depths = load_refine_depths(plane_root_path)
    if len(refine_depths) != len(train_viewpoints):
        raise RuntimeError(
            f"Loaded {len(refine_depths)} depth maps for {len(train_viewpoints)} views"
        )

    print("********** stream point visibility and pseudo-view colors **********")
    num_points = points.shape[0]
    seen_in_input = torch.zeros(num_points, dtype=torch.bool, device="cuda")
    for view_idx in range(input_view_num):
        visible_indices, _, _ = project_visible_points(
            train_viewpoints[view_idx],
            points,
            refine_depths[view_idx],
            depth_threshold,
        )
        seen_in_input[visible_indices] = True
        print(f"Accumulated input support for view {view_idx}")

    point_colors = torch.zeros((num_points, 3), dtype=torch.uint8, device="cuda")
    color_assigned = torch.zeros(num_points, dtype=torch.bool, device="cuda")
    pseudo_confident_maps = []
    pseudo_rgb_maps = []
    for view_idx in range(input_view_num, len(train_viewpoints)):
        viewpoint = train_viewpoints[view_idx]
        refine_depth = refine_depths[view_idx]
        visible_indices, u_coords, v_coords = project_visible_points(
            viewpoint,
            points,
            refine_depth,
            depth_threshold,
        )
        height, width = refine_depth.shape
        confident_map = torch.ones(
            (height, width),
            dtype=torch.uint8,
            device="cuda",
        )
        rgb_image = viewpoint.original_image.cuda().permute(1, 2, 0).clone()

        visible_seen_in_input = seen_in_input[visible_indices]
        confident_map[
            v_coords[visible_seen_in_input],
            u_coords[visible_seen_in_input],
        ] = 0

        novel_indices = visible_indices[~visible_seen_in_input]
        novel_u = u_coords[~visible_seen_in_input]
        novel_v = v_coords[~visible_seen_in_input]
        unassigned = ~color_assigned[novel_indices]
        if torch.any(unassigned):
            new_indices = novel_indices[unassigned]
            sampled_colors = (
                rgb_image[novel_v[unassigned], novel_u[unassigned]] * 255.0
            ).round().clamp(0, 255).to(torch.uint8)
            point_colors[new_indices] = sampled_colors
            color_assigned[new_indices] = True
        if novel_indices.numel() > 0:
            rgb_image[novel_v, novel_u] = point_colors[novel_indices].float() / 255.0

        pseudo_confident_maps.append(confident_map.cpu().numpy())
        pseudo_rgb_maps.append(
            (rgb_image.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        )
        print(
            f"Processed See3D view {view_idx - input_view_num}: "
            f"{visible_indices.numel()} visible point(s)"
        )

    print(
        f"Assigned colors to {int(color_assigned.sum().item())} point(s) "
        "not seen in input views"
    )

    old_inpaint_root_dir = os.path.join(see3d_root_path, "inpainted_images_ori")
    if os.path.exists(old_inpaint_root_dir):
        shutil.rmtree(old_inpaint_root_dir)
    os.rename(inpaint_root_dir, old_inpaint_root_dir)
    os.makedirs(inpaint_root_dir, exist_ok=True)
    for view_idx, rgb_map in enumerate(pseudo_rgb_maps):
        rgb_path = os.path.join(
            inpaint_root_dir,
            f"predict_warp_frame{view_idx:06d}.png",
        )
        Image.fromarray(rgb_map).save(rgb_path)
        print(f"Saved assigned rgb image for See3D view {view_idx} to {rgb_path}")

    print("********** save confident maps **********")
    for view_idx in range(len(train_viewpoints)):
        if view_idx < input_view_num:
            confident_map = np.ones(refine_depths[view_idx].shape, dtype=np.uint8)
            print(f"Input view {view_idx} confident map set to all 1")
        else:
            confident_map = pseudo_confident_maps[view_idx - input_view_num]
        save_confident_map(plane_root_path, view_idx, confident_map)

    elapsed = time.time() - start_time
    print(f"Streaming confident map generation completed! Time cost: {elapsed:.2f}s")
