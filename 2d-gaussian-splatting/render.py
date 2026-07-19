#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from scene import Scene
from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)
from scene.dataset_readers import CameraInfo
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.render_utils import save_img_u8
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
from utils.system_utils import searchForMaxIteration
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.render_utils import generate_path, create_videos

import open3d as o3d
from PIL import Image
from pathlib import Path
from types import SimpleNamespace


@torch.no_grad()
def export_rgb_stream(
    viewpoint_stack,
    *,
    gaussians,
    pipe,
    background: torch.Tensor,
    output_path: str,
    index_offset: int = 0,
) -> None:
    """Render and write RGB one camera at a time.

    ``GaussianExtractor.reconstruction`` intentionally retains RGB and depth
    maps for mesh extraction.  That is unsuitable for the 1,487-view
    train-fit audit, where it would accumulate tens of gigabytes before it
    writes the first image.  This path is used only by ``--rgb_only`` and
    therefore never changes geometry outputs or optimisation behaviour.
    """
    render_path = os.path.join(output_path, "renders")
    gts_path = os.path.join(output_path, "gt")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    for index, viewpoint_cam in tqdm(
        enumerate(viewpoint_stack),
        total=len(viewpoint_stack),
        desc="stream RGB images",
    ):
        output_index = index + index_offset
        rendered = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            rgb_only=True,
        )["render"]
        gt = viewpoint_cam.original_image[0:3, :, :]
        save_img_u8(
            gt.permute(1, 2, 0).cpu().numpy(),
            os.path.join(gts_path, f"{output_index:05d}.png"),
        )
        save_img_u8(
            rendered.permute(1, 2, 0).cpu().numpy(),
            os.path.join(render_path, f"{output_index:05d}.png"),
        )


def _colmap_camera_metadata(dataset) -> list[SimpleNamespace]:
    """Read ordered COLMAP camera metadata without opening every RGB image.

    The normal :class:`Scene` constructor retains a PIL image and a tensor for
    every camera.  For Cambridge's 1,487 real training views that is needlessly
    expensive for an RGB-only audit.  The sparse-model selection and image-name
    order match ``readColmapSceneInfo`` exactly; only image decoding is delayed
    until the corresponding camera is rendered.
    """
    source_path = Path(dataset.source_path)
    if dataset.eval:
        sparse_relative = "all-sparse/0"
        if not (source_path / sparse_relative).is_dir():
            sparse_relative = "sparse/0"
    else:
        dense_points = source_path / "dense-view-sparse/0/points3D.ply"
        sparse_relative = "dense-view-sparse/0" if dense_points.is_file() else "sparse/0"
    sparse_root = source_path / sparse_relative
    try:
        extrinsics = read_extrinsics_binary(str(sparse_root / "images.bin"))
        intrinsics = read_intrinsics_binary(str(sparse_root / "cameras.bin"))
    except OSError:
        extrinsics = read_extrinsics_text(str(sparse_root / "images.txt"))
        intrinsics = read_intrinsics_text(str(sparse_root / "cameras.txt"))

    image_root = source_path / (dataset.images or "images")
    records: list[SimpleNamespace] = []
    for extrinsic in extrinsics.values():
        intrinsic = intrinsics[extrinsic.camera_id]
        if intrinsic.model == "SIMPLE_PINHOLE":
            focal_x = focal_y = float(intrinsic.params[0])
        elif intrinsic.model == "PINHOLE":
            focal_x = float(intrinsic.params[0])
            focal_y = float(intrinsic.params[1])
        else:
            raise ValueError(
                "Lazy RGB export supports the same undistorted COLMAP models "
                f"as Scene: received {intrinsic.model!r}"
            )
        image_path = image_root / Path(extrinsic.name).name
        records.append(
            SimpleNamespace(
                uid=int(intrinsic.id),
                R=np.transpose(qvec2rotmat(extrinsic.qvec)),
                T=np.asarray(extrinsic.tvec),
                FovY=focal2fov(focal_y, int(intrinsic.height)),
                FovX=focal2fov(focal_x, int(intrinsic.width)),
                image_path=image_path,
                image_name=image_path.stem,
                width=int(intrinsic.width),
                height=int(intrinsic.height),
            )
        )
    return sorted(records, key=lambda item: item.image_name)


def _load_lazy_camera(dataset, record: SimpleNamespace, index: int):
    """Decode exactly one RGB image and construct the existing Camera type."""
    with Image.open(record.image_path) as opened:
        opened.load()
        image = opened.copy()
    camera_info = CameraInfo(
        uid=record.uid,
        R=record.R,
        T=record.T,
        FovY=record.FovY,
        FovX=record.FovX,
        image=image,
        image_path=str(record.image_path),
        image_name=record.image_name,
        width=record.width,
        height=record.height,
    )
    return loadCam(dataset, index, camera_info, 1.0)


@torch.no_grad()
def export_lazy_colmap_rgb_stream(
    records: list[SimpleNamespace],
    *,
    dataset,
    gaussians,
    pipe,
    background: torch.Tensor,
    output_path: str,
    index_offset: int = 0,
) -> None:
    """Render a metadata list with one decoded image/camera at a time."""
    render_path = os.path.join(output_path, "renders")
    gts_path = os.path.join(output_path, "gt")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    for local_index, record in tqdm(
        enumerate(records), total=len(records), desc="lazy stream RGB images"
    ):
        output_index = local_index + index_offset
        camera = _load_lazy_camera(dataset, record, output_index)
        rendered = render(camera, gaussians, pipe, background, rgb_only=True)["render"]
        gt = camera.original_image[0:3, :, :]
        save_img_u8(
            gt.permute(1, 2, 0).cpu().numpy(),
            os.path.join(gts_path, f"{output_index:05d}.png"),
        )
        save_img_u8(
            rendered.permute(1, 2, 0).cpu().numpy(),
            os.path.join(render_path, f"{output_index:05d}.png"),
        )
        del rendered, camera

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--train_view_start",
        type=int,
        default=0,
        help="Inclusive index in the deterministically ordered training-camera list.",
    )
    parser.add_argument(
        "--train_view_end",
        type=int,
        default=None,
        help="Exclusive index in the deterministically ordered training-camera list.",
    )
    parser.add_argument(
        "--rgb_only",
        action="store_true",
        help="Export RGB renders and ground truth only; omit depth visualization TIFFs.",
    )
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # A full Cambridge train-fit pass uses all 1,487 real cameras.  In the
    # narrow RGB-only evaluation mode below, avoid constructing ``Scene``:
    # Scene eagerly turns every source image into a tensor even though only one
    # camera is needed at a time.  Other rendering modes retain the original
    # Scene/GaussianExtractor implementation unchanged.
    lazy_rgb_train = (
        bool(getattr(args, "rgb_only", False))
        and not bool(getattr(args, "skip_train", False))
        and bool(getattr(args, "skip_test", False))
        and bool(getattr(args, "skip_mesh", False))
        and not bool(getattr(args, "render_path", False))
    )
    if lazy_rgb_train:
        dataset = model.extract(args)
        iteration = int(args.iteration)
        if iteration == -1:
            iteration = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
        gaussians = GaussianModel(dataset.sh_degree)
        point_cloud = os.path.join(
            args.model_path,
            "point_cloud",
            f"iteration_{iteration}",
            "point_cloud.ply",
        )
        print(f"Loading trained model at iteration {iteration}")
        gaussians.load_ply(point_cloud)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        train_dir = getattr(args, "output_dir", None) or os.path.join(
            args.model_path, "train", f"ours_{iteration}"
        )
        records = _colmap_camera_metadata(dataset)
        start = int(getattr(args, "train_view_start", 0))
        train_view_end = getattr(args, "train_view_end", None)
        end = len(records) if train_view_end is None else int(train_view_end)
        if start < 0 or end < start or end > len(records):
            raise ValueError(
                f"Invalid training-view range [{start}, {end}) for {len(records)} cameras"
            )
        print(
            f"Lazy rendering training-camera range [{start}, {end}) "
            f"out of {len(records)}"
        )
        export_lazy_colmap_rgb_stream(
            records[start:end],
            dataset=dataset,
            gaussians=gaussians,
            pipe=pipeline.extract(args),
            background=background,
            output_path=train_dir,
            index_offset=start,
        )
        raise SystemExit(0)


    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    train_dir = getattr(args, "output_dir", None) or os.path.join(
        args.model_path, 'train', "ours_{}".format(scene.loaded_iter)
    )
    test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)    
    
    if not args.skip_train:
        print("export training images ...")
        os.makedirs(train_dir, exist_ok=True)
        train_cameras = scene.getTrainCameras()
        # ``get_combined_args`` deliberately omits command-line options whose
        # value is ``None`` when it merges them with ``cfg_args``.  A model
        # created before these range flags existed therefore has no
        # ``train_view_end`` attribute at all.  Treat that omission as the
        # documented default (render through the final train camera), rather
        # than failing after the whole reconstruction has completed.
        start = getattr(args, "train_view_start", 0)
        train_view_end = getattr(args, "train_view_end", None)
        end = len(train_cameras) if train_view_end is None else train_view_end
        if start < 0 or end < start or end > len(train_cameras):
            raise ValueError(
                f"Invalid training-view range [{start}, {end}) for {len(train_cameras)} cameras"
            )
        selected_train_cameras = train_cameras[start:end]
        print(
            f"Rendering training-camera range [{start}, {end}) "
            f"out of {len(train_cameras)}"
        )
        if args.rgb_only:
            export_rgb_stream(
                selected_train_cameras,
                gaussians=gaussians,
                pipe=pipe,
                background=background,
                output_path=train_dir,
                index_offset=start,
            )
        else:
            gaussExtractor.reconstruction(selected_train_cameras)
            gaussExtractor.export_image(
                train_dir,
                rgb_only=False,
                index_offset=start,
            )
        
    
    if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
        print("export rendered testing images ...")
        os.makedirs(test_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTestCameras())
        gaussExtractor.export_image(test_dir, rgb_only=args.rgb_only)
    
    
    if args.render_path:
        print("render videos ...")
        traj_dir = os.path.join(args.model_path, 'traj', "ours_{}".format(scene.loaded_iter))
        os.makedirs(traj_dir, exist_ok=True)
        n_fames = 240
        cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
        gaussExtractor.reconstruction(cam_traj)
        gaussExtractor.export_image(traj_dir, rgb_only=args.rgb_only)
        create_videos(base_dir=traj_dir,
                    input_dir=traj_dir, 
                    out_name='render_traj', 
                    num_frames=n_fames)

    if not args.skip_mesh:
        print("export mesh ...")
        os.makedirs(train_dir, exist_ok=True)
        # set the active_sh to 0 to export only diffuse texture
        gaussExtractor.gaussians.active_sh_degree = 0
        gaussExtractor.reconstruction(scene.getTrainCameras())
        # extract the mesh and save
        if args.unbounded:
            name = 'fuse_unbounded.ply'
            mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
        else:
            name = 'fuse.ply'
            depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
            voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
            sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
            mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
        
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
        print("mesh saved at {}".format(os.path.join(train_dir, name)))
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))
