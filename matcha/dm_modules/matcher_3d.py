import torch
from torch.utils.checkpoint import checkpoint
from matcha.dm_scene.cameras import CamerasWrapper, P3DCameras
from matcha.dm_utils.rendering import depths_to_points_parallel


def get_points_depth_in_depthmap_parallel(
    pts:torch.Tensor, 
    depthmap:torch.Tensor, 
    cameras:CamerasWrapper,
    padding_mode='zeros',  # 'reflection', 'border'
    znear=1e-6,
    chunk_size=None,
):
    """_summary_

    Args:
        pts (torch.Tensor): Has shape (n_depths, N, 3).
        depthmap (torch.Tensor): Has shape (n_depths, H, W) or (n_depths, H, W, 1).
        p3d_camera (P3DCameras): Should contain n_depths cameras.

    Returns:
        _type_: _description_
    """
    n_depths, image_height, image_width = depthmap.shape[:3]
    n_points = pts.shape[1]

    if chunk_size is not None and chunk_size > 0 and n_points > chunk_size:
        map_z = depthmap.new_empty((n_depths, n_points))
        fov_mask = torch.empty((n_depths, n_points), dtype=torch.bool, device=pts.device)
        for start in range(0, n_points, chunk_size):
            end = min(start + chunk_size, n_points)
            map_z_chunk, fov_mask_chunk = get_points_depth_in_depthmap_parallel(
                pts=pts[:, start:end],
                depthmap=depthmap,
                cameras=cameras,
                padding_mode=padding_mode,
                znear=znear,
                chunk_size=None,
            )
            map_z[:, start:end] = map_z_chunk
            fov_mask[:, start:end] = fov_mask_chunk
        return map_z, fov_mask

    pts_projections = cameras.transform_points_world_to_view(pts)  # (n_depths, N, 3)
    fov_mask = pts_projections[..., 2] > 0.  # (n_depths, N)
    pts_projections.clamp(min=torch.tensor([[[-1e8, -1e8, znear]]]).to(pts_projections.device))
    
    pts_projections = cameras.project_points(pts_projections, points_are_already_in_view_space=True, znear=znear)  # (n_depths, N, 2)
    fov_mask = fov_mask & pts_projections.isfinite().all(dim=-1)  # (n_depths, N)
    pts_projections = pts_projections.nan_to_num(nan=0., posinf=0., neginf=0.)
    
    if False:
        print("TOREMOVE-pts_projections:", pts_projections.shape)
        print("TOREMOVE-pts_projections Min/Max/Mean/Std:", pts_projections.min(), pts_projections.max(), pts_projections.mean(), pts_projections.std())
        
    factor = -1 * min(image_height, image_width)
    factors = torch.tensor([[[factor / image_width, factor / image_height]]]).to(pts.device)  # (1, 1, 2)
    # pts_projections[..., 0] = factor / image_width * pts_projections[..., 0]
    # pts_projections[..., 1] = factor / image_height * pts_projections[..., 1]
    pts_projections = pts_projections[..., :2] * factors  # (n_depths, N, 2)
    pts_projections = pts_projections.view(n_depths, -1, 1, 2)

    depth_view = depthmap.reshape(n_depths, 1, image_height, image_width)  # (n_depths, 1, H, W)
    map_z = torch.nn.functional.grid_sample(
        input=depth_view,
        grid=pts_projections,
        mode='bilinear',
        padding_mode=padding_mode,  # 'reflection', 'zeros'
        align_corners=False,
    )  # (n_depths, 1, N, 1)
    map_z = map_z[:, 0, :, 0]  # (n_depths, N)
    fov_mask = (map_z > 0.) & fov_mask
    map_z = map_z * fov_mask
    
    return map_z, fov_mask


class Matcher3D:
    def __init__(
        self, 
        cameras:CamerasWrapper,
        reference_pts:torch.Tensor=None, 
        reference_depths:torch.Tensor=None,
        reference_masks:torch.Tensor=None,
        projection_chunk_size:int=262_144,
    ):
        """_summary_

        Args:
            reference_pts (torch.Tensor): Should have shape (n_charts, height, width, 3).
            reference_depths (torch.Tensor): Should have shape (n_charts, height, width).
            camera (CamerasWrapper): _description_
            match_thr (float): _description_
        """
        self.cameras = cameras
        self.znear = 1e-6
        self.projection_chunk_size = projection_chunk_size
        self.reference_masks = reference_masks
        self.update_references(reference_pts, reference_depths)
        
    @torch.no_grad()
    def update_references(
        self, 
        reference_pts:torch.Tensor=None, 
        reference_depths:torch.Tensor=None,
    ):
        if reference_pts is None and reference_depths is None:
            raise ValueError("Either reference_pts or reference_depths should be provided.")
        
        if reference_depths is None:  
            reference_depths = self.cameras.p3d_cameras.get_world_to_view_transform().transform_points(
                reference_pts
            )[..., 2]  # (n_charts, height, width)
            
        if reference_pts is None:
            reference_pts = depths_to_points_parallel(
                reference_depths,
                cameras=self.cameras,
            ).view(*reference_depths.shape, 3)  # (n_charts, height, width, 3)
            
        self.reference_pts = reference_pts  # (n_charts, height, width, 3)
        self.reference_depths = reference_depths  # (n_charts, height, width)
        self.n_charts, self.height, self.width, _ = reference_pts.shape
        self.reference_pts = reference_pts
        
    @torch.no_grad()
    def match(
        self,
        matching_thr:float,
        normal_threshold=None,
        source_stride: int = 1,
    ):
        if normal_threshold is not None:
            raise NotImplementedError("Normal threshold not implemented yet.")

        source_stride = max(int(source_stride), 1)
        reference_pts = self.reference_pts[
            :, ::source_stride, ::source_stride
        ].contiguous()
        source_height, source_width = reference_pts.shape[1:3]
        n_pts_per_chart = source_height * source_width
        all_points_to_match = reference_pts.reshape(-1, 3)
        n_total_points = all_points_to_match.shape[0]
        chunk_size = self.projection_chunk_size or n_total_points
        # Only the thresholded correspondence relation is used after this
        # method.  Keeping the dense float error tensor costs O(N^2HW) VRAM
        # and was a major source of avoidable pressure for long trajectories.
        reference_matches = torch.empty(
            (self.n_charts, n_total_points),
            dtype=torch.bool,
            device=self.reference_depths.device,
        )

        for start in range(0, n_total_points, chunk_size):
            end = min(start + chunk_size, n_total_points)
            points_to_match = all_points_to_match[start:end][None].expand(self.n_charts, -1, -1)

            # For each camera, get the depth of all points in the camera's view
            true_depth = self.cameras.p3d_cameras.get_world_to_view_transform().transform_points(points_to_match)[..., 2]

            # For each camera, get the depth of the projections of all points in the camera's depth map
            projected_depths, fov_mask = get_points_depth_in_depthmap_parallel(
                pts=points_to_match,
                depthmap=self.reference_depths,
                cameras=self.cameras,
                padding_mode='zeros',
                znear=self.znear,
                chunk_size=None,
            )

            depth_errors_chunk = (true_depth - projected_depths).abs()
            reference_matches[:, start:end] = (
                fov_mask
                & torch.isfinite(depth_errors_chunk)
                & (depth_errors_chunk < matching_thr)
            )

        self.reference_errors = None
        self.reference_match_source_stride = source_stride
        self.reference_matches = reference_matches.view(
            self.n_charts,
            self.n_charts,
            source_height,
            source_width,
        )
        if self.reference_masks is not None:
            sampled_masks = self.reference_masks[
                :, ::source_stride, ::source_stride
            ].to(dtype=torch.bool)
            self.reference_matches &= sampled_masks[None]

    def _resolve_points_and_depths(self, depths=None, points=None):
        if points is None and depths is None:
            raise ValueError("Either depths or points should be provided.")
        if points is None:
            points = depths_to_points_parallel(depths, cameras=self.cameras)
        if depths is None:
            depths = self.cameras.p3d_cameras.get_world_to_view_transform().transform_points(
                points
            )[..., 2]
        return points, depths

    def _subsample_source_points(self, points: torch.Tensor, source_stride: int):
        source_stride = max(int(source_stride), 1)
        if points.ndim == 3:
            expected_points = self.height * self.width
            if points.shape[1] != expected_points:
                raise RuntimeError(
                    "Unexpected flattened point layout: "
                    f"{tuple(points.shape)} for {self.height}x{self.width}"
                )
            points = points.reshape(self.n_charts, self.height, self.width, 3)
        elif points.ndim != 4:
            raise RuntimeError(f"Unexpected point layout: {tuple(points.shape)}")
        if source_stride > 1:
            points = points[:, ::source_stride, ::source_stride]
        return points, source_stride

    def _reference_matches_at_stride(self, source_stride: int) -> torch.Tensor:
        if not hasattr(self, "reference_matches"):
            raise RuntimeError("Call match() before evaluating the matching loss.")
        stored_stride = int(getattr(self, "reference_match_source_stride", 1))
        if source_stride < stored_stride or source_stride % stored_stride:
            raise ValueError(
                "Requested matching source stride is incompatible with the "
                f"stored reference stride ({source_stride} vs {stored_stride})."
            )
        stride_factor = source_stride // stored_stride
        return self.reference_matches[
            ..., ::stride_factor, ::stride_factor
        ]
    
    def compute_reprojection_errors(
        self, 
        depths=None,
        points=None,
        source_stride: int = 1,
    ):
        """_summary_

        Args:
            depths (_type_, optional): Shape (n_charts, height, width). Defaults to None.
            points (_type_, optional): Shape (n_charts, height, width, 3). Defaults to None.

        Raises:
            ValueError: _description_
        """
        points_to_match, depths_to_match = self._resolve_points_and_depths(
            depths=depths,
            points=points,
        )
        points_to_match, source_stride = self._subsample_source_points(
            points_to_match,
            source_stride,
        )
        source_height, source_width = points_to_match.shape[1:3]
        
        n_pts_per_chart = source_height * source_width
        all_points_to_match = points_to_match.reshape(-1, 3)  # (n_charts * n_pts_per_chart, 3)
        n_total_points = all_points_to_match.shape[0]
        chunk_size = self.projection_chunk_size or n_total_points
        depth_errors = depths_to_match.new_empty((self.n_charts, n_total_points))
        fov_mask = torch.empty((self.n_charts, n_total_points), dtype=torch.bool, device=depths_to_match.device)

        for start in range(0, n_total_points, chunk_size):
            end = min(start + chunk_size, n_total_points)
            points_to_match_chunk = all_points_to_match[start:end][None].expand(self.n_charts, -1, -1)

            # For each camera, get the depth of all points in the camera's view
            true_depth = self.cameras.p3d_cameras.get_world_to_view_transform().transform_points(points_to_match_chunk)[..., 2]

            # For each camera, get the depth of the projections of all points in the camera's depth map
            projected_depths_chunk, fov_mask_chunk = get_points_depth_in_depthmap_parallel(
                pts=points_to_match_chunk,
                depthmap=depths_to_match,
                cameras=self.cameras,
                padding_mode='zeros',
                znear=self.znear,
                chunk_size=None,
            )

            depth_errors[:, start:end] = (true_depth - projected_depths_chunk).abs().nan_to_num()
            fov_mask[:, start:end] = fov_mask_chunk

        # depth_errors[~fov_mask] = 1e8
        depth_errors = depth_errors.view(
            self.n_charts, self.n_charts, source_height, source_width
        )
        fov_mask = fov_mask.view(
            self.n_charts, self.n_charts, source_height, source_width
        )
        return depth_errors, fov_mask

    def compute_matching_loss(
        self,
        *,
        depths: torch.Tensor = None,
        points: torch.Tensor = None,
        source_stride: int = 1,
        source_masks: torch.Tensor = None,
        source_confidence: torch.Tensor = None,
        checkpoint_chunks: bool = False,
    ) -> torch.Tensor:
        """Compute the existing all-pairs matching objective without materialising it.

        ``compute_reprojection_errors`` is useful for diagnostics, but storing
        all ``(target_chart, source_chart, y, x)`` errors retains every chunk's
        autograd graph.  This reducer has the same numerator and denominator as
        the historical loss while retaining only scalar checkpoint nodes.
        """
        points_to_match, depths_to_match = self._resolve_points_and_depths(
            depths=depths,
            points=points,
        )
        points_to_match, source_stride = self._subsample_source_points(
            points_to_match,
            source_stride,
        )
        source_height, source_width = points_to_match.shape[1:3]
        reference_matches = self._reference_matches_at_stride(source_stride)
        expected_shape = (
            self.n_charts,
            self.n_charts,
            source_height,
            source_width,
        )
        if tuple(reference_matches.shape) != expected_shape:
            raise RuntimeError(
                "Reference matching layout does not match sampled source points: "
                f"{tuple(reference_matches.shape)} vs {expected_shape}"
            )

        all_points = points_to_match.reshape(-1, 3)
        n_total_points = all_points.shape[0]
        chunk_size = self.projection_chunk_size or n_total_points
        reference_matches = reference_matches.reshape(self.n_charts, n_total_points)

        source_valid = None
        if source_masks is not None:
            source_valid = source_masks[:, ::source_stride, ::source_stride]
            if tuple(source_valid.shape) != (
                self.n_charts,
                source_height,
                source_width,
            ):
                raise ValueError(
                    "source_masks must match the full Chart depth layout, got "
                    f"{tuple(source_masks.shape)}"
                )
            source_valid = source_valid.to(dtype=torch.bool).reshape(-1)
            denominator = (
                source_valid.sum().to(dtype=depths_to_match.dtype)
                * self.n_charts
            ).clamp_min(1)
        else:
            denominator = depths_to_match.new_tensor(self.n_charts * n_total_points)

        source_weights = None
        if source_confidence is not None:
            source_weights = source_confidence[
                :, ::source_stride, ::source_stride
            ]
            if tuple(source_weights.shape) != (
                self.n_charts,
                source_height,
                source_width,
            ):
                raise ValueError(
                    "source_confidence must match the full Chart depth layout, got "
                    f"{tuple(source_confidence.shape)}"
                )
            source_weights = source_weights.reshape(-1)

        numerator = depths_to_match.new_zeros(())
        for start in range(0, n_total_points, chunk_size):
            end = min(start + chunk_size, n_total_points)
            points_chunk = all_points[start:end][None].expand(self.n_charts, -1, -1)
            reference_matches_chunk = reference_matches[:, start:end]
            source_valid_chunk = (
                None if source_valid is None else source_valid[start:end]
            )
            source_weights_chunk = (
                None if source_weights is None else source_weights[start:end]
            )

            def reduce_chunk(
                current_points: torch.Tensor,
                current_depths: torch.Tensor,
                *,
                _reference_matches=reference_matches_chunk,
                _source_valid=source_valid_chunk,
                _source_weights=source_weights_chunk,
            ) -> torch.Tensor:
                true_depth = self.cameras.p3d_cameras.get_world_to_view_transform().transform_points(
                    current_points
                )[..., 2]
                projected_depths, fov_mask = get_points_depth_in_depthmap_parallel(
                    pts=current_points,
                    depthmap=current_depths,
                    cameras=self.cameras,
                    padding_mode="zeros",
                    znear=self.znear,
                    chunk_size=None,
                )
                errors = (true_depth - projected_depths).abs().nan_to_num()
                valid = fov_mask & _reference_matches
                if _source_valid is not None:
                    valid = valid & _source_valid[None]
                safe_errors = torch.where(valid, errors, torch.zeros_like(errors))
                if _source_weights is not None:
                    safe_errors = safe_errors * _source_weights[None]
                return safe_errors.sum()

            if checkpoint_chunks and (
                points_chunk.requires_grad or depths_to_match.requires_grad
            ):
                chunk_sum = checkpoint(
                    reduce_chunk,
                    points_chunk,
                    depths_to_match,
                    use_reentrant=False,
                )
            else:
                chunk_sum = reduce_chunk(points_chunk, depths_to_match)
            numerator = numerator + chunk_sum

        return numerator / denominator
