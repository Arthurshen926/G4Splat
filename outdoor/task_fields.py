"""Runtime semantic task fields for honest Cambridge residual refinement.

The available Cambridge pickle provides keep masks, not a dense semantic
segmentation.  Building and ground therefore remain explicit rigid proxies.
Trunk/branch support comes only from repeatable calibrated multiview tracks
and their local 3D line graph.  The fields are deterministic runtime tensors
consumed directly by loss and topology allocation.
"""

from __future__ import annotations

import json
import hashlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from matcha.cambridge_masks import CambridgeMaskLookup


TASK_FIELD_VERSION = (
    "cambridge_runtime_task_fields_v9_boundary_quarantined_renderer_"
    "visible_tree_occluded_rigid_geometry_"
    "and_depth_share_projected_track_owner__separate_from_exact_"
    "current_view_visibility"
)


def _sigmoid_numpy(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


class _MultiviewTreeRolePosterior:
    """Sparse immutable image-space posterior from calibrated MASt3R tracks.

    A binary tree mask says only that a ray intersects vegetation.  It cannot
    distinguish a repeatably triangulated trunk/branch or stable leaf cluster
    from sequence-local crown texture.  This lookup keeps that distinction as
    a continuous observation-space field; it never turns an unobserved pixel
    into negative tree evidence.
    """

    def __init__(self, archive_path: Path):
        from scipy.spatial import cKDTree

        self.archive_path = Path(archive_path).resolve()
        if not self.archive_path.is_file():
            raise FileNotFoundError(self.archive_path)
        digest = hashlib.sha256()
        with self.archive_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        self.archive_sha256 = digest.hexdigest()
        with np.load(self.archive_path, allow_pickle=False) as archive:
            required = {
                "xyz",
                "role_probabilities",
                "reprojection_error",
                "valid_observation_count",
                "sequence_count",
                "triangulation_angle_median",
                "cycle_consistency",
                "observation_offsets",
                "observation_camera_indices",
                "observation_pixels",
                "camera_names",
                "camera_image_sizes",
            }
            missing = sorted(required - set(archive.files))
            if missing:
                raise RuntimeError(
                    "MASt3R role posterior lacks: " + ", ".join(missing)
                )
            xyz = np.asarray(archive["xyz"], dtype=np.float64)
            role = np.asarray(
                archive["role_probabilities"], dtype=np.float64
            )
            reprojection = np.asarray(
                archive["reprojection_error"], dtype=np.float64
            )
            observations = np.asarray(
                archive["valid_observation_count"], dtype=np.float64
            )
            sequences = np.asarray(
                archive["sequence_count"], dtype=np.float64
            )
            angle = np.asarray(
                archive["triangulation_angle_median"], dtype=np.float64
            )
            cycle = np.asarray(
                archive["cycle_consistency"], dtype=np.float64
            )
            offsets = np.asarray(
                archive["observation_offsets"], dtype=np.int64
            )
            observation_camera = np.asarray(
                archive["observation_camera_indices"], dtype=np.int32
            )
            observation_pixels = np.asarray(
                archive["observation_pixels"], dtype=np.float32
            )
            camera_names = np.asarray(archive["camera_names"]).astype(str)
            camera_sizes = np.asarray(
                archive["camera_image_sizes"], dtype=np.int32
            )

        if role.ndim != 2 or role.shape[1] < 2:
            raise RuntimeError("Track role posterior has no canopy channel")
        if offsets.shape != (len(xyz) + 1,):
            raise RuntimeError("Track observation offsets do not align")
        track_for_observation = np.repeat(
            np.arange(len(xyz), dtype=np.int32), np.diff(offsets)
        )
        if len(track_for_observation) != len(observation_camera):
            raise RuntimeError("Track observation table is not ragged-aligned")

        rigid = np.clip(role[:, 0], 0.0, 1.0)
        canopy = np.clip(role[:, 1], 0.0, 1.0)
        cross_sequence = np.clip(sequences - 1.0, 0.0, 1.0)
        view_precision = 1.0 - np.exp(
            -np.maximum(observations - 1.0, 0.0) / 2.0
        )
        reprojection_precision = np.exp(
            -np.square(np.maximum(reprojection, 0.0) / 1.5)
        )
        baseline_precision = 1.0 - np.exp(
            -np.maximum(angle, 0.0) / 3.0
        )
        cycle_precision = np.clip((cycle - 0.50) / 0.35, 0.0, 1.0)
        stability = np.sqrt(
            np.clip(
                view_precision
                * reprojection_precision
                * baseline_precision
                * cycle_precision,
                0.0,
                1.0,
            )
        )
        tree_surface = (
            canopy * cross_sequence * stability
        ).astype(np.float32)
        rigid_rescue = (
            rigid * cross_sequence * stability
        ).astype(np.float32)

        # Geometry supplies a soft trunk/branch sub-role.  It is evaluated
        # only on cross-sequence canopy tracks, with no class hallucination in
        # the rest of the tree mask.  Cambridge's calibrated world frame uses
        # axis 1 as vertical, matching the existing skeleton initializer.
        skeleton = np.zeros(len(xyz), dtype=np.float32)
        vertical = np.zeros(len(xyz), dtype=np.float32)
        stable_rows = np.flatnonzero(tree_surface > 0.0)
        if len(stable_rows) >= 8:
            stable_xyz = xyz[stable_rows]
            neighbours = cKDTree(stable_xyz).query_ball_point(
                stable_xyz, r=0.36
            )
            for local_index, adjacent in enumerate(neighbours):
                if len(adjacent) < 8:
                    continue
                local = stable_xyz[np.asarray(adjacent, dtype=np.int64)]
                centered = local - local.mean(axis=0, keepdims=True)
                covariance = centered.T @ centered / max(len(local) - 1, 1)
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                eigenvalues = np.maximum(eigenvalues, 1.0e-12)
                linearity = eigenvalues[-1] / (
                    eigenvalues[-2] + eigenvalues[-3]
                )
                spread = np.sqrt(eigenvalues[-1])
                row = int(stable_rows[local_index])
                shape = _sigmoid_numpy(
                    np.asarray((linearity - 1.65) / 0.30)
                ) * _sigmoid_numpy(
                    np.asarray((spread - 0.045) / 0.012)
                )
                skeleton[row] = float(tree_surface[row] * shape)
                vertical[row] = abs(float(eigenvectors[1, -1]))
        trunk = skeleton * _sigmoid_numpy((vertical - 0.58) / 0.10)
        branch = np.clip(skeleton - trunk, 0.0, 1.0)

        order = np.argsort(observation_camera, kind="stable")
        sorted_camera = observation_camera[order]
        camera_offsets = np.searchsorted(
            sorted_camera,
            np.arange(len(camera_names) + 1, dtype=np.int32),
        )
        self.camera_names = tuple(camera_names.tolist())
        self.camera_sizes = camera_sizes
        self._camera_index = {
            Path(name).stem: index
            for index, name in enumerate(self.camera_names)
        }
        self._camera_offsets = camera_offsets
        self._pixels = observation_pixels[order]
        observation_tracks = track_for_observation[order]
        self._tree_surface = tree_surface[observation_tracks]
        self._rigid_rescue = rigid_rescue[observation_tracks]
        self._trunk = trunk[observation_tracks].astype(np.float32)
        self._branch = branch[observation_tracks].astype(np.float32)

        # Preserve an explicit sparse physical graph instead of treating
        # isolated skeleton tracks as a complete trunk/branch representation.
        # Each node keeps at most its two nearest local line neighbours.  The
        # projected segments are still clipped by the real tree mask in
        # ``OutdoorTaskFieldLookup.fields``; the graph is positive evidence
        # and can never relabel an observed facade pixel as vegetation.
        graph_edges: set[tuple[int, int]] = set()
        skeleton_rows = np.flatnonzero(skeleton > 0.05)
        if len(skeleton_rows) >= 2:
            graph_xyz = xyz[skeleton_rows]
            neighbour_count = min(3, len(graph_xyz))
            distances, neighbours = cKDTree(graph_xyz).query(
                graph_xyz, k=neighbour_count
            )
            distances = np.asarray(distances).reshape(
                len(graph_xyz), neighbour_count
            )
            neighbours = np.asarray(neighbours).reshape(
                len(graph_xyz), neighbour_count
            )
            for local_row in range(len(graph_xyz)):
                for distance, local_neighbour in zip(
                    distances[local_row, 1:],
                    neighbours[local_row, 1:],
                ):
                    if not 0.015 <= float(distance) <= 0.36:
                        continue
                    first = int(skeleton_rows[local_row])
                    second = int(skeleton_rows[int(local_neighbour)])
                    graph_edges.add(tuple(sorted((first, second))))

        edge_pixels_by_camera: list[np.ndarray] = []
        edge_trunk_by_camera: list[np.ndarray] = []
        edge_branch_by_camera: list[np.ndarray] = []
        sorted_edges = sorted(graph_edges)
        projected_edge_count = 0
        for camera_index in range(len(camera_names)):
            start = int(camera_offsets[camera_index])
            end = int(camera_offsets[camera_index + 1])
            local_tracks = observation_tracks[start:end]
            # A track has at most one calibrated observation in a camera.
            local_lookup = {
                int(track): local
                for local, track in enumerate(local_tracks.tolist())
            }
            source_width, source_height = map(
                float, camera_sizes[camera_index]
            )
            pixels = self._pixels[start:end]
            camera_edges = []
            camera_trunk = []
            camera_branch = []
            for first, second in sorted_edges:
                first_local = local_lookup.get(first)
                second_local = local_lookup.get(second)
                if first_local is None or second_local is None:
                    continue
                endpoints = np.stack(
                    [pixels[first_local], pixels[second_local]]
                ).astype(np.float32)
                endpoints /= np.asarray(
                    [source_width, source_height], dtype=np.float32
                )[None]
                camera_edges.append(endpoints)
                camera_trunk.append(float(min(trunk[first], trunk[second])))
                camera_branch.append(
                    float(min(branch[first], branch[second]))
                )
            edge_pixels_by_camera.append(
                np.asarray(camera_edges, dtype=np.float32).reshape(-1, 2, 2)
            )
            edge_trunk_by_camera.append(
                np.asarray(camera_trunk, dtype=np.float32)
            )
            edge_branch_by_camera.append(
                np.asarray(camera_branch, dtype=np.float32)
            )
            projected_edge_count += len(camera_edges)
        self._edge_pixels_by_camera = tuple(edge_pixels_by_camera)
        self._edge_trunk_by_camera = tuple(edge_trunk_by_camera)
        self._edge_branch_by_camera = tuple(edge_branch_by_camera)
        self._track_audit = {
            "track_count": int(len(xyz)),
            "canopy_track_count": int((canopy > 0.5).sum()),
            "cross_sequence_rigid_rescue_tracks": int(
                (rigid_rescue > 0).sum()
            ),
            "cross_sequence_tree_surface_tracks": int(
                (tree_surface > 0).sum()
            ),
            "geometric_skeleton_tracks": int((skeleton > 0.05).sum()),
            "skeleton_graph_edges_3d": int(len(graph_edges)),
            "skeleton_graph_projected_edges": int(projected_edge_count),
            "observation_count": int(len(observation_camera)),
            "camera_count": int(len(camera_names)),
        }

    @property
    def view_names(self) -> tuple[str, ...]:
        return self.camera_names

    def fields(
        self,
        image_name: str,
        shape: tuple[int, int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = map(int, shape)
        zero = torch.zeros((height, width), device=device)
        camera_index = self._camera_index.get(Path(str(image_name)).stem)
        if camera_index is None:
            return zero, zero, zero, zero
        start = int(self._camera_offsets[camera_index])
        end = int(self._camera_offsets[camera_index + 1])
        if end <= start:
            return zero, zero, zero, zero
        source_width, source_height = map(
            int, self.camera_sizes[camera_index]
        )
        pixels = torch.from_numpy(self._pixels[start:end]).to(device=device)
        x = torch.floor(pixels[:, 0] / source_width * width).long()
        y = torch.floor(pixels[:, 1] / source_height * height).long()
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not bool(valid.any()):
            return zero, zero, zero, zero
        flat_index = y[valid] * width + x[valid]

        def rasterize(values: np.ndarray) -> torch.Tensor:
            result = torch.zeros(height * width, device=device)
            probability = torch.from_numpy(values[start:end]).to(
                device=device
            )[valid]
            result.scatter_reduce_(
                0, flat_index, probability, reduce="amax", include_self=True
            )
            result = result.reshape(height, width)
            # Sparse calibrated tracks are positive point evidence, not an
            # exact one-pixel semantic contour. A small target-resolution
            # footprint covers interpolation/reprojection uncertainty while
            # remaining far narrower than the tree-mask boundary feather.
            return F.max_pool2d(
                result[None, None], 5, stride=1, padding=2
            )[0, 0]

        rigid_rescue_field = rasterize(self._rigid_rescue)
        tree_surface_field = rasterize(self._tree_surface)
        trunk_field = rasterize(self._trunk)
        branch_field = rasterize(self._branch)

        edge_pixels = self._edge_pixels_by_camera[camera_index]
        if len(edge_pixels):
            endpoints = torch.from_numpy(edge_pixels).to(device=device)
            endpoints[..., 0] *= width
            endpoints[..., 1] *= height

            def rasterize_edges(values: np.ndarray) -> torch.Tensor:
                samples: list[torch.Tensor] = []
                sample_values: list[torch.Tensor] = []
                probabilities = torch.from_numpy(values).to(device=device)
                for edge_index in range(len(endpoints)):
                    delta = endpoints[edge_index, 1] - endpoints[edge_index, 0]
                    steps = int(
                        torch.ceil(delta.abs().max()).clamp(2, 64).item()
                    )
                    interpolation = torch.linspace(
                        0.0, 1.0, steps, device=device
                    )[:, None]
                    samples.append(
                        endpoints[edge_index, 0][None]
                        + interpolation * delta[None]
                    )
                    sample_values.append(
                        probabilities[edge_index].expand(steps)
                    )
                sample_pixels = torch.cat(samples)
                sample_probability = torch.cat(sample_values)
                x_edge = torch.floor(sample_pixels[:, 0]).long()
                y_edge = torch.floor(sample_pixels[:, 1]).long()
                valid_edge = (
                    (x_edge >= 0)
                    & (x_edge < width)
                    & (y_edge >= 0)
                    & (y_edge < height)
                )
                result = torch.zeros(height * width, device=device)
                result.scatter_reduce_(
                    0,
                    y_edge[valid_edge] * width + x_edge[valid_edge],
                    sample_probability[valid_edge],
                    reduce="amax",
                    include_self=True,
                )
                # One-pixel line rasterization plus a one-pixel uncertainty
                # band is narrower than the 5x5 point support above.
                return F.max_pool2d(
                    result.reshape(1, 1, height, width),
                    3,
                    stride=1,
                    padding=1,
                )[0, 0]

            trunk_field = torch.maximum(
                trunk_field,
                rasterize_edges(
                    self._edge_trunk_by_camera[camera_index]
                ),
            )
            branch_field = torch.maximum(
                branch_field,
                rasterize_edges(
                    self._edge_branch_by_camera[camera_index]
                ),
            )
            tree_surface_field = torch.maximum(
                tree_surface_field, trunk_field + branch_field
            ).clamp(0.0, 1.0)

        return (
            rigid_rescue_field,
            tree_surface_field,
            trunk_field,
            branch_field,
        )

    def audit(self) -> dict:
        return {
            "enabled": True,
            "archive": str(self.archive_path),
            "archive_sha256": self.archive_sha256,
            "contract": (
                "cross_sequence_canopy_track_stability_times_reprojection_"
                "baseline_cycle_precision__rigid_track_positive_rescue__"
                "local_line_geometry_subrole__"
                "explicit_nearest_neighbour_skeleton_graph"
            ),
            **self._track_audit,
        }


class _ProjectedRigidConflictPosterior:
    """Ragged all-camera positive rigid evidence at mask conflicts."""

    def __init__(self, archive_path: Path):
        from outdoor.projected_role_posterior import (
            PROJECTED_RIGID_MAX_CAMERA_DEPTH,
            PROJECTED_RIGID_POSTERIOR_VERSION,
        )

        self.archive_path = Path(archive_path).resolve()
        if not self.archive_path.is_file():
            raise FileNotFoundError(self.archive_path)
        digest = hashlib.sha256()
        with self.archive_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        self.archive_sha256 = digest.hexdigest()
        with np.load(self.archive_path, allow_pickle=False) as archive:
            required = {
                "schema_version",
                "camera_names",
                "camera_image_sizes",
                "offsets",
                "pixels",
                "geometry_support_probability",
                "visible_observation_probability",
                "camera_depth",
            }
            missing = sorted(required - set(archive.files))
            if missing:
                raise RuntimeError(
                    "Projected rigid posterior lacks: " + ", ".join(missing)
                )
            schema = str(archive["schema_version"].item())
            if schema != PROJECTED_RIGID_POSTERIOR_VERSION:
                raise RuntimeError(
                    f"Unsupported projected rigid posterior {schema!r}"
                )
            camera_names = np.asarray(archive["camera_names"]).astype(str)
            camera_sizes = np.asarray(
                archive["camera_image_sizes"], dtype=np.int32
            )
            offsets = np.asarray(archive["offsets"], dtype=np.int64)
            pixels = np.asarray(archive["pixels"], dtype=np.float32)
            geometry = np.asarray(
                archive["geometry_support_probability"], dtype=np.float32
            )
            visible = np.asarray(
                archive["visible_observation_probability"], dtype=np.float32
            )
            depth = np.asarray(archive["camera_depth"], dtype=np.float32)
        if offsets.shape != (len(camera_names) + 1,):
            raise RuntimeError("Projected rigid posterior offsets do not align")
        if not (
            len(pixels)
            == len(geometry)
            == len(visible)
            == len(depth)
            == int(offsets[-1])
        ):
            raise RuntimeError("Projected rigid posterior rows do not align")
        if camera_sizes.shape != (len(camera_names), 2):
            raise RuntimeError("Projected rigid camera sizes do not align")
        self.camera_names = tuple(camera_names.tolist())
        self._camera_index = {
            Path(name).stem: index
            for index, name in enumerate(self.camera_names)
        }
        self._camera_sizes = camera_sizes
        self._offsets = offsets
        self._pixels = pixels
        renderer_visible_depth = (
            np.isfinite(depth)
            & (depth > 0.05)
            & (depth < float(PROJECTED_RIGID_MAX_CAMERA_DEPTH))
        )
        self._discarded_out_of_range_rows = int(
            ((geometry > 0.0) & ~renderer_visible_depth).sum()
        )
        self._discarded_out_of_range_mass = float(
            np.clip(geometry, 0.0, 1.0)[~renderer_visible_depth].sum()
        )
        self._geometry = (
            np.clip(geometry, 0.0, 1.0) * renderer_visible_depth
        )
        self._visible = np.minimum(
            np.clip(visible, 0.0, 1.0), self._geometry
        )
        self._depth = np.where(renderer_visible_depth, depth, 0.0)

    def fields(
        self,
        image_name: str,
        shape: tuple[int, int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = map(int, shape)
        zero = torch.zeros((height, width), device=device)
        camera_index = self._camera_index.get(Path(str(image_name)).stem)
        if camera_index is None:
            return zero, zero, zero
        start = int(self._offsets[camera_index])
        end = int(self._offsets[camera_index + 1])
        if end <= start:
            return zero, zero, zero
        source_width, source_height = map(
            int, self._camera_sizes[camera_index]
        )
        pixels = torch.from_numpy(self._pixels[start:end]).to(device=device)
        x = torch.floor(
            (pixels[:, 0] + 0.5) / source_width * width
        ).long()
        y = torch.floor(
            (pixels[:, 1] + 0.5) / source_height * height
        ).long()
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not bool(valid.any()):
            return zero, zero, zero
        flat_index = y[valid] * width + x[valid]

        def rasterize(values: np.ndarray) -> torch.Tensor:
            result = torch.zeros(height * width, device=device)
            probability = torch.from_numpy(values[start:end]).to(
                device=device
            )[valid]
            result.scatter_reduce_(
                0,
                flat_index,
                probability,
                reduce="amax",
                include_self=True,
            )
            # Five pixels cover point reprojection and source-raster
            # interpolation uncertainty without filling an entire mask
            # component from one isolated track.
            return F.max_pool2d(
                result.reshape(1, 1, height, width),
                5,
                stride=1,
                padding=2,
            )[0, 0]

        # Geometry confidence and target depth must come from the same track.
        # Pooling them independently allowed a high-confidence row A to grant
        # authority while the nearest inverse depth of neighbouring row B was
        # used as its target.  First choose the maximum-confidence owner in
        # each training pixel (nearest depth only breaks exact confidence
        # ties), then propagate that owner's confidence and depth through one
        # shared reprojection footprint.
        geometry_probability = torch.from_numpy(
            self._geometry[start:end]
        ).to(device=device)[valid]
        source_depth = torch.from_numpy(self._depth[start:end]).to(
            device=device
        )[valid]
        valid_depth = torch.isfinite(source_depth) & (source_depth > 0.0)
        geometry_grid = torch.zeros(height * width, device=device)
        geometry_grid.scatter_reduce_(
            0,
            flat_index,
            geometry_probability,
            reduce="amax",
            include_self=True,
        )
        winning_geometry = geometry_probability == geometry_grid[flat_index]
        inverse_depth = torch.zeros(height * width, device=device)
        depth_owner = valid_depth & winning_geometry
        if bool(depth_owner.any()):
            inverse_depth.scatter_reduce_(
                0,
                flat_index[depth_owner],
                source_depth[depth_owner].reciprocal(),
                reduce="amax",
                include_self=True,
            )
        geometry, footprint_owner = F.max_pool2d(
            geometry_grid.reshape(1, 1, height, width),
            5,
            stride=1,
            padding=2,
            return_indices=True,
        )
        geometry = geometry[0, 0]
        footprint_owner = footprint_owner[0, 0].reshape(-1)
        inverse_depth = inverse_depth[footprint_owner].reshape(height, width)
        depth = torch.where(
            inverse_depth > 0.0,
            inverse_depth.clamp_min(1.0e-8).reciprocal(),
            torch.zeros_like(inverse_depth),
        )
        visible = torch.minimum(rasterize(self._visible), geometry)
        return geometry, visible, depth

    def audit(self) -> dict:
        return {
            "enabled": True,
            "archive": str(self.archive_path),
            "archive_sha256": self.archive_sha256,
            "camera_count": len(self.camera_names),
            "projected_conflict_observation_count": int(
                self._offsets[-1]
            ),
            "geometry_support_mass": float(self._geometry.sum()),
            "visible_observation_mass": float(self._visible.sum()),
            "discarded_out_of_renderer_depth_rows": int(
                self._discarded_out_of_range_rows
            ),
            "discarded_out_of_renderer_depth_mass": float(
                self._discarded_out_of_range_mass
            ),
            "contract": (
                "all_fixed_camera_projection__geometry_support_separate_"
                "from_current_rgb_visibility__positive_evidence_only"
            ),
        }


def _soft_boundary(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return torch.zeros_like(mask, dtype=torch.float32)
    value = mask.to(dtype=torch.float32)
    kernel = 2 * int(radius) + 1
    extrema = F.max_pool2d(
        torch.stack([value, -value])[:, None],
        kernel_size=kernel,
        stride=1,
        padding=radius,
    )
    return (extrema[0, 0] + extrema[1, 0]).clamp(0.0, 1.0)


def _scale_aware_radius(mask: torch.Tensor, base_radius: int) -> int:
    """Resolution/foreground-scale-aware fallback without invented depth."""
    if base_radius <= 0 or not bool(mask.any()):
        return 0
    height, width = mask.shape
    equivalent_radius = torch.sqrt(
        mask.float().sum() / torch.pi
    ).item()
    resolution_scale = ((height * width) / (540.0 * 960.0)) ** 0.25
    component_scale = max(0.5, min(2.0, (equivalent_radius / 120.0) ** 0.5))
    return int(round(max(1.0, min(8.0, base_radius * resolution_scale * component_scale))))


class OutdoorTaskFieldLookup:
    """Materialize per-pixel task probabilities from the four real mask channels."""

    def __init__(
        self,
        dataset_path: Path,
        tree_mask_pickle: Path,
        semantic_manifest: Path,
        *,
        multiview_track_archive: Path | None = None,
        projected_rigid_posterior_archive: Path | None = None,
        boundary_radius: int = 3,
        max_cached_views: int = 16,
    ):
        self.dataset_path = Path(dataset_path).resolve()
        self.tree_mask_pickle = Path(tree_mask_pickle).resolve()
        self.semantic_manifest_path = Path(semantic_manifest).resolve()
        self.boundary_radius = int(boundary_radius)
        self.max_cached_views = int(max_cached_views)
        if self.boundary_radius < 0:
            raise ValueError("boundary_radius must be non-negative")
        if self.max_cached_views < 0:
            raise ValueError("max_cached_views must be non-negative")
        if not self.semantic_manifest_path.is_file():
            raise FileNotFoundError(self.semantic_manifest_path)
        self.semantic_manifest = json.loads(
            self.semantic_manifest_path.read_text(encoding="utf-8")
        )
        availability = self.semantic_manifest.get("class_availability", {})
        if availability.get("canopy") != "tree_mask_index_3":
            raise RuntimeError(
                "Residual task fields require a real tree mask at channel index 3"
            )
        expected_tree_hash = self.semantic_manifest.get(
            "input_hashes", {}
        ).get("tree_mask_pickle")
        if not expected_tree_hash:
            raise RuntimeError(
                "Semantic manifest does not bind its tree mask content hash"
            )
        digest = hashlib.sha256()
        with self.tree_mask_pickle.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        self.tree_mask_sha256 = digest.hexdigest()
        if self.tree_mask_sha256 != expected_tree_hash:
            raise RuntimeError(
                "Runtime tree mask differs from the immutable semantic "
                f"contract: expected={expected_tree_hash}, "
                f"actual={self.tree_mask_sha256}, "
                f"path={self.tree_mask_pickle}"
            )
        self.lookup = CambridgeMaskLookup(
            self.dataset_path,
            self.tree_mask_pickle,
            mask_indices=[0, 1, 2, 3],
        )
        self.multiview_roles = (
            _MultiviewTreeRolePosterior(multiview_track_archive)
            if multiview_track_archive is not None
            else None
        )
        self.projected_rigid_roles = (
            _ProjectedRigidConflictPosterior(
                projected_rigid_posterior_archive
            )
            if projected_rigid_posterior_archive is not None
            else None
        )
        self._mask_cache: OrderedDict[
            tuple[str, int, int, str], tuple[torch.Tensor, ...]
        ] = OrderedDict()
        # The adaptive radius requires a scalar GPU reduction.  Cache the two
        # deterministic integers per camera/resolution so subsequent epochs
        # do not introduce four device synchronizations per training step.
        self._radius_cache: dict[
            tuple[str, int, int], tuple[int, int]
        ] = {}
        self._canopy_fraction_cache: dict[str, float] = {}

    def _keep_masks(
        self,
        image_name: str,
        shape: tuple[int, int],
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        key = (str(image_name), int(shape[0]), int(shape[1]), str(torch.device(device)))
        if key in self._mask_cache:
            masks = self._mask_cache.pop(key)
            self._mask_cache[key] = masks
            return masks
        stacked = self.lookup.get_index_masks(
            image_name,
            (0, 1, 2, 3),
            shape,
            device,
        )
        masks = tuple(stacked[index] for index in range(4))
        if self.max_cached_views:
            self._mask_cache[key] = masks
            while len(self._mask_cache) > self.max_cached_views:
                self._mask_cache.popitem(last=False)
        return masks

    def fields(
        self,
        image_name: str,
        shape: tuple[int, int],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        object_keep, sky_keep, distortion_keep, tree_keep = self._keep_masks(
            image_name, shape, device
        )
        p_mask_transient = (~object_keep).float()
        p_mask_sky = (~sky_keep).float() * (1.0 - p_mask_transient)
        p_mask_canopy = (
            (~tree_keep).float()
            * (1.0 - p_mask_transient)
            * (1.0 - p_mask_sky)
        )
        p_semantic_rigid = (
            object_keep.float()
            * sky_keep.float()
            * distortion_keep.float()
            * tree_keep.float()
        )
        uncertain_union = (
            p_mask_transient + p_mask_sky + p_mask_canopy
        ).clamp(0.0, 1.0)
        radius_key = (str(image_name), int(shape[0]), int(shape[1]))
        cached_radii = self._radius_cache.get(radius_key)
        if cached_radii is None:
            adaptive_radius = _scale_aware_radius(
                uncertain_union > 0.5, self.boundary_radius
            )
            canopy_radius = _scale_aware_radius(
                p_mask_canopy > 0.5, self.boundary_radius
            )
            cached_radii = (adaptive_radius, canopy_radius)
            self._radius_cache[radius_key] = cached_radii
        else:
            adaptive_radius, canopy_radius = cached_radii
        p_boundary = _soft_boundary(uncertain_union > 0.5, adaptive_radius)
        p_canopy_boundary = _soft_boundary(
            p_mask_canopy > 0.5, canopy_radius
        ) * p_mask_canopy
        if self.multiview_roles is None:
            p_direct_rigid_rescue = torch.zeros_like(p_mask_canopy)
            p_tree_surface = torch.zeros_like(p_mask_canopy)
            p_trunk = torch.zeros_like(p_mask_canopy)
            p_branch = torch.zeros_like(p_mask_canopy)
        else:
            (
                p_direct_rigid_rescue,
                p_tree_surface,
                p_trunk,
                p_branch,
            ) = (
                self.multiview_roles.fields(image_name, shape, device)
            )
        # Exact track observations are positive evidence and may overturn
        # any of the three noisy semantic priors. They can never reduce an
        # ordinary rigid pixel's ownership.
        p_direct_rigid_rescue = p_direct_rigid_rescue * (
            1.0 - p_semantic_rigid
        )
        if self.projected_rigid_roles is None:
            p_projected_geometry = torch.zeros_like(p_mask_canopy)
            p_projected_visible = torch.zeros_like(p_mask_canopy)
            projected_rigid_depth = torch.zeros_like(p_mask_canopy)
        else:
            (
                p_projected_geometry,
                p_projected_visible,
                projected_rigid_depth,
            ) = self.projected_rigid_roles.fields(
                image_name, shape, device
            )
            projected_conflict = (
                p_mask_transient + p_mask_sky + p_mask_canopy
            ).clamp(0.0, 1.0)
            p_projected_geometry = (
                p_projected_geometry * projected_conflict
            )
            p_projected_visible = torch.minimum(
                p_projected_visible * projected_conflict,
                p_projected_geometry,
            )
            # A projected row with no current-view appearance witness is an
            # occluded existence/depth factor, not permission to enlarge a
            # surfel across a semantic occlusion edge. Retain it in the
            # uncertain-region interior and attenuate it continuously to zero
            # across the measured boundary band. Current-view-visible support
            # remains legal at the boundary because its RGB/depth ownership is
            # directly observed in this camera.
            projected_hidden_geometry = (
                p_projected_geometry - p_projected_visible
            ).clamp(0.0, 1.0)
            p_projected_geometry = (
                p_projected_visible
                + projected_hidden_geometry * (1.0 - p_boundary)
            ).clamp(0.0, 1.0)
            projected_rigid_depth = torch.where(
                p_projected_geometry > 0.0,
                projected_rigid_depth,
                torch.zeros_like(projected_rigid_depth),
            )
        # Probabilistic OR retains continuous authority. Geometry behind an
        # occluder is reported separately and does not restore current-view
        # RGB supervision unless its appearance is also consistent.
        p_rigid_rescue = 1.0 - (
            (1.0 - p_direct_rigid_rescue)
            * (1.0 - p_projected_visible)
        )
        p_rigid = torch.maximum(p_semantic_rigid, p_rigid_rescue)
        p_transient = p_mask_transient * (1.0 - p_rigid_rescue)
        p_sky = p_mask_sky * (1.0 - p_rigid_rescue)
        p_canopy = p_mask_canopy * (1.0 - p_rigid_rescue)
        p_canopy_boundary = p_canopy_boundary * (1.0 - p_rigid_rescue)
        p_canopy_core = (p_canopy - p_canopy_boundary).clamp_min(0.0)
        if self.multiview_roles is not None:
            p_tree_surface = p_tree_surface * p_canopy
            p_trunk = torch.minimum(p_trunk * p_canopy, p_tree_surface)
            p_branch = torch.minimum(
                p_branch * p_canopy,
                (p_tree_surface - p_trunk).clamp_min(0.0),
            )
        p_crown = p_canopy * (1.0 - p_tree_surface)
        boundary_unknown = (
            p_boundary
            * (1.0 - p_rigid_rescue)
            * (1.0 - p_tree_surface)
        ).clamp(0.0, 1.0)
        occluded_rigid_unknown = (
            p_projected_geometry - p_projected_visible
        ).clamp(0.0, 1.0)
        p_unknown_ownership = torch.maximum(
            boundary_unknown, occluded_rigid_unknown
        )
        distortion = distortion_keep.float()

        # Actual loss/task weights.  Building and ground are intentionally
        # aliases of the only available rigid proxy, not invented labels.
        w_gaussian_rgb = (
            distortion
            * (1.20 * p_rigid + 0.70 * p_canopy)
            * (1.0 - p_sky)
            * (1.0 - 0.8 * p_transient)
            * (1.0 - 0.35 * p_boundary)
        )
        w_sky_rgb = (
            distortion
            * p_sky
            * (1.0 - 0.8 * p_transient)
            * (1.0 - 0.35 * p_boundary)
        )
        w_topology = (
            distortion
            * p_canopy
            * (1.0 - p_transient)
            * (1.0 - p_sky)
            * (1.0 - p_boundary)
        )
        w_skeleton = (
            distortion
            * p_tree_surface
            * (1.0 - p_transient)
            * (1.0 - p_sky)
            * (1.0 - 0.5 * p_boundary)
        )
        w_geometry = distortion * (p_rigid + 0.1 * p_canopy)
        w_plane = distortion * p_rigid * (1.0 - p_boundary)

        return {
            "p_building": p_rigid,
            "p_ground": p_rigid,
            "p_rigid": p_rigid,
            "p_multiview_rigid_rescue": p_rigid_rescue,
            "p_direct_rigid_rescue": p_direct_rigid_rescue,
            "p_projected_rigid_geometry": p_projected_geometry,
            "p_projected_rigid_visible": p_projected_visible,
            "projected_rigid_depth": projected_rigid_depth,
            "p_tree_surface": p_tree_surface,
            "p_trunk": p_trunk,
            "p_branch": p_branch,
            "p_canopy": p_canopy,
            "p_crown": p_crown,
            "p_canopy_core": p_canopy_core,
            "p_canopy_boundary": p_canopy_boundary,
            "p_sky": p_sky,
            "p_transient": p_transient,
            "p_mask_sky_prior": p_mask_sky,
            "p_mask_transient_prior": p_mask_transient,
            "p_mask_canopy_prior": p_mask_canopy,
            "p_distortion_valid": distortion,
            "p_boundary_uncertain": p_boundary,
            "p_unknown_ownership": p_unknown_ownership,
            "w_rgb": distortion * (1.0 - 0.8 * p_transient),
            "w_gaussian_rgb": w_gaussian_rgb,
            "w_sky_rgb": w_sky_rgb,
            "w_geometry": w_geometry,
            "w_plane": w_plane,
            "w_matching": w_plane,
            "w_topology": w_topology,
            "w_skeleton": w_skeleton,
        }

    def canopy_fraction(
        self,
        image_name: str,
        shape: tuple[int, int],
    ) -> float:
        """Fast source-grid CPU ranking signal.

        Ranking does not need a training-resolution tensor.  Avoid stacking
        and resizing all four channels for every camera at cold start; doing
        that for the 1,487-view Cambridge set used several CPU-minutes and
        retained gigabytes of compact-mask cache entries before iteration one.
        A bounded regular sample of the source grid is sufficient for this
        view-priority statistic.
        """
        source_name = self.lookup.source_name_for(image_name)
        cached = self._canopy_fraction_cache.get(source_name)
        if cached is not None:
            return cached
        mask_tuple = self.lookup.masks[source_name]
        source_height, source_width = mask_tuple[0].shape
        target_height = max(int(shape[0]), 1)
        target_width = max(int(shape[1]), 1)
        row_stride = max(
            (source_height + target_height - 1) // target_height, 1
        )
        column_stride = max(
            (source_width + target_width - 1) // target_width, 1
        )
        sample = (
            slice(None, None, row_stride),
            slice(None, None, column_stride),
        )
        object_keep = mask_tuple[0][sample].to(
            device="cpu", dtype=torch.bool
        )
        sky_keep = mask_tuple[1][sample].to(
            device="cpu", dtype=torch.bool
        )
        tree_keep = mask_tuple[3][sample].to(
            device="cpu", dtype=torch.bool
        )
        if not (
            object_keep.shape == sky_keep.shape == tree_keep.shape
        ):
            raise RuntimeError(
                f"Semantic mask shapes differ for {source_name}"
            )
        canopy = (~tree_keep) & object_keep & sky_keep
        fraction = float(
            canopy.sum(dtype=torch.float32).item() / canopy.numel()
        )
        self._canopy_fraction_cache[source_name] = fraction
        return fraction

    def audit(self) -> dict:
        return {
            "version": TASK_FIELD_VERSION,
            "materialization": "deterministic_runtime_per_sample",
            "source_mask_pickle": str(self.tree_mask_pickle),
            "source_mask_pickle_sha256": self.tree_mask_sha256,
            "semantic_manifest": str(self.semantic_manifest_path),
            "mask_channel_contract": {
                "transient": "inverse_keep_channel_0",
                "sky": "inverse_keep_channel_1",
                "distortion_valid": "keep_channel_2",
                "canopy": "inverse_keep_channel_3",
                "tree_surface": (
                    "cross_sequence_multiview_track_posterior"
                    if self.multiview_roles is not None
                    else "unavailable_zero"
                ),
                "rigid_rescue": (
                    "exact_track_observation_or_all_camera_projected_"
                    "visible_rigid_positive_evidence"
                    if (
                        self.multiview_roles is not None
                        or self.projected_rigid_roles is not None
                    )
                    else "unavailable_zero"
                ),
                "trunk": (
                    "multiview_local_line_subrole"
                    if self.multiview_roles is not None
                    else "unavailable_zero"
                ),
                "branch": (
                    "multiview_local_line_subrole"
                    if self.multiview_roles is not None
                    else "unavailable_zero"
                ),
                "building": "rigid_proxy_not_semantic_segmentation",
                "ground": "rigid_proxy_not_semantic_segmentation",
            },
            "boundary_policy": {
                "type": "resolution_and_foreground_scale_aware_global_fallback",
                "base_radius_pixels_at_540x960": self.boundary_radius,
                "range_pixels": [1, 8],
                "limitation": (
                    "independent component depth is unavailable; no metric-depth "
                    "boundary scale is invented"
                ),
            },
            "max_cached_views": self.max_cached_views,
            "multiview_tree_role_posterior": (
                self.multiview_roles.audit()
                if self.multiview_roles is not None
                else {"enabled": False}
            ),
            "all_camera_projected_rigid_posterior": (
                self.projected_rigid_roles.audit()
                if self.projected_rigid_roles is not None
                else {"enabled": False}
            ),
            "ownership_arbitration": (
                "semantic_masks_are_priors__exact_multiview_roles_and_"
                "appearance_visible_all_camera_projection_are_positive_"
                "rigid_evidence__occluded_geometry_becomes_unknown_not_rgb"
            ),
        }
