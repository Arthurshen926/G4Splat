"""Cross-view accumulator for uncovered foliage hit-ray births."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

import numpy as np
import torch


@dataclass
class _Cell:
    weighted_center: torch.Tensor
    weight: float
    weighted_color: torch.Tensor
    color_weight: float
    cameras: set[int] = field(default_factory=set)
    sequences: set[str] = field(default_factory=set)
    depth_constraints: dict[int, tuple[torch.Tensor, float, float]] = field(default_factory=dict)
    camera_sequences: dict[int, str] = field(default_factory=dict)
    first_hit_witnesses: dict[int, tuple[int, int, int, int]] = field(default_factory=dict)


class StaticRayBirthAccumulator:
    """Promote uncovered hits only after independent static confirmation."""

    def __init__(
        self,
        *,
        voxel_size: float = 0.15,
        visual_hull_voxel_size: float = 0.30,
        maximum_segment_samples: int = 16,
        maximum_proposals_per_update: int = 256,
        minimum_birth_separation: float = 0.0,
    ):
        self.voxel_size = float(voxel_size)
        if self.voxel_size <= 0:
            raise ValueError("static birth voxel size must be positive")
        self.visual_hull_voxel_size = float(visual_hull_voxel_size)
        if self.visual_hull_voxel_size < self.voxel_size:
            raise ValueError(
                "static visual-hull voxel size must be at least the "
                "midpoint voxel size"
            )
        self.maximum_segment_samples = int(maximum_segment_samples)
        if self.maximum_segment_samples < 2:
            raise ValueError(
                "static visual-hull intervals require at least two samples"
            )
        self.maximum_proposals_per_update = int(maximum_proposals_per_update)
        self.minimum_birth_separation = float(minimum_birth_separation)
        if not math.isfinite(self.minimum_birth_separation) or self.minimum_birth_separation < 0:
            raise ValueError("minimum birth separation must be finite and nonnegative")
        self.emitted_centers: list[tuple[float, float, float]] = []
        self._center_index: dict[tuple[int, int, int], list[tuple[float, float, float]]] = {}
        if self.maximum_proposals_per_update <= 0:
            raise ValueError("maximum proposals per update must be positive")
        self.cells: dict[tuple[int, int, int], _Cell] = {}
        self.visual_hull_cells: dict[tuple[int, int, int], _Cell] = {}
        # A drained consensus cell represents one persistent occupancy
        # hypothesis.  Deleting it without a tombstone allowed later cameras to
        # rebuild the same voxel and append another alpha=.04 Gaussian every
        # topology event.  Persist both grids independently because their cell
        # sizes and evidence meanings differ.
        self.consumed_midpoint_cells: set[tuple[int, int, int]] = set()
        self.consumed_visual_hull_cells: set[tuple[int, int, int]] = set()
        self.suppressed_consumed_cells = 0
        self.total_proposals = 0
        self.segment_proposals = 0
        self.total_births = 0
        self.consumed_first_hit_witnesses: set[tuple[int, int, int, int]] = set()
        self.suppressed_first_hit_rays = 0

    @staticmethod
    def _first_hit_key(camera_id, direction):
        return (int(camera_id), *(int(round(float(v)*1e5)) for v in direction))

    def _available_support(self, cell):
        cameras = cell.cameras - {camera for camera,token in cell.first_hit_witnesses.items()
                                 if token in self.consumed_first_hit_witnesses}
        if cameras == cell.cameras:
            return cameras, cell.sequences
        if all(camera in cell.camera_sequences for camera in cameras):
            return cameras, {cell.camera_sequences[camera] for camera in cameras}
        return cameras, set()

    def _register_center(self, center) -> None:
        if self.minimum_birth_separation <= 0:
            return
        point = tuple(float(v) for v in center)
        key = tuple(math.floor(v / self.minimum_birth_separation) for v in point)
        self.emitted_centers.append(point)
        self._center_index.setdefault(key, []).append(point)

    def _near_emitted_center(self, center) -> bool:
        radius = self.minimum_birth_separation
        if radius <= 0:
            return False
        point = tuple(float(v) for v in center)
        key = tuple(math.floor(v / radius) for v in point)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for other in self._center_index.get((key[0]+dx, key[1]+dy, key[2]+dz), ()):
                        if sum((a-b)**2 for a,b in zip(point,other)) < radius**2:
                            return True
        return False

    @staticmethod
    def _accumulate(
        cells: dict[tuple[int, int, int], _Cell],
        *,
        key: tuple[int, int, int],
        center: torch.Tensor,
        color: torch.Tensor | None,
        weight: float,
        camera_id: int,
        sequence_id: str,
        depth_constraint: tuple[torch.Tensor, float, float] | None = None,
        first_hit_witness: tuple[int, int, int, int] | None = None,
    ) -> None:
        cell = cells.get(key)
        # A repeated training visit (or neighbouring ray in the same image)
        # is not an independent observation. Without this guard, one camera
        # could dominate the centre, colour and drain priority indefinitely
        # while the camera-count gate misleadingly still looked correct.
        if cell is not None and int(camera_id) in cell.cameras:
            # Legacy pending cells can acquire their missing geometric
            # witness without counting the camera's weight a second time.
            if depth_constraint is not None:
                cell.depth_constraints.setdefault(int(camera_id), depth_constraint)
            return
        if cell is None:
            cell = _Cell(
                weighted_center=center.clone() * weight,
                weight=weight,
                weighted_color=(
                    torch.zeros(3)
                    if color is None
                    else color.clone() * weight
                ),
                color_weight=0.0 if color is None else weight,
            )
            cells[key] = cell
        else:
            cell.weighted_center += center * weight
            cell.weight += weight
            if color is not None:
                # Robust cross-view colour: a single specular or blurred
                # observation cannot turn a newborn into a bright dot.
                if cell.color_weight <= 0:
                    robust_weight = weight
                else:
                    mean = cell.weighted_color / cell.color_weight
                    residual = float((color - mean).norm())
                    robust_weight = weight * min(
                        1.0, 0.25 / max(residual, 1e-8)
                    )
                cell.weighted_color += color * robust_weight
                cell.color_weight += robust_weight
        cell.cameras.add(int(camera_id))
        cell.sequences.add(str(sequence_id))
        cell.camera_sequences[int(camera_id)] = str(sequence_id)
        if first_hit_witness is not None:
            cell.first_hit_witnesses[int(camera_id)] = first_hit_witness
        if depth_constraint is not None:
            cell.depth_constraints[int(camera_id)] = depth_constraint

    def add(self, proposals: dict | None, *, sequence_id: str) -> int:
        if not proposals:
            return 0
        centers = torch.as_tensor(proposals["centers"]).float().cpu()
        confidence = torch.as_tensor(proposals["confidence"]).float().cpu()
        colors_value = proposals.get("colors")
        colors = (
            None
            if colors_value is None
            else torch.as_tensor(colors_value).float().cpu()
        )
        camera_id = int(proposals["camera_id"])
        segment_names = ("origins", "directions", "hit_start", "hit_end")
        segment_values = [proposals.get(name) for name in segment_names]
        has_segments = all(value is not None for value in segment_values)
        single_first_hit = bool(proposals.get("single_first_hit", False))
        if single_first_hit and not has_segments:
            raise ValueError("Single first-hit evidence requires calibrated ray intervals")
        if any(value is not None for value in segment_values) and not has_segments:
            raise ValueError(
                "static birth interval proposals must provide origins, "
                "directions, hit_start and hit_end together"
            )
        origins = directions = hit_start = hit_end = None
        if has_segments:
            origins = torch.as_tensor(proposals["origins"]).float().cpu()
            directions = torch.as_tensor(
                proposals["directions"]
            ).float().cpu()
            hit_start = torch.as_tensor(
                proposals["hit_start"]
            ).float().cpu()
            hit_end = torch.as_tensor(proposals["hit_end"]).float().cpu()
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError("static birth proposals must have shape [N,3]")
        if confidence.shape != (len(centers),):
            raise ValueError("static birth confidence must match centers")
        if colors is not None and colors.shape != (len(centers), 3):
            raise ValueError("static birth colors must have shape [N,3]")
        if has_segments:
            if origins.shape != centers.shape or directions.shape != centers.shape:
                raise ValueError(
                    "static birth interval origins/directions must have "
                    "shape [N,3]"
                )
            if hit_start.shape != (len(centers),) or hit_end.shape != (
                len(centers),
            ):
                raise ValueError(
                    "static birth interval endpoints must have shape [N]"
                )
            if not bool(torch.isfinite(origins).all()) or not bool(
                torch.isfinite(directions).all()
            ):
                raise ValueError("static birth intervals must be finite")
            if (not bool(torch.isfinite(hit_start).all())
                    or not bool(torch.isfinite(hit_end).all())
                    or bool((hit_start <= 0).any())
                    or bool((directions.norm(dim=1) <= 1e-8).any())):
                raise ValueError("static birth intervals require finite positive depths and nonzero rays")
            if bool((hit_end < hit_start).any()):
                raise ValueError("static birth interval end precedes start")
            directions = directions / directions.norm(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
        # One difficult camera can expose thousands of empty rays.  Keeping
        # every proposal makes the Python-side consensus map dominate the
        # trainer without adding independent evidence.  Retain the strongest
        # bounded subset; a birth still requires other cameras/sequences.
        if len(centers) > self.maximum_proposals_per_update:
            selected = torch.topk(
                confidence,
                k=self.maximum_proposals_per_update,
                largest=True,
                sorted=False,
            ).indices
            centers = centers[selected]
            confidence = confidence[selected]
            if colors is not None:
                colors = colors[selected]
            if has_segments:
                origins = origins[selected]
                directions = directions[selected]
                hit_start = hit_start[selected]
                hit_end = hit_end[selected]
        color_rows = (
            [None] * len(centers)
            if colors is None
            else colors.clamp(0.0, 1.0)
        )
        for row, (center, color, value) in enumerate(
            zip(centers, color_rows, confidence.clamp(0.05, 1.0))
        ):
            weight = float(value)
            first_hit_witness = self._first_hit_key(camera_id,directions[row]) if single_first_hit else None
            if first_hit_witness in self.consumed_first_hit_witnesses:
                self.suppressed_first_hit_rays += 1
                continue
            if first_hit_witness is not None and self._near_emitted_center(center):
                # A new camera's posterior centre agrees with an existing
                # local hit. Consume it there before occupied-cell skipping
                # could redirect it to the far end of its uncertainty slab.
                self.consumed_first_hit_witnesses.add(first_hit_witness)
                self.suppressed_first_hit_rays += 1
                continue
            if not has_segments:
                key_tensor = torch.floor(
                    center / self.voxel_size
                ).to(torch.int64)
                key = tuple(int(part) for part in key_tensor.tolist())
                if key in self.consumed_midpoint_cells:
                    self.suppressed_consumed_cells += 1
                    continue
                self._accumulate(
                    self.cells,
                    key=key,
                    center=center,
                    color=color,
                    weight=weight,
                    camera_id=camera_id,
                    sequence_id=sequence_id,
                )
                continue

            # A depth posterior is an interval on a calibrated ray, not a
            # single midpoint.  Midpoints from two traversals of a moving
            # crown can differ by more than 15 cm even when their intervals
            # intersect the same persistent volume.  Rasterize a bounded
            # number of samples through a coarser visual-hull grid.  Each
            # camera contributes at most once to a cell, while promotion
            # still requires independent cameras and sequences in drain().
            start = float(hit_start[row])
            end = float(hit_end[row])
            interval_length = max(end - start, 0.0)
            sample_count = min(
                self.maximum_segment_samples,
                max(
                    2,
                    int(
                        math.ceil(
                            interval_length
                            / (0.5 * self.visual_hull_voxel_size)
                        )
                    )
                    + 1,
                ),
            )
            depth = torch.linspace(start, end, sample_count)
            points = origins[row][None] + directions[row][None] * depth[:, None]
            keys = torch.floor(
                points / self.visual_hull_voxel_size
            ).to(torch.int64)
            unique_keys = torch.unique(keys, dim=0)
            for key_tensor in unique_keys:
                key = tuple(int(part) for part in key_tensor.tolist())
                if key in self.consumed_visual_hull_cells:
                    self.suppressed_consumed_cells += 1
                    continue
                member = (keys == key_tensor[None]).all(dim=1)
                representative = points[member].mean(dim=0)
                self._accumulate(
                    self.visual_hull_cells,
                    key=key,
                    center=representative,
                    color=color,
                    weight=weight,
                    camera_id=camera_id,
                    sequence_id=sequence_id,
                    depth_constraint=(
                        directions[row].clone(),
                        start + float(origins[row].dot(directions[row])),
                        end + float(origins[row].dot(directions[row])),
                    ),
                    first_hit_witness=first_hit_witness,
                )
        self.total_proposals += len(centers)
        if has_segments:
            self.segment_proposals += len(centers)
        return len(centers)

    def drain(
        self,
        *,
        maximum_births: int,
        minimum_cameras: int = 2,
        minimum_sequences: int = 2,
        progress_callback=None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, int | float | str],
    ]:
        maximum_births = max(int(maximum_births), 0)
        started=time.perf_counter()
        def progress(stage,**values):
            if progress_callback is not None:
                progress_callback(dict(stage=stage,elapsed_sec=time.perf_counter()-started,**values))
        progress('filtering',pending_cells=len(self.cells)+len(self.visual_hull_cells),maximum_births=maximum_births)
        if maximum_births==0:
            # No selection or tombstone mutation is possible at zero capacity.
            # Do not solve/sort millions of cells just to break before row0.
            audit=dict(contract='uncovered_hit_ray_voxel_consensus_across_independent_cameras_and_configured_sequence_support',
                minimum_cameras=int(minimum_cameras),minimum_sequences=int(minimum_sequences),
                pending_cells=len(self.cells)+len(self.visual_hull_cells),pending_midpoint_cells=len(self.cells),
                pending_visual_hull_cells=len(self.visual_hull_cells),eligible_cells=0,eligible_midpoint_cells=0,
                eligible_visual_hull_cells=0,eligibility_evaluated=False,skipped_reason='zero_birth_capacity',
                born=0,visual_hull_born=0,depth_consensus_rejected_cells=0,duplicate_center_cells=0,
                minimum_birth_separation=self.minimum_birth_separation,spent_first_hit_witness_cells=0,
                spent_witness_color_unknown_births=0,consumed_first_hit_witnesses=len(self.consumed_first_hit_witnesses),
                suppressed_first_hit_rays=self.suppressed_first_hit_rays,total_proposals=self.total_proposals,
                segment_proposals=self.segment_proposals,total_births=self.total_births,voxel_size=self.voxel_size,
                visual_hull_voxel_size=self.visual_hull_voxel_size,consumed_midpoint_cells=len(self.consumed_midpoint_cells),
                consumed_visual_hull_cells=len(self.consumed_visual_hull_cells),suppressed_consumed_cells=self.suppressed_consumed_cells)
            progress('skipped_zero_capacity')
            return torch.empty(0,3),torch.empty(0,3),torch.empty(0,0,dtype=torch.int32),torch.empty(0,dtype=torch.int16),audit
        midpoint_eligible = [
            ("midpoint", key, cell)
            for key, cell in self.cells.items()
            if len(cell.cameras) >= int(minimum_cameras)
            and len(cell.sequences) >= int(minimum_sequences)
        ]
        visual_hull_eligible = [
            ("visual_hull", key, cell)
            for key, cell in self.visual_hull_cells.items()
            if len(cell.cameras) >= int(minimum_cameras)
            and len(cell.sequences) >= int(minimum_sequences)
        ]
        eligible = midpoint_eligible + visual_hull_eligible
        unspent = []
        spent_before_solve = 0
        for item in eligible:
            cameras,sequences = self._available_support(item[2])
            if len(cameras) >= minimum_cameras and len(sequences) >= minimum_sequences:
                unspent.append(item)
            else:
                spent_before_solve += 1
        eligible = unspent
        eligible.sort(
            key=lambda item: (
                -len(item[2].sequences),
                -len(item[2].cameras),
                -item[2].weight,
                item[0],
                item[1],
            )
        )
        # Voxel co-membership is only a broad-phase lookup, not proof of
        # overlapping depth intervals. Resolve centres inside all recorded
        # depth slabs and the original voxel before giving them support IDs.
        hull = [item for item in eligible if item[0] == "visual_hull"]
        solved_support = {key:self._available_support(cell)[0] for _,key,cell in hull}
        progress('depth_solve',candidate_cells=len(hull))
        resolved_centers, valid_hull = self._resolve_depth_centers(hull)
        hull_centers = {
            key: resolved_centers[i]
            for i, (_, key, _) in enumerate(hull) if bool(valid_hull[i])
        }
        rejected_depth = len(hull) - len(hull_centers)
        progress('priority',feasible_hull_cells=len(hull_centers),rejected_hull_cells=rejected_depth)
        eligible = [item for item in eligible
                    if item[0] == "midpoint" or item[1] in hull_centers]
        # An interval describes uncertainty about one hit, not evidence for
        # filling every crossed voxel. Prefer the common depth posterior's
        # centre before spending that ray's one-hit witness budget.
        def priority(item):
            source,key,cell = item
            residual = 0.0
            if cell.first_hit_witnesses and source == "visual_hull":
                point = hull_centers[key]
                residual = sum(((float(point.dot(normal))-.5*(lower+upper))
                    / max(.5*(upper-lower),1e-5))**2
                    for camera,(normal,lower,upper) in cell.depth_constraints.items()
                    if camera in solved_support[key]) / max(len(solved_support[key]),1)
            cameras,sequences = self._available_support(cell)
            return (-len(sequences),-len(cameras),residual,-cell.weight,source,key)
        eligible.sort(key=priority)
        selected = []
        selected_support = {}
        duplicate_cells = []
        spent_witness_cells = spent_before_solve
        progress('selection',eligible_cells=len(eligible))
        last_progress=time.perf_counter()
        for scanned,item in enumerate(eligible):
            if scanned%4096==0 and time.perf_counter()-last_progress>=10:
                progress('selection',scanned_cells=scanned,eligible_cells=len(eligible),selected_cells=len(selected))
                last_progress=time.perf_counter()
            if len(selected) >= maximum_births:
                break
            source, key, cell = item
            cameras,sequences = self._available_support(cell)
            if len(cameras) < minimum_cameras or len(sequences) < minimum_sequences:
                spent_witness_cells += 1
                continue
            # Earlier selections in this same drain can spend another one of
            # this cell's witnesses. Re-solve before using its stale center.
            if source == "visual_hull" and cameras != solved_support[key]:
                repaired,valid = self._resolve_depth_centers([item])
                if not bool(valid[0]):
                    rejected_depth += 1
                    continue
                hull_centers[key] = repaired[0]
            center = (hull_centers[key] if source == "visual_hull"
                      else cell.weighted_center / max(cell.weight, 1e-8))
            if self._near_emitted_center(center):
                duplicate_cells.append(item)
                # Associate a new first-hit witness to the existing local
                # point; do not let it fall through to a worse depth cell
                # and manufacture a second point along the same interval.
                self.consumed_first_hit_witnesses.update(cell.first_hit_witnesses.values())
                continue
            selected.append(item)
            selected_support[(source,key)] = (cameras,sequences)
            self._register_center(center)
            self.consumed_first_hit_witnesses.update(cell.first_hit_witnesses.values())
        centers = torch.stack(
            [
                (hull_centers[key] if source == "visual_hull"
                 else cell.weighted_center / max(cell.weight, 1e-8))
                for source, key, cell in selected
            ]
        ) if selected else torch.empty(0, 3)
        colors = torch.stack(
            [
                (
                    cell.weighted_color / cell.color_weight
                    if cell.color_weight > 0 and selected_support[(source,key)][0] == cell.cameras
                    else torch.full((3,), float("nan"))
                )
                for source, key, cell in selected
            ]
        ) if selected else torch.empty(0, 3)
        maximum_support = max(
            (len(selected_support[(source,key)][0]) for source,key,_ in selected), default=0
        )
        support_camera_ids = torch.full(
            (len(selected), maximum_support), -1, dtype=torch.int32
        )
        support_sequence_count = torch.zeros(
            len(selected), dtype=torch.int16
        )
        for row, (_, _, cell) in enumerate(selected):
            source,key,_ = selected[row]
            cameras = sorted(selected_support[(source,key)][0])
            support_camera_ids[row, : len(cameras)] = torch.tensor(
                cameras, dtype=torch.int32
            )
            support_sequence_count[row] = len(selected_support[(source,key)][1])
        for source, key, _ in selected + duplicate_cells:
            if source == "midpoint":
                self.consumed_midpoint_cells.add(key)
                del self.cells[key]
            else:
                self.consumed_visual_hull_cells.add(key)
                del self.visual_hull_cells[key]
        self.total_births += len(selected)
        visual_hull_born = sum(
            source == "visual_hull" for source, _, _ in selected
        )
        progress('complete',selected_cells=len(selected))
        return (
            centers,
            colors,
            support_camera_ids,
            support_sequence_count,
            {
            "contract": (
                "uncovered_hit_ray_voxel_consensus_across_independent_"
                "cameras_and_configured_sequence_support"
            ),
            "minimum_cameras": int(minimum_cameras),
            "minimum_sequences": int(minimum_sequences),
            "pending_cells": len(self.cells) + len(self.visual_hull_cells),
            "pending_midpoint_cells": len(self.cells),
            "pending_visual_hull_cells": len(self.visual_hull_cells),
            "eligible_cells": len(eligible),
            "eligibility_evaluated": True,
            "eligible_midpoint_cells": len(midpoint_eligible),
            "eligible_visual_hull_cells": len(visual_hull_eligible),
            "born": len(selected),
            "visual_hull_born": int(visual_hull_born),
            "depth_consensus_rejected_cells": rejected_depth,
            "duplicate_center_cells": len(duplicate_cells),
            "minimum_birth_separation": self.minimum_birth_separation,
            "spent_first_hit_witness_cells": spent_witness_cells,
            "spent_witness_color_unknown_births": sum(
                selected_support[(source,key)][0] != cell.cameras for source,key,cell in selected),
            "consumed_first_hit_witnesses": len(self.consumed_first_hit_witnesses),
            "suppressed_first_hit_rays": self.suppressed_first_hit_rays,
            "total_proposals": self.total_proposals,
            "segment_proposals": self.segment_proposals,
            "total_births": self.total_births,
            "voxel_size": self.voxel_size,
            "visual_hull_voxel_size": self.visual_hull_voxel_size,
            "consumed_midpoint_cells": len(self.consumed_midpoint_cells),
            "consumed_visual_hull_cells": len(
                self.consumed_visual_hull_cells
            ),
            "suppressed_consumed_cells": self.suppressed_consumed_cells,
            },
        )

    def _resolve_depth_centers(self, entries):
        # Padding every candidate to the largest camera count can dominate
        # CPU memory/compute even when almost every cell has only two views.
        # Group only a clearly wasteful, homogeneous CPU-float32 batch. The
        # per-cell projection order and feasibility tolerances stay identical.
        if len(entries) >= 1024 and all(
            cell.weighted_center.dtype == torch.float32 and cell.weighted_center.device.type == 'cpu'
            for _, _, cell in entries
        ):
            widths = [len(self._available_support(cell)[0] & cell.depth_constraints.keys())
                      for _, _, cell in entries]
            if min(widths) > 0 and max(widths) * len(widths) > 2 * sum(widths):
                groups = {}
                for row, width in enumerate(widths):
                    groups.setdefault(width, []).append(row)
                centers = entries[0][2].weighted_center.new_empty((len(entries), 3))
                feasible = torch.empty(len(entries), dtype=torch.bool, device='cpu')
                for rows in groups.values():
                    group_centers, group_feasible = self._resolve_depth_centers_padded([entries[i] for i in rows])
                    centers[rows] = group_centers
                    feasible[rows] = group_feasible
                return centers, feasible
        return self._resolve_depth_centers_padded(entries)

    def _resolve_depth_centers_padded(self, entries):
        """Conservative batched feasibility solve, not ray triangulation.

        Projection onto convex depth slabs repairs a biased mean where
        possible. Only independently checked feasible results may be born;
        unconverged/inconsistent cells stay pending, without a tombstone.
        Lateral ray consistency still requires downstream verification.
        """
        if not entries:
            return torch.empty(0, 3), torch.empty(0, dtype=torch.bool)
        centers = torch.stack([c.weighted_center / max(c.weight, 1e-8)
                               for _, _, c in entries])
        count = len(entries)
        supports = [self._available_support(c)[0] for _,_,c in entries]
        width = max((len(cameras & c.depth_constraints.keys())
                     for cameras,(_,_,c) in zip(supports,entries)),default=0)
        if not width:
            return centers, torch.zeros(count, dtype=torch.bool)
        # Filling millions of individual Torch scalar/slice views dominated
        # topology CPU time. These are CPU float32 metadata, not trainable
        # tensors: contiguous NumPy construction retains the exact values.
        normals_array = np.zeros((count,width,3),dtype=np.float32)
        lower_array = np.full((count,width),-np.inf,dtype=np.float32)
        upper_array = np.full((count,width),np.inf,dtype=np.float32)
        complete_array = np.zeros(count,dtype=bool)
        for i, (_, _, cell) in enumerate(entries):
            cameras = supports[i]
            complete_array[i] = bool(cameras) and cameras.issubset(cell.depth_constraints)
            for j, camera in enumerate(sorted(cameras & cell.depth_constraints.keys())):
                normal, lo, hi = cell.depth_constraints[camera]
                normals_array[i,j] = normal.numpy()
                lower_array[i,j],upper_array[i,j] = lo,hi
        normals = torch.from_numpy(normals_array)
        lower = torch.from_numpy(lower_array)
        upper = torch.from_numpy(upper_array)
        complete = torch.from_numpy(complete_array)
        box_lower = torch.tensor([key for _, key, _ in entries]).float() * self.visual_hull_voxel_size
        box_upper = box_lower + self.visual_hull_voxel_size
        # Aggregated historical means cannot be unmixed after a witness is
        # spent. Start from the neutral voxel center instead of pretending
        # its old first-hit position still supplies positive geometry.
        partial = torch.tensor([cameras != cell.cameras for cameras,(_,_,cell)
                                in zip(supports,entries)],dtype=torch.bool)
        historical_centers = centers.clone() if bool(partial.any()) else None
        centers[partial] = (box_lower[partial]+box_upper[partial])*.5
        initial_depth = (centers[:,None,:]*normals).sum(dim=2)
        already_feasible = ((initial_depth >= lower).all(dim=1)
            & (initial_depth <= upper).all(dim=1)
            & (centers >= box_lower).all(dim=1) & (centers <= box_upper).all(dim=1))
        # Every cyclic projection is identically zero for an already-feasible
        # point. Skip only those exact fixed points, with no relaxed tolerance.
        active = ~already_feasible
        if bool(active.any()):
            repair = centers[active].clone()
            repair_normals,repair_lower,repair_upper = normals[active],lower[active],upper[active]
            repair_box_lower,repair_box_upper = box_lower[active],box_upper[active]
            for _ in range(24):
                for j in range(width):
                    depth = (repair * repair_normals[:,j]).sum(dim=1)
                    delta = depth.clamp(min=repair_lower[:,j],max=repair_upper[:,j])-depth
                    repair += delta[:,None]*repair_normals[:,j]
                repair = torch.maximum(repair_box_lower,torch.minimum(repair_box_upper,repair))
            centers[active] = repair
        depth = (centers[:, None, :] * normals).sum(dim=2)
        feasible = (complete & torch.isfinite(centers).all(dim=1)
                    & (depth >= lower - 1e-5).all(dim=1)
                    & (depth <= upper + 1e-5).all(dim=1))
        fallback = partial & ~feasible
        if bool(fallback.any()):
            # Near-parallel slabs may converge slowly from the neutral point.
            # A historical center is also an admissible numerical warm start,
            # but ONLY the remaining cameras may certify the result.
            repair = historical_centers[fallback].clone()
            fn,fl,fu = normals[fallback],lower[fallback],upper[fallback]
            for _ in range(24):
                for j in range(width):
                    d = (repair*fn[:,j]).sum(dim=1)
                    repair += (d.clamp(min=fl[:,j],max=fu[:,j])-d)[:,None]*fn[:,j]
                repair = torch.maximum(box_lower[fallback],torch.minimum(box_upper[fallback],repair))
            d = (repair[:,None,:]*fn).sum(dim=2)
            valid = (complete[fallback]&torch.isfinite(repair).all(dim=1)
                     &(d>=fl-1e-5).all(dim=1)&(d<=fu+1e-5).all(dim=1))
            rows = torch.nonzero(fallback,as_tuple=True)[0][valid]
            centers[rows] = repair[valid]
            feasible[rows] = True
        return centers, feasible

    def mark_consumed_centers(self, centers: torch.Tensor) -> int:
        """Tombstone runtime births restored from a pre-v5 checkpoint.

        Older accumulator states did not retain the drained cell key.  The
        corresponding model rows do retain their world-space centres and
        ``initialization_source=6`` provenance, which is sufficient to block
        both midpoint and visual-hull grids around every extant birth.
        """

        centers = torch.as_tensor(centers).float().cpu().reshape(-1, 3)
        before = len(self.consumed_midpoint_cells) + len(
            self.consumed_visual_hull_cells
        )
        if len(centers):
            for center in centers:
                if not self._near_emitted_center(center):
                    self._register_center(center)
            midpoint = torch.floor(centers / self.voxel_size).to(torch.int64)
            visual_hull = torch.floor(
                centers / self.visual_hull_voxel_size
            ).to(torch.int64)
            self.consumed_midpoint_cells.update(
                tuple(int(value) for value in row.tolist()) for row in midpoint
            )
            self.consumed_visual_hull_cells.update(
                tuple(int(value) for value in row.tolist()) for row in visual_hull
            )
        return (
            len(self.consumed_midpoint_cells)
            + len(self.consumed_visual_hull_cells)
            - before
        )

    def capture(self) -> dict:
        return {
            "version": "static-ray-birth-accumulator-v6",
            "first_hit_capacity": "one_physical_hit_per_camera_ray__posterior_width_is_not_matter",
            "consumed_first_hit_witnesses": sorted(self.consumed_first_hit_witnesses),
            "suppressed_first_hit_rays": self.suppressed_first_hit_rays,
            "evidence_weighting": "one_contribution_per_camera_per_cell",
            "interval_consensus": "all_support_depth_slabs_and_voxel",
            "voxel_size": self.voxel_size,
            "visual_hull_voxel_size": self.visual_hull_voxel_size,
            "maximum_segment_samples": self.maximum_segment_samples,
            "maximum_proposals_per_update": self.maximum_proposals_per_update,
            "minimum_birth_separation": self.minimum_birth_separation,
            "emitted_centers": self.emitted_centers.copy(),
            "total_proposals": self.total_proposals,
            "segment_proposals": self.segment_proposals,
            "total_births": self.total_births,
            "suppressed_consumed_cells": self.suppressed_consumed_cells,
            "consumed_midpoint_cells": sorted(self.consumed_midpoint_cells),
            "consumed_visual_hull_cells": sorted(
                self.consumed_visual_hull_cells
            ),
            "cells": [
                {
                    "key": key,
                    "weighted_center": cell.weighted_center,
                    "weight": cell.weight,
                    "weighted_color": cell.weighted_color,
                    "color_weight": cell.color_weight,
                    "cameras": sorted(cell.cameras),
                    "sequences": sorted(cell.sequences),
                    "camera_sequences": cell.camera_sequences,
                }
                for key, cell in self.cells.items()
            ],
            "visual_hull_cells": [
                {
                    "key": key,
                    "weighted_center": cell.weighted_center,
                    "weight": cell.weight,
                    "weighted_color": cell.weighted_color,
                    "color_weight": cell.color_weight,
                    "cameras": sorted(cell.cameras),
                    "sequences": sorted(cell.sequences),
                    "depth_constraints": cell.depth_constraints,
                    "camera_sequences": cell.camera_sequences,
                    "first_hit_witnesses": cell.first_hit_witnesses,
                }
                for key, cell in self.visual_hull_cells.items()
            ],
        }

    def restore(self, state: dict | None) -> None:
        if not state:
            return
        version = state.get("version")
        if version not in {
            "static-ray-birth-accumulator-v3",
            "static-ray-birth-accumulator-v4",
            "static-ray-birth-accumulator-v5",
            "static-ray-birth-accumulator-v6",
        }:
            raise RuntimeError("Unsupported static ray-birth accumulator state")
        if float(state["voxel_size"]) != self.voxel_size:
            raise RuntimeError("Static ray-birth voxel size changed on resume")
        if float(state.get("minimum_birth_separation", 0.0)) != self.minimum_birth_separation:
            raise RuntimeError("Static ray-birth separation changed on resume")
        self.emitted_centers = []
        self._center_index = {}
        for center in state.get("emitted_centers", []):
            self._register_center(center)
        if (
            int(state["maximum_proposals_per_update"])
            != self.maximum_proposals_per_update
        ):
            raise RuntimeError("Static ray-birth proposal cap changed on resume")
        if version in {
            "static-ray-birth-accumulator-v4",
            "static-ray-birth-accumulator-v5",
            "static-ray-birth-accumulator-v6",
        }:
            if (
                float(state["visual_hull_voxel_size"])
                != self.visual_hull_voxel_size
            ):
                raise RuntimeError(
                    "Static visual-hull voxel size changed on resume"
                )
            if (
                int(state["maximum_segment_samples"])
                != self.maximum_segment_samples
            ):
                raise RuntimeError(
                    "Static visual-hull segment sampling changed on resume"
                )
        self.total_proposals = int(state.get("total_proposals", 0))
        self.segment_proposals = int(state.get("segment_proposals", 0))
        self.total_births = int(state.get("total_births", 0))
        self.consumed_first_hit_witnesses = {tuple(int(v) for v in token)
            for token in state.get("consumed_first_hit_witnesses", [])}
        self.suppressed_first_hit_rays = int(state.get("suppressed_first_hit_rays",0))
        self.suppressed_consumed_cells = int(
            state.get("suppressed_consumed_cells", 0)
        )
        self.consumed_midpoint_cells = {
            tuple(int(value) for value in row)
            for row in state.get("consumed_midpoint_cells", [])
        }
        self.consumed_visual_hull_cells = {
            tuple(int(value) for value in row)
            for row in state.get("consumed_visual_hull_cells", [])
        }
        def restore_cells(rows: list[dict]) -> dict[tuple[int, int, int], _Cell]:
            restored = {}
            for row in rows:
                key = tuple(int(value) for value in row["key"])
                restored[key] = _Cell(
                    weighted_center=torch.as_tensor(
                        row["weighted_center"]
                    ).float().cpu(),
                    weight=float(row["weight"]),
                    weighted_color=torch.as_tensor(
                        row["weighted_color"]
                    ).float().cpu(),
                    color_weight=float(row["color_weight"]),
                    cameras={int(value) for value in row["cameras"]},
                    sequences={str(value) for value in row["sequences"]},
                    camera_sequences={int(k):str(v) for k,v in row.get("camera_sequences",{}).items()},
                    first_hit_witnesses={int(k):tuple(int(x) for x in v)
                        for k,v in row.get("first_hit_witnesses",{}).items()},
                    depth_constraints={
                        int(camera): (torch.as_tensor(value[0]).float().cpu(),
                                      float(value[1]), float(value[2]))
                        for camera, value in row.get("depth_constraints", {}).items()
                    },
                )
            return restored

        # v3 stored only posterior midpoints. Preserve those pending cells
        # exactly and start the interval visual hull from subsequent evidence.
        self.cells = restore_cells(state.get("cells", []))
        self.visual_hull_cells = restore_cells(
            state.get("visual_hull_cells", [])
        )
