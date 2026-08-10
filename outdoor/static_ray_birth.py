"""Cross-view accumulator for uncovered foliage hit-ray births."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch


@dataclass
class _Cell:
    weighted_center: torch.Tensor
    weight: float
    weighted_color: torch.Tensor
    color_weight: float
    cameras: set[int] = field(default_factory=set)
    sequences: set[str] = field(default_factory=set)


class StaticRayBirthAccumulator:
    """Promote uncovered hits only after independent static confirmation."""

    def __init__(
        self,
        *,
        voxel_size: float = 0.15,
        visual_hull_voxel_size: float = 0.30,
        maximum_segment_samples: int = 16,
        maximum_proposals_per_update: int = 256,
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
    ) -> None:
        cell = cells.get(key)
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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, int | float | str],
    ]:
        maximum_births = max(int(maximum_births), 0)
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
        eligible.sort(
            key=lambda item: (
                -len(item[2].sequences),
                -len(item[2].cameras),
                -item[2].weight,
                item[0],
                item[1],
            )
        )
        selected = eligible[:maximum_births]
        centers = torch.stack(
            [
                cell.weighted_center / max(cell.weight, 1e-8)
                for _, _, cell in selected
            ]
        ) if selected else torch.empty(0, 3)
        colors = torch.stack(
            [
                (
                    cell.weighted_color / cell.color_weight
                    if cell.color_weight > 0
                    else torch.full((3,), float("nan"))
                )
                for _, _, cell in selected
            ]
        ) if selected else torch.empty(0, 3)
        maximum_support = max(
            (len(cell.cameras) for _, _, cell in selected), default=0
        )
        support_camera_ids = torch.full(
            (len(selected), maximum_support), -1, dtype=torch.int32
        )
        support_sequence_count = torch.zeros(
            len(selected), dtype=torch.int16
        )
        for row, (_, _, cell) in enumerate(selected):
            cameras = sorted(cell.cameras)
            support_camera_ids[row, : len(cameras)] = torch.tensor(
                cameras, dtype=torch.int32
            )
            support_sequence_count[row] = len(cell.sequences)
        for source, key, _ in selected:
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
        return (
            centers,
            colors,
            support_camera_ids,
            support_sequence_count,
            {
            "contract": (
                "uncovered_hit_ray_voxel_consensus_across_independent_"
                "cameras_and_sequences"
            ),
            "pending_cells": len(self.cells) + len(self.visual_hull_cells),
            "pending_midpoint_cells": len(self.cells),
            "pending_visual_hull_cells": len(self.visual_hull_cells),
            "eligible_cells": len(eligible),
            "eligible_midpoint_cells": len(midpoint_eligible),
            "eligible_visual_hull_cells": len(visual_hull_eligible),
            "born": len(selected),
            "visual_hull_born": int(visual_hull_born),
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
            "version": "static-ray-birth-accumulator-v5",
            "voxel_size": self.voxel_size,
            "visual_hull_voxel_size": self.visual_hull_voxel_size,
            "maximum_segment_samples": self.maximum_segment_samples,
            "maximum_proposals_per_update": self.maximum_proposals_per_update,
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
        }:
            raise RuntimeError("Unsupported static ray-birth accumulator state")
        if float(state["voxel_size"]) != self.voxel_size:
            raise RuntimeError("Static ray-birth voxel size changed on resume")
        if (
            int(state["maximum_proposals_per_update"])
            != self.maximum_proposals_per_update
        ):
            raise RuntimeError("Static ray-birth proposal cap changed on resume")
        if version in {
            "static-ray-birth-accumulator-v4",
            "static-ray-birth-accumulator-v5",
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
                )
            return restored

        # v3 stored only posterior midpoints. Preserve those pending cells
        # exactly and start the interval visual hull from subsequent evidence.
        self.cells = restore_cells(state.get("cells", []))
        self.visual_hull_cells = restore_cells(
            state.get("visual_hull_cells", [])
        )
