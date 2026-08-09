"""Convert sequence observations into one persistent static foliage model."""

from __future__ import annotations

from collections import Counter

import torch

from outdoor.hybrid_gaussian_renderer import (
    LAYER_CANONICAL_CROWN,
    LAYER_DYNAMIC_LEAF,
)


def _first_valid_camera(table: torch.Tensor) -> torch.Tensor:
    if table.ndim != 2:
        raise ValueError("support camera table must have shape [N,K]")
    valid = table >= 0
    first = valid.to(torch.int64).argmax(dim=1)
    result = table.gather(1, first[:, None]).squeeze(1).to(torch.int64)
    return torch.where(valid.any(dim=1), result, torch.full_like(result, -1))


def _camera_sequence_table(
    selected_views: list[dict], maximum_camera_id: int
) -> tuple[torch.Tensor, list[str]]:
    """Map calibrated camera ids to stable sequence ids."""
    names = sorted(
        {
            str(record.get("sequence_id", ""))
            for record in selected_views
            if str(record.get("sequence_id", ""))
        }
    )
    name_to_id = {name: index for index, name in enumerate(names)}
    table = torch.full(
        (max(int(maximum_camera_id) + 1, 0),), -1, dtype=torch.int64
    )
    for record in selected_views:
        camera_id = int(record.get("image_id", -1))
        name = str(record.get("sequence_id", ""))
        if camera_id < 0 or camera_id >= len(table) or name not in name_to_id:
            continue
        encoded = int(name_to_id[name])
        if int(table[camera_id]) >= 0 and int(table[camera_id]) != encoded:
            raise ValueError(
                "fixed camera has conflicting sequence identities: "
                f"camera={camera_id}"
            )
        table[camera_id] = encoded
    return table, names


def _camera_quality_table(
    records: list[dict], maximum_camera_id: int
) -> torch.Tensor:
    """Return bounded canonical-appearance weights for fixed cameras."""
    table = torch.ones(
        max(int(maximum_camera_id) + 1, 0), dtype=torch.float32
    )
    for record in records:
        camera_id = int(record.get("image_id", -1))
        if camera_id < 0 or camera_id >= len(table):
            continue
        quality = float(record.get("canonical_quality", 1.0))
        if not torch.isfinite(torch.tensor(quality)) or quality <= 0:
            raise ValueError(
                "fixed-camera canonical quality must be finite and positive"
            )
        table[camera_id] = float(min(max(quality, 0.25), 2.0))
    return table


def _canonical_sequence_per_tree(
    tree_ids: torch.Tensor,
    sequence_ids: torch.Tensor,
    camera_ids: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[int, int]]:
    """Choose one coherent static snapshot sequence for each tree.

    Independent calibrated cameras dominate the score; accumulated posterior
    confidence breaks ties.  This deliberately chooses a persistent static
    snapshot instead of averaging incompatible leaf positions across time.
    """
    if not (
        tree_ids.shape
        == sequence_ids.shape
        == camera_ids.shape
        == weights.shape
    ):
        raise ValueError("canonical sequence evidence arrays must align")
    chosen: dict[int, int] = {}
    valid = (tree_ids >= 0) & (sequence_ids >= 0) & (camera_ids >= 0)
    for tree_id in torch.unique(tree_ids[valid]).tolist():
        tree_mask = valid & (tree_ids == int(tree_id))
        best_score = None
        best_sequence = -1
        for sequence_id in torch.unique(sequence_ids[tree_mask]).tolist():
            rows = tree_mask & (sequence_ids == int(sequence_id))
            score = (
                int(torch.unique(camera_ids[rows]).numel()),
                float(weights[rows].sum()),
                -int(sequence_id),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_sequence = int(sequence_id)
        chosen[int(tree_id)] = best_sequence
    per_row = torch.full_like(sequence_ids, -1)
    for tree_id, sequence_id in chosen.items():
        per_row[tree_ids == tree_id] = int(sequence_id)
    return per_row, chosen


def _canonical_sequence_for_scene(
    tree_ids: torch.Tensor,
    group_ids: torch.Tensor,
    sequence_ids: torch.Tensor,
    camera_ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    minimum_supporting_views: int,
) -> tuple[int, list[dict[str, int | float]]]:
    """Choose one calibrated snapshot for the entire static scene.

    A static localization map cannot assign a different acquisition time to
    every tree: doing so creates a scene that never existed and turns denser
    leaf detail into stronger cross-view disagreement.  The score therefore
    first maximizes scene/tree coverage, then independently verified tree
    cells and calibrated camera support.  Confidence is only a final tie
    breaker, so a few high-confidence closeups cannot select a partial scene.
    """
    if not (
        tree_ids.shape
        == group_ids.shape
        == sequence_ids.shape
        == camera_ids.shape
        == weights.shape
    ):
        raise ValueError("scene canonical sequence evidence arrays must align")
    valid = (
        (tree_ids >= 0)
        & (group_ids >= 0)
        & (sequence_ids >= 0)
        & (camera_ids >= 0)
    )
    if not bool(valid.any()):
        raise ValueError("scene canonical sequence selection has no valid rows")
    maximum_camera = int(camera_ids[valid].max()) + 2
    score_table: list[dict[str, int | float]] = []
    best_score = None
    best_sequence = -1
    for sequence_id in torch.unique(sequence_ids[valid]).tolist():
        rows = valid & (sequence_ids == int(sequence_id))
        pairs = torch.unique(
            group_ids[rows] * maximum_camera + camera_ids[rows] + 1
        )
        pair_groups = torch.div(
            pairs, maximum_camera, rounding_mode="floor"
        )
        unique_groups, group_camera_count = torch.unique(
            pair_groups, return_counts=True
        )
        verified_group_count = int(
            (group_camera_count >= int(minimum_supporting_views)).sum()
        )
        record: dict[str, int | float] = {
            "sequence_id": int(sequence_id),
            "tree_count": int(torch.unique(tree_ids[rows]).numel()),
            "verified_group_count": verified_group_count,
            "camera_count": int(torch.unique(camera_ids[rows]).numel()),
            "group_count": int(len(unique_groups)),
            "row_count": int(rows.sum()),
            "posterior_weight": float(weights[rows].sum()),
        }
        score_table.append(record)
        score = (
            int(record["tree_count"]),
            int(record["verified_group_count"]),
            int(record["camera_count"]),
            int(record["group_count"]),
            float(record["posterior_weight"]),
            -int(sequence_id),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_sequence = int(sequence_id)
    return best_sequence, score_table


def _append_ownerless_static_ray_births(
    result: dict,
    original: dict,
    *,
    voxel_size: float = 0.15,
    minimum_supporting_views: int = 2,
    minimum_supporting_sequences: int = 2,
    initial_opacity: float = 0.02,
) -> tuple[dict, dict[str, int | float]]:
    """Promote only cross-sequence consensus from ownerless hit births."""
    centers = torch.as_tensor(original["centers"]).float().cpu()
    count = len(centers)
    layer = torch.as_tensor(original["layer_role"]).cpu()
    groups = torch.as_tensor(original["replacement_group"]).cpu()
    candidates = (layer == LAYER_DYNAMIC_LEAF) & (groups < 0)
    rows = torch.nonzero(candidates, as_tuple=False).flatten()
    if not len(rows):
        return result, {
            "ownerless_input_rows": 0,
            "ownerless_static_births": 0,
            "voxel_size": float(voxel_size),
        }
    support = torch.as_tensor(original["support_camera_ids"]).cpu()
    camera = _first_valid_camera(support)[rows]
    valid_camera = camera >= 0
    rows = rows[valid_camera]
    camera = camera[valid_camera]
    if not len(rows):
        return result, {
            "ownerless_input_rows": int(candidates.sum()),
            "ownerless_static_births": 0,
            "voxel_size": float(voxel_size),
        }
    tree = torch.as_tensor(original["tree_instance_id"]).long().cpu()[rows]
    quantized = torch.floor(centers[rows] / float(voxel_size)).long()
    cell_key = torch.cat([tree[:, None], quantized], dim=1)
    cells, inverse = torch.unique(cell_key, dim=0, return_inverse=True)
    cell_count = len(cells)
    maximum_camera = int(camera.max()) + 2
    unique_camera_pair = torch.unique(
        inverse * maximum_camera + camera + 1
    )
    pair_cell = torch.div(
        unique_camera_pair, maximum_camera, rounding_mode="floor"
    )
    view_count = torch.zeros(cell_count, dtype=torch.int64)
    view_count.index_add_(
        0, pair_cell, torch.ones_like(pair_cell, dtype=torch.int64)
    )
    selected_views = original.get("audit", {}).get("selected_views", [])
    sequence_name_to_id: dict[str, int] = {}
    camera_sequence = torch.full(
        (maximum_camera - 1,), -1, dtype=torch.int64
    )
    for record in selected_views:
        image_id = int(record.get("image_id", -1))
        if image_id < 0 or image_id >= len(camera_sequence):
            continue
        name = str(record.get("sequence_id", ""))
        if name not in sequence_name_to_id:
            sequence_name_to_id[name] = len(sequence_name_to_id)
        camera_sequence[image_id] = sequence_name_to_id[name]
    sequence = camera_sequence[camera.clamp(0, len(camera_sequence) - 1)]
    valid_sequence = sequence >= 0
    if bool(valid_sequence.any()):
        maximum_sequence = int(sequence[valid_sequence].max()) + 2
        unique_sequence_pair = torch.unique(
            inverse[valid_sequence] * maximum_sequence
            + sequence[valid_sequence]
            + 1
        )
        sequence_cell = torch.div(
            unique_sequence_pair,
            maximum_sequence,
            rounding_mode="floor",
        )
        sequence_count = torch.zeros(cell_count, dtype=torch.int64)
        sequence_count.index_add_(
            0,
            sequence_cell,
            torch.ones_like(sequence_cell, dtype=torch.int64),
        )
    else:
        sequence_count = torch.zeros(cell_count, dtype=torch.int64)
    valid_cells = torch.nonzero(
        (view_count >= int(minimum_supporting_views))
        & (sequence_count >= int(minimum_supporting_sequences)),
        as_tuple=False,
    ).flatten()
    if not len(valid_cells):
        return result, {
            "ownerless_input_rows": int(candidates.sum()),
            "ownerless_multiview_cells": int(
                (view_count >= minimum_supporting_views).sum()
            ),
            "ownerless_static_births": 0,
            "voxel_size": float(voxel_size),
        }
    valid_lookup = torch.full((cell_count,), -1, dtype=torch.int64)
    valid_lookup[valid_cells] = torch.arange(len(valid_cells))
    source_mask = valid_lookup[inverse] >= 0
    source_rows = rows[source_mask]
    source_cell = valid_lookup[inverse[source_mask]]
    occupancy = torch.as_tensor(
        original.get("occupancy_probability", torch.ones(count))
    ).float().cpu()[source_rows]
    weight = occupancy.clamp(0.05, 1.0)
    weight_sum = torch.zeros(len(valid_cells))
    weight_sum.index_add_(0, source_cell, weight)

    def mean(value: torch.Tensor, local_weight: torch.Tensor) -> torch.Tensor:
        total = torch.zeros(
            (len(valid_cells),) + tuple(value.shape[1:]), dtype=value.dtype
        )
        index = source_cell.reshape(
            (-1,) + (1,) * (value.ndim - 1)
        ).expand_as(value)
        total.scatter_add_(
            0,
            index,
            value
            * local_weight.reshape(
                (-1,) + (1,) * (value.ndim - 1)
            ),
        )
        denominator = torch.zeros(len(valid_cells), dtype=local_weight.dtype)
        denominator.index_add_(0, source_cell, local_weight)
        return total / denominator.clamp_min(1e-8).reshape(
            (-1,) + (1,) * (value.ndim - 1)
        )

    birth_centers = mean(centers[source_rows], weight)
    colors = torch.as_tensor(original["colors"]).float().cpu()[source_rows]
    first_color = mean(colors, weight)
    residual = colors - first_color[source_cell]
    robust_weight = weight / (
        1.0 + (residual.norm(dim=1) / 0.20).square()
    )
    birth_colors = mean(colors, robust_weight).clamp(0.02, 0.98)
    source_scales = torch.as_tensor(original["scales"]).float().cpu()[
        source_rows
    ]
    tangent = torch.sort(source_scales, dim=1).values[:, :2].mean(dim=1)
    birth_radius = mean(tangent[:, None], weight)[:, 0].clamp(0.008, 0.06)
    birth_scales = birth_radius[:, None].expand(-1, 3).clone()
    order = torch.argsort(source_cell)
    ordered_cell = source_cell[order]
    first = torch.ones(len(order), dtype=torch.bool)
    first[1:] = ordered_cell[1:] != ordered_cell[:-1]
    representative_rows = source_rows[order[first]]
    if len(representative_rows) != len(valid_cells):
        raise RuntimeError("ownerless static birth representative mismatch")
    # A promoted cell is owned by every calibrated camera that contributed
    # to the cross-sequence consensus.  Copying the representative row here
    # silently reduced a verified multi-view birth back to a single-camera
    # primitive: support_view_count stayed >= 2, but exact ray/RGB ownership
    # was unavailable from all other supporting views.  Materialize the
    # actual union so the metadata contract matches the evidence that made
    # the birth eligible in the first place.
    support_width = int(support.shape[1])
    birth_support_camera_ids = torch.full(
        (len(valid_cells), support_width), -1, dtype=support.dtype
    )
    source_camera = camera[source_mask]
    camera_pairs = torch.unique(
        source_cell * maximum_camera + source_camera + 1
    )
    pair_cell = torch.div(
        camera_pairs, maximum_camera, rounding_mode="floor"
    )
    pair_camera = camera_pairs.remainder(maximum_camera) - 1
    pair_count = torch.bincount(pair_cell, minlength=len(valid_cells))
    pair_start = torch.cumsum(pair_count, dim=0) - pair_count
    pair_rank = torch.arange(len(pair_cell)) - torch.repeat_interleave(
        pair_start, pair_count
    )
    within_width = pair_rank < support_width
    birth_support_camera_ids[
        pair_cell[within_width], pair_rank[within_width]
    ] = pair_camera[within_width].to(support.dtype)
    existing_count = len(torch.as_tensor(result["centers"]))
    original_row_fields = {
        name: value.cpu()
        for name, value in original.items()
        if torch.is_tensor(value) and value.ndim and len(value) == count
    }
    for name, existing in list(result.items()):
        if not (
            torch.is_tensor(existing)
            and existing.ndim
            and len(existing) == existing_count
        ):
            continue
        if name in original_row_fields:
            detail = original_row_fields[name][representative_rows].clone()
        else:
            detail = torch.zeros(
                (len(valid_cells),) + tuple(existing.shape[1:]),
                dtype=existing.dtype,
            )
        if name in {"centers", "initialization_center"}:
            detail = birth_centers.to(existing.dtype)
        elif name == "colors":
            detail = birth_colors.to(existing.dtype)
        elif name == "scales":
            detail = birth_scales.to(existing.dtype)
        elif name == "opacities":
            detail = torch.full(
                (len(valid_cells), 1),
                float(initial_opacity),
                dtype=existing.dtype,
            )
        elif name == "layer_role":
            detail.fill_(LAYER_CANONICAL_CROWN)
        elif name == "replacement_group":
            detail.fill_(-1)
        elif name == "support_view_count":
            detail = view_count[valid_cells].to(existing.dtype)
        elif name == "support_sequence_count":
            detail = sequence_count[valid_cells].to(existing.dtype)
        elif name in {"track_id", "evidence_primitive_id"}:
            detail.fill_(-1)
        elif name in {"track_linearity", "static_skeleton_confidence"}:
            detail.zero_()
        elif name == "initialization_source":
            detail.fill_(6)
        elif name in {"observation_camera_ids", "support_camera_ids"}:
            detail = birth_support_camera_ids[
                :, : existing.shape[1]
            ].to(existing.dtype)
        elif name in {"observation_uv", "observation_depth"}:
            detail.fill_(float("nan"))
        elif name == "position_covariance":
            detail = torch.diag_embed(birth_scales.square()).to(existing.dtype)
        elif name == "scale_ceiling":
            detail = (birth_scales * 1.5).to(existing.dtype)
        elif name in {
            "free_space_violation_count",
            "unknown_view_count",
            "split_generation",
            "replacement_observation_count",
        }:
            detail.zero_()
        elif name in {"replacement_overlap_ema"}:
            detail.zero_()
        elif name == "static_detail":
            detail.fill_(True)
        result[name] = torch.cat([existing, detail], dim=0)
    return result, {
        "ownerless_input_rows": int(candidates.sum()),
        "ownerless_multiview_cells": int(
            (view_count >= minimum_supporting_views).sum()
        ),
        "ownerless_cross_sequence_cells": int(len(valid_cells)),
        "ownerless_contributing_rows": int(len(source_rows)),
        "ownerless_static_births": int(len(valid_cells)),
        "ownerless_support_camera_pairs_retained": int(within_width.sum()),
        "ownerless_support_camera_pairs_truncated": int(
            len(camera_pairs) - int(within_width.sum())
        ),
        "voxel_size": float(voxel_size),
        "minimum_supporting_views": int(minimum_supporting_views),
        "minimum_supporting_sequences": int(minimum_supporting_sequences),
    }


def fuse_sequence_evidence_into_static_leaves(
    payload: dict,
    *,
    minimum_supporting_views: int = 2,
    initial_opacity: float = 0.03,
    canonical_mode_voxel_size: float = 0.08,
    maximum_modes_per_group: int = 2,
    canonical_sequence_policy: str = "per_tree",
    fixed_camera_sequences: list[dict] | None = None,
) -> tuple[dict, dict[str, int | float | str]]:
    """Fuse camera observations into a coherent multi-mode static snapshot.

    A sequence-owned Gaussian is not a valid primitive in a localization map.
    Its calibrated xyz/RGB observation remains valuable evidence, however.
    Cross-view evidence first verifies each persistent visual-hull cell. One
    internally consistent canonical sequence is selected for each independent
    tree instance, and up to a bounded number of spatial/color modes are
    retained per cell. Different trees have no shared non-rigid state, so a
    scene-global sequence needlessly discards valid trees absent from that
    acquisition. The emitted primitives still form one unconditional static
    model; sequence identity is training evidence, never a render condition.
    """
    minimum_supporting_views = int(minimum_supporting_views)
    if minimum_supporting_views < 2:
        raise ValueError("static leaf fusion requires at least two views")
    if not 0.0 < float(initial_opacity) < 1.0:
        raise ValueError("initial static leaf opacity must lie in (0,1)")
    if float(canonical_mode_voxel_size) <= 0:
        raise ValueError("canonical static mode voxel size must be positive")
    if int(maximum_modes_per_group) <= 0:
        raise ValueError("maximum static modes per group must be positive")
    if canonical_sequence_policy not in {"scene", "per_tree"}:
        raise ValueError(
            "canonical sequence policy must be 'scene' or 'per_tree'"
        )
    result = dict(payload)
    centers = torch.as_tensor(payload["centers"]).cpu()
    count = len(centers)
    layer = torch.as_tensor(payload["layer_role"], dtype=torch.int8).cpu()
    groups = torch.as_tensor(
        payload["replacement_group"], dtype=torch.int64
    ).cpu()
    if layer.shape != (count,) or groups.shape != (count,):
        raise ValueError("foliage role/group tables do not match centers")
    dynamic = layer == LAYER_DYNAMIC_LEAF
    envelope = layer == LAYER_CANONICAL_CROWN
    envelope_rows = torch.nonzero(envelope, as_tuple=False).flatten()
    group_count = len(envelope_rows)
    if group_count:
        expected = torch.arange(group_count, dtype=torch.int64)
        if not torch.equal(groups[envelope_rows], expected):
            raise ValueError("persistent envelope groups must be contiguous")
    associated = dynamic & (groups >= 0) & (groups < group_count)
    associated_rows = torch.nonzero(associated, as_tuple=False).flatten()
    keep = ~dynamic
    if not len(associated_rows):
        for name, value in payload.items():
            if torch.is_tensor(value) and value.ndim and len(value) == count:
                result[name] = value.cpu()[keep]
        result["static_detail"] = torch.zeros(int(keep.sum()), dtype=torch.bool)
        return result, {
            "contract": "multiview_sequence_evidence_to_static_leaf_clusters",
            "input_dynamic_rows": int(dynamic.sum()),
            "associated_dynamic_rows": 0,
            "fused_static_leaf_clusters": 0,
            "discarded_single_view_or_ownerless_rows": int(dynamic.sum()),
        }

    support = torch.as_tensor(payload["support_camera_ids"]).cpu()
    owner_camera = _first_valid_camera(support)[associated_rows]
    local_groups = groups[associated_rows]
    valid_camera = owner_camera >= 0
    if not bool(valid_camera.any()):
        raise ValueError("associated leaf observations have no owner camera")
    maximum_camera = int(owner_camera[valid_camera].max()) + 2
    pair_key = (
        local_groups[valid_camera] * maximum_camera
        + owner_camera[valid_camera]
        + 1
    )
    unique_pair = torch.unique(pair_key)
    pair_group = torch.div(
        unique_pair, maximum_camera, rounding_mode="floor"
    )
    view_count = torch.zeros(group_count, dtype=torch.int64)
    view_count.index_add_(
        0, pair_group, torch.ones_like(pair_group, dtype=torch.int64)
    )
    valid_groups = torch.nonzero(
        view_count >= minimum_supporting_views, as_tuple=False
    ).flatten()
    if not len(valid_groups):
        for name, value in payload.items():
            if torch.is_tensor(value) and value.ndim and len(value) == count:
                result[name] = value.cpu()[keep]
        result["static_detail"] = torch.zeros(int(keep.sum()), dtype=torch.bool)
        return result, {
            "contract": "multiview_sequence_evidence_to_static_leaf_clusters",
            "input_dynamic_rows": int(dynamic.sum()),
            "associated_dynamic_rows": int(associated.sum()),
            "fused_static_leaf_clusters": 0,
            "discarded_single_view_or_ownerless_rows": int(dynamic.sum()),
        }

    occupancy = torch.as_tensor(
        payload.get("occupancy_probability", torch.ones(count))
    ).float().cpu()
    persisted_fixed_cameras = payload.get("audit", {}).get(
        "fixed_camera_sequences", []
    )
    selected_views = payload.get("audit", {}).get("selected_views", [])
    if fixed_camera_sequences is not None:
        camera_sequence_records = list(fixed_camera_sequences)
        camera_sequence_metadata_source = "runtime_fixed_camera_contract"
    elif persisted_fixed_cameras:
        camera_sequence_records = list(persisted_fixed_cameras)
        camera_sequence_metadata_source = "initialization_fixed_camera_contract"
    else:
        # Backward compatibility for old initialization files. Production
        # training passes the complete fixed-camera table explicitly, so the
        # selected posterior subset is never the authority for sequence
        # identity again.
        camera_sequence_records = list(selected_views)
        camera_sequence_metadata_source = "legacy_selected_view_subset"
    maximum_selected_camera = max(
        [
            int(record.get("image_id", -1))
            for record in camera_sequence_records
        ]
        + [int(owner_camera[valid_camera].max())]
    )
    camera_sequence, sequence_names = _camera_sequence_table(
        camera_sequence_records, maximum_selected_camera
    )
    # Runtime camera objects are authoritative for identity, while persisted
    # initialization records additionally carry the RGB-quality audit. Merge
    # only that bounded tiebreaker by camera id.
    camera_quality_records = (
        list(persisted_fixed_cameras)
        if persisted_fixed_cameras
        else camera_sequence_records
    )
    camera_quality = _camera_quality_table(
        camera_quality_records, maximum_selected_camera
    )
    owner_sequence = torch.full_like(owner_camera, -1)
    sequence_camera = valid_camera & (
        owner_camera < len(camera_sequence)
    )
    owner_sequence[sequence_camera] = camera_sequence[
        owner_camera[sequence_camera]
    ]
    valid_sequence = owner_sequence >= 0
    if not bool(valid_sequence.any()):
        raise ValueError(
            "static leaf fusion requires calibrated camera-to-sequence metadata"
        )
    owner_quality = torch.ones_like(owner_camera, dtype=torch.float32)
    quality_camera = valid_camera & (owner_camera < len(camera_quality))
    owner_quality[quality_camera] = camera_quality[
        owner_camera[quality_camera]
    ]

    # Record actual cross-sequence persistence for each envelope group.  The
    # previous implementation wrote a synthetic value of two even when all
    # evidence came from one sequence, which made an unverified hand-off look
    # persistent in downstream audits.
    maximum_sequence = max(len(sequence_names) + 1, 2)
    sequence_pair = torch.unique(
        local_groups[valid_sequence] * maximum_sequence
        + owner_sequence[valid_sequence]
        + 1
    )
    sequence_group = torch.div(
        sequence_pair, maximum_sequence, rounding_mode="floor"
    )
    sequence_count = torch.zeros(group_count, dtype=torch.int64)
    sequence_count.index_add_(
        0, sequence_group, torch.ones_like(sequence_group)
    )

    tree_ids = torch.as_tensor(payload["tree_instance_id"]).long().cpu()[
        associated_rows
    ]
    sequence_score_table: list[dict[str, int | float]] = []
    scene_sequence = -1
    if canonical_sequence_policy == "scene":
        scene_sequence, sequence_score_table = _canonical_sequence_for_scene(
            tree_ids,
            local_groups,
            owner_sequence,
            owner_camera,
            occupancy[associated_rows].clamp(0.05, 1.0) * owner_quality,
            minimum_supporting_views=minimum_supporting_views,
        )
        canonical_sequence = torch.full_like(owner_sequence, scene_sequence)
        scene_rows = valid_sequence & (owner_sequence == scene_sequence)
        chosen_sequences = {
            int(tree_id): scene_sequence
            for tree_id in torch.unique(tree_ids[scene_rows]).tolist()
        }
    else:
        canonical_sequence, chosen_sequences = _canonical_sequence_per_tree(
            tree_ids,
            owner_sequence,
            owner_camera,
            occupancy[associated_rows].clamp(0.05, 1.0) * owner_quality,
        )
    group_is_valid = torch.zeros(group_count, dtype=torch.bool)
    group_is_valid[valid_groups] = True
    canonical_row = (
        group_is_valid[local_groups]
        & valid_camera
        & valid_sequence
        & (owner_sequence == canonical_sequence)
    )
    # A tree-wide canonical acquisition may not observe every one of its
    # persistent crown cells. Dropping those cells made static foliage
    # bandwidth depend on sequence overlap rather than geometric evidence.
    # For a cell that is independently supported in at least two sequences,
    # choose its strongest internally coherent local acquisition. This does
    # not average time-varying positions and emits no render-time condition;
    # cross-sequence evidence only verifies the persistent envelope.
    fallback_sequences: dict[int, int] = {}
    if canonical_sequence_policy == "per_tree":
        represented = torch.zeros(group_count, dtype=torch.bool)
        represented[torch.unique(local_groups[canonical_row])] = True
        fallback_group_mask = (
            group_is_valid & (sequence_count >= 2) & ~represented
        )
        fallback_groups = torch.nonzero(
            fallback_group_mask, as_tuple=False
        ).flatten()
        if len(fallback_groups):
            eligible_rows = (
                valid_camera
                & valid_sequence
                & fallback_group_mask[local_groups]
            )
            eligible_groups = local_groups[eligible_rows]
            eligible_sequences = owner_sequence[eligible_rows]
            eligible_cameras = owner_camera[eligible_rows]
            eligible_weights = (
                occupancy[associated_rows][eligible_rows].clamp(0.05, 1.0)
                * owner_quality[eligible_rows]
            )
            group_sequence_key = (
                eligible_groups * maximum_sequence + eligible_sequences
            )
            unique_group_sequence, inverse_group_sequence = torch.unique(
                group_sequence_key, return_inverse=True
            )
            unique_camera_key = torch.unique(
                group_sequence_key * maximum_camera + eligible_cameras
            )
            camera_group_sequence = torch.div(
                unique_camera_key,
                maximum_camera,
                rounding_mode="floor",
            )
            camera_count = torch.bincount(
                torch.searchsorted(
                    unique_group_sequence, camera_group_sequence
                ),
                minlength=len(unique_group_sequence),
            )
            weight_sum = torch.zeros(len(unique_group_sequence))
            weight_sum.index_add_(
                0, inverse_group_sequence, eligible_weights
            )
            candidate_group = torch.div(
                unique_group_sequence,
                maximum_sequence,
                rounding_mode="floor",
            )
            candidate_sequence = unique_group_sequence.remainder(
                maximum_sequence
            )
            # Select the lexicographic maximum
            # (camera_count, weight_sum, -sequence_id) independently for
            # every group.  The former Python loop issued one scalar tensor
            # conversion and one full-table search per candidate.  A dense
            # Cambridge initialization has O(1e5) candidates, so identical
            # static evidence could take several CPU minutes to fuse before
            # iteration one.  Stable tensor sorts preserve the exact tie
            # policy while making the work O(N log N) native operations.
            candidate_order = torch.arange(
                len(unique_group_sequence), dtype=torch.int64
            )
            candidate_order = candidate_order[
                torch.argsort(
                    candidate_sequence[candidate_order], stable=True
                )
            ]
            candidate_order = candidate_order[
                torch.argsort(
                    weight_sum[candidate_order],
                    descending=True,
                    stable=True,
                )
            ]
            candidate_order = candidate_order[
                torch.argsort(
                    camera_count[candidate_order],
                    descending=True,
                    stable=True,
                )
            ]
            candidate_order = candidate_order[
                torch.argsort(
                    candidate_group[candidate_order], stable=True
                )
            ]
            ordered_candidate_group = candidate_group[candidate_order]
            first_candidate = torch.ones(
                len(candidate_order), dtype=torch.bool
            )
            first_candidate[1:] = (
                ordered_candidate_group[1:]
                != ordered_candidate_group[:-1]
            )
            selected_candidates = candidate_order[first_candidate]
            selected_fallback_groups = candidate_group[
                selected_candidates
            ]
            selected_fallback_sequences = candidate_sequence[
                selected_candidates
            ]
            fallback_sequences = dict(
                zip(
                    selected_fallback_groups.tolist(),
                    selected_fallback_sequences.tolist(),
                )
            )
            fallback_lookup = torch.full(
                (group_count,), -1, dtype=torch.int64
            )
            fallback_lookup[selected_fallback_groups] = (
                selected_fallback_sequences
            )
            canonical_row |= (
                group_is_valid[local_groups]
                & valid_camera
                & valid_sequence
                & (owner_sequence == fallback_lookup[local_groups])
            )
    canonical_rows = associated_rows[canonical_row]
    canonical_cameras = owner_camera[canonical_row]
    canonical_groups = groups[canonical_rows]
    if not len(canonical_rows):
        raise RuntimeError("canonical static view-set selection produced no rows")

    # Preserve several spatial modes from the selected snapshot.  A fixed
    # metric voxel is intentional: exact-(K) geometry is already in metres,
    # so this partition is independent of camera resolution and tree depth.
    quantized = torch.floor(
        centers[canonical_rows] / float(canonical_mode_voxel_size)
    ).to(torch.int64)
    cells, inverse = torch.unique(
        torch.cat([canonical_groups[:, None], quantized], dim=1),
        dim=0,
        return_inverse=True,
    )
    cell_count = len(cells)
    cell_weight = torch.zeros(cell_count)
    cell_weight.index_add_(
        0, inverse, occupancy[canonical_rows].clamp(0.05, 1.0)
    )
    unique_cell_camera = torch.unique(
        inverse * maximum_camera + canonical_cameras + 1
    )
    camera_cell = torch.div(
        unique_cell_camera, maximum_camera, rounding_mode="floor"
    )
    cell_view_count = torch.zeros(cell_count, dtype=torch.int64)
    cell_view_count.index_add_(
        0, camera_cell, torch.ones_like(camera_cell)
    )
    # Rank modes with the same lexicographic policy as the former per-group
    # Python dictionaries and ``sorted`` calls:
    #   multiview first, then view count, posterior mass, stable cell id.
    # Keeping the complete partition in tensors removes another O(groups)
    # interpreter loop without changing which static modes are emitted.
    cell_id = torch.arange(cell_count, dtype=torch.int64)
    cell_order = cell_id
    cell_order = cell_order[
        torch.argsort(
            cell_weight[cell_order], descending=True, stable=True
        )
    ]
    cell_order = cell_order[
        torch.argsort(
            cell_view_count[cell_order], descending=True, stable=True
        )
    ]
    cell_is_multiview = (
        cell_view_count >= int(minimum_supporting_views)
    ).to(torch.int8)
    cell_order = cell_order[
        torch.argsort(
            cell_is_multiview[cell_order],
            descending=True,
            stable=True,
        )
    ]
    cell_order = cell_order[
        torch.argsort(cells[cell_order, 0], stable=True)
    ]
    ordered_cell_group = cells[cell_order, 0]
    first_cell_in_group = torch.ones(len(cell_order), dtype=torch.bool)
    first_cell_in_group[1:] = (
        ordered_cell_group[1:] != ordered_cell_group[:-1]
    )
    group_starts = torch.nonzero(
        first_cell_in_group, as_tuple=False
    ).flatten()
    group_ends = torch.cat(
        [group_starts[1:], torch.tensor([len(cell_order)])]
    )
    group_sizes = group_ends - group_starts
    rank_within_group = torch.arange(len(cell_order)) - (
        torch.repeat_interleave(group_starts, group_sizes)
    )
    selected_cells = cell_order[
        rank_within_group < int(maximum_modes_per_group)
    ]
    selected_lookup = torch.full((cell_count,), -1, dtype=torch.int64)
    selected_lookup[selected_cells] = torch.arange(len(selected_cells))
    retained = selected_lookup[inverse] >= 0
    source_rows = canonical_rows[retained]
    source_modes = selected_lookup[inverse[retained]]
    source_groups = groups[source_rows]
    mode_groups = cells[selected_cells, 0].to(torch.int64)
    mode_count = len(mode_groups)
    if mode_count == 0:
        raise RuntimeError("canonical static mode partition produced no modes")

    base_weight = occupancy[source_rows].clamp(0.05, 1.0)

    def weighted_mode_mean(
        value: torch.Tensor, local_weight: torch.Tensor
    ) -> torch.Tensor:
        total = torch.zeros(
            (mode_count,) + tuple(value.shape[1:]), dtype=value.dtype
        )
        index = source_modes.reshape(
            (-1,) + (1,) * (value.ndim - 1)
        ).expand_as(value)
        total.scatter_add_(
            0,
            index,
            value
            * local_weight.reshape((-1,) + (1,) * (value.ndim - 1)),
        )
        denominator = torch.zeros(mode_count, dtype=local_weight.dtype)
        denominator.index_add_(0, source_modes, local_weight)
        return total / denominator.clamp_min(1e-8).reshape(
            (-1,) + (1,) * (value.ndim - 1)
        )

    parent_centers = centers[envelope_rows]
    fused_centers = weighted_mode_mean(
        centers[source_rows], base_weight
    )
    parent_scales_all = torch.as_tensor(payload["scales"]).float().cpu()[
        envelope_rows
    ]
    parent_scales = parent_scales_all[mode_groups]
    displacement = fused_centers - parent_centers[mode_groups]
    displacement_limit = 1.5 * parent_scales.max(dim=1).values.clamp_min(0.01)
    displacement_norm = displacement.norm(dim=1).clamp_min(1e-8)
    fused_centers = parent_centers[mode_groups] + displacement * (
        displacement_limit / displacement_norm
    ).clamp_max(1.0)[:, None]
    source_colors = torch.as_tensor(payload["colors"]).float().cpu()[
        source_rows
    ]
    first_color = weighted_mode_mean(source_colors, base_weight)
    color_residual = source_colors - first_color[source_modes]
    robust_color_weight = base_weight / (
        1.0 + (color_residual.norm(dim=1) / 0.20).square()
    )
    fused_colors = weighted_mode_mean(source_colors, robust_color_weight)
    parent_colors = torch.as_tensor(payload["colors"]).float().cpu()[
        envelope_rows
    ]
    parent_detail_colors = parent_colors[mode_groups]
    selected_mode_view_count = cell_view_count[selected_cells]
    color_delta = torch.where(
        selected_mode_view_count >= minimum_supporting_views,
        torch.full((mode_count,), 0.40),
        torch.full((mode_count,), 0.25),
    )[:, None]
    fused_colors = torch.maximum(
        torch.minimum(fused_colors, parent_detail_colors + color_delta),
        parent_detail_colors - color_delta,
    ).clamp(0.02, 0.98)
    fused_scales = (parent_scales * 0.45).clamp_min(0.004)
    fused_opacity = torch.full(
        (mode_count, 1), float(initial_opacity), dtype=torch.float32
    )

    # Preserve the actual canonical cameras supporting each mode instead of
    # copying an unrelated envelope table.
    mode_support_camera_ids = torch.full(
        (mode_count, support.shape[1]), -1, dtype=support.dtype
    )
    mode_camera_pairs = torch.unique(
        source_modes * maximum_camera
        + canonical_cameras[retained]
        + 1
    )
    pair_mode = torch.div(
        mode_camera_pairs, maximum_camera, rounding_mode="floor"
    )
    pair_camera = mode_camera_pairs.remainder(maximum_camera) - 1
    pair_count = torch.bincount(pair_mode, minlength=mode_count)
    pair_start = torch.cumsum(pair_count, dim=0) - pair_count
    pair_rank = torch.arange(len(pair_mode)) - torch.repeat_interleave(
        pair_start, pair_count
    )
    within_width = pair_rank < support.shape[1]
    mode_support_camera_ids[
        pair_mode[within_width], pair_rank[within_width]
    ] = pair_camera[within_width].to(support.dtype)
    parent_rows = envelope_rows[mode_groups]

    row_fields = {
        name: value.cpu()
        for name, value in payload.items()
        if torch.is_tensor(value) and value.ndim and len(value) == count
    }
    for name, value in row_fields.items():
        detail = value[parent_rows].clone()
        if name == "centers" or name == "initialization_center":
            detail = fused_centers.to(value.dtype)
        elif name == "colors":
            detail = fused_colors.to(value.dtype)
        elif name == "scales":
            detail = fused_scales.to(value.dtype)
        elif name == "opacities":
            detail = fused_opacity.to(value.dtype)
        elif name == "quaternions":
            detail = value[parent_rows].clone()
        elif name == "layer_role":
            detail.fill_(LAYER_CANONICAL_CROWN)
        elif name == "replacement_group":
            detail = mode_groups.to(value.dtype)
        elif name == "support_view_count":
            detail = view_count[mode_groups].clamp_max(
                torch.iinfo(value.dtype).max
            ).to(value.dtype)
        elif name == "support_sequence_count":
            detail = sequence_count[mode_groups].clamp_max(
                torch.iinfo(value.dtype).max
            ).to(value.dtype)
        elif name in {"track_id", "evidence_primitive_id"}:
            detail.fill_(-1)
        elif name in {"track_linearity", "static_skeleton_confidence"}:
            detail.zero_()
        elif name == "initialization_source":
            detail.fill_(7)
        elif name in {"observation_camera_ids", "support_camera_ids"}:
            detail = mode_support_camera_ids[
                :, : value.shape[1]
            ].to(value.dtype)
        elif name == "observation_uv" or name == "observation_depth":
            detail.fill_(float("nan"))
        elif name == "position_covariance":
            detail = torch.diag_embed(fused_scales.square()).to(value.dtype)
        elif name == "scale_ceiling":
            detail = (parent_scales * 0.80).clamp_min(
                fused_scales
            ).to(value.dtype)
        elif name in {
            "free_space_violation_count",
            "unknown_view_count",
            "split_generation",
        }:
            detail.zero_()
        result[name] = torch.cat([value[keep], detail], dim=0)

    static_detail = torch.zeros(int(keep.sum()) + mode_count, dtype=torch.bool)
    static_detail[int(keep.sum()) :] = True
    result["static_detail"] = static_detail
    result["replacement_overlap_ema"] = torch.zeros(len(static_detail))
    result["replacement_observation_count"] = torch.zeros(
        len(static_detail), dtype=torch.int16
    )
    result, ownerless_audit = _append_ownerless_static_ray_births(
        result,
        payload,
        minimum_supporting_views=minimum_supporting_views,
    )
    result["audit"] = {
        **dict(payload.get("audit", {})),
        "static_fusion": {
            "minimum_supporting_views": minimum_supporting_views,
            "input_dynamic_rows": int(dynamic.sum()),
            "associated_dynamic_rows": int(associated.sum()),
            "globally_verified_groups": int(len(valid_groups)),
            "camera_sequence_metadata_source": (
                camera_sequence_metadata_source
            ),
            "camera_sequence_metadata_count": int(
                len(camera_sequence_records)
            ),
            "canonical_camera_quality_source": (
                "initialization_fixed_camera_contract"
                if persisted_fixed_cameras
                else "uniform_default"
            ),
            "mapped_associated_rows": int(valid_sequence.sum()),
            "unmapped_associated_rows": int((~valid_sequence).sum()),
            "canonical_sequence_policy": canonical_sequence_policy,
            "canonical_scene_sequence": (
                sequence_names[scene_sequence]
                if scene_sequence >= 0
                else None
            ),
            "canonical_sequence_score_table": [
                {
                    **record,
                    "sequence_name": sequence_names[
                        int(record["sequence_id"])
                    ],
                }
                for record in sequence_score_table
            ],
            "canonical_sequence_tree_count": int(len(chosen_sequences)),
            "canonical_sequence_histogram": dict(
                Counter(
                    sequence_names[value]
                    for value in chosen_sequences.values()
                )
            ),
            "canonical_cross_sequence_fallback_groups": int(
                len(fallback_sequences)
            ),
            "canonical_cross_sequence_fallback_histogram": dict(
                Counter(
                    sequence_names[value]
                    for value in fallback_sequences.values()
                )
            ),
            "canonical_mode_voxel_size": float(canonical_mode_voxel_size),
            "maximum_modes_per_group": int(maximum_modes_per_group),
            "canonical_source_rows": int(len(canonical_rows)),
            "contributing_multiview_rows": int(len(source_rows)),
            "fused_static_leaf_clusters": int(mode_count),
            "multiview_canonical_modes": int(
                (
                    selected_mode_view_count
                    >= int(minimum_supporting_views)
                ).sum()
            ),
            **ownerless_audit,
        },
    }
    return result, {
        "contract": (
            "cross_view_verified_tree_cell__"
            f"{canonical_sequence_policy}_canonical_sequence__"
            "bounded_multimode_static_leaf_clusters"
        ),
        "input_dynamic_rows": int(dynamic.sum()),
        "associated_dynamic_rows": int(associated.sum()),
        "globally_verified_groups": int(len(valid_groups)),
        "camera_sequence_metadata_source": camera_sequence_metadata_source,
        "camera_sequence_metadata_count": int(
            len(camera_sequence_records)
        ),
        "canonical_camera_quality_source": (
            "initialization_fixed_camera_contract"
            if persisted_fixed_cameras
            else "uniform_default"
        ),
        "mapped_associated_rows": int(valid_sequence.sum()),
        "unmapped_associated_rows": int((~valid_sequence).sum()),
        "canonical_sequence_policy": canonical_sequence_policy,
        "canonical_scene_sequence": (
            sequence_names[scene_sequence] if scene_sequence >= 0 else None
        ),
        "canonical_sequence_score_table": [
            {
                **record,
                "sequence_name": sequence_names[int(record["sequence_id"])],
            }
            for record in sequence_score_table
        ],
        "canonical_sequence_tree_count": int(len(chosen_sequences)),
        "canonical_sequence_histogram": dict(
            Counter(
                sequence_names[value]
                for value in chosen_sequences.values()
            )
        ),
        "canonical_cross_sequence_fallback_groups": int(
            len(fallback_sequences)
        ),
        "canonical_cross_sequence_fallback_histogram": dict(
            Counter(
                sequence_names[value]
                for value in fallback_sequences.values()
            )
        ),
        "canonical_mode_voxel_size": float(canonical_mode_voxel_size),
        "maximum_modes_per_group": int(maximum_modes_per_group),
        "canonical_source_rows": int(len(canonical_rows)),
        "contributing_multiview_rows": int(len(source_rows)),
        "fused_static_leaf_clusters": int(mode_count),
        "multiview_canonical_modes": int(
            (
                selected_mode_view_count
                >= int(minimum_supporting_views)
            ).sum()
        ),
        "discarded_single_view_or_ownerless_rows": int(
            dynamic.sum() - len(source_rows)
        ),
        **ownerless_audit,
        "output_static_rows": int(len(result["static_detail"])),
        "initial_opacity": float(initial_opacity),
    }
