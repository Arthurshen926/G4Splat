"""Native jointly sorted structural surfels and volumetric foliage.

Structural primitives retain the exact perspective-correct 2DGS ray/surfel
projection. Foliage uses a true three-axis 3D EWA footprint. Both families
emit into one CUDA tile list, share one center-depth radix sort, and are
interleaved in the same front-to-back pixel loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

PRIMITIVE_HULL = 0
PRIMITIVE_SFM_TRACK = 1

LAYER_CANONICAL_CROWN = 0
LAYER_STATIC_SKELETON = 1
LAYER_DYNAMIC_LEAF = 2
VERIFICATION_UNVERIFIED = 0
VERIFICATION_VERIFIED = 1
VERIFICATION_REJECTED = 2
DYNAMIC_TEMPORAL_FALLBACK_MAX = 0.35
EXACT_RAY_RENDER_MAX_DEPTH_TO_TANGENT_RATIO = 4.0


def _identity_with_gradient_gate(
    value: torch.Tensor,
    gate: torch.Tensor | None,
) -> torch.Tensor:
    """Keep the forward value unchanged while routing its backward ownership.

    Sequence-local leaves are useful as a read-only temporal fallback in a
    neighbouring camera, but letting that camera optimize their position,
    colour and opacity destroys the exact owner-camera observation they were
    created from.  This straight-through identity preserves the rendered
    interpolation while making parameter ownership explicit.
    """
    if gate is None:
        return value
    gate = torch.as_tensor(
        gate, device=value.device, dtype=value.dtype
    ).reshape(-1)
    if len(gate) != int(value.shape[0]):
        raise ValueError("gradient gate must have one value per primitive")
    shape = (len(gate),) + (1,) * (value.ndim - 1)
    gate = gate.reshape(shape).clamp(0.0, 1.0)
    detached = value.detach()
    return detached + gate * (value - detached)


def dynamic_visibility_gate(
    foliage,
    camera_id: int,
    camera_sequence_lookup: torch.Tensor | None,
    camera_frame_lookup: torch.Tensor | None,
    *,
    support_chunk_size: int = 65_536,
) -> torch.Tensor:
    """Return the same sequence/time ownership gate in training and inference.

    A legacy state without an ownership contract must not activate every
    sequence-local primitive globally. Canonical/static primitives remain
    visible, while dynamic leaves default to invisible.
    """
    gate = torch.ones(
        len(foliage), device=foliage.xyz.device, dtype=foliage.xyz.dtype
    )
    dynamic = foliage.dynamic_leaf_mask
    if not bool(dynamic.any()):
        return gate
    if (
        camera_sequence_lookup is None
        or camera_frame_lookup is None
        or foliage.support_camera_ids.numel() == 0
        or camera_id < 0
        or camera_id >= len(camera_sequence_lookup)
        or int(camera_sequence_lookup[camera_id]) < 0
    ):
        gate[dynamic] = 0
        return gate
    current_sequence = camera_sequence_lookup[camera_id]
    current_frame = camera_frame_lookup[camera_id]
    if int(support_chunk_size) <= 0:
        raise ValueError("support_chunk_size must be positive")
    # ``support_camera_ids`` is N x K.  Expanding the complete million-leaf
    # table to int64 indices, sequence ids, frame ids and float temporal
    # weights at once added >1 GiB of short-lived CUDA tensors and OOMed before
    # the renderer itself ran.  The reduction is independent per primitive,
    # so chunking is mathematically identical and bounds peak memory by
    # O(chunk_size x K) rather than O(N x K).
    support = foliage.support_camera_ids
    exact = torch.zeros(
        len(foliage), dtype=torch.bool, device=gate.device
    )
    nearby = torch.zeros_like(gate)
    for start in range(0, len(foliage), int(support_chunk_size)):
        stop = min(start + int(support_chunk_size), len(foliage))
        support_chunk = support[start:stop]
        valid = (support_chunk >= 0) & (
            support_chunk < len(camera_sequence_lookup)
        )
        safe = support_chunk.clamp(
            0, len(camera_sequence_lookup) - 1
        ).long()
        support_sequence = camera_sequence_lookup[safe]
        support_frame = camera_frame_lookup[safe]
        exact[start:stop] = (
            support_chunk == int(camera_id)
        ).any(dim=1)
        same_sequence = valid & (
            support_sequence == current_sequence
        )
        nearby[start:stop] = torch.where(
            same_sequence,
            DYNAMIC_TEMPORAL_FALLBACK_MAX
            * torch.exp(
                -(
                    support_frame.to(gate.dtype)
                    - current_frame.to(gate.dtype)
                ).abs()
                / 12.0
            ),
            torch.zeros_like(support_frame, dtype=gate.dtype),
        ).max(dim=1).values
    # Exact observation-space births and temporal fallbacks must not remain
    # two equally strong alpha owners.  Suppress the fallback continuously
    # per physical tree as exact support becomes dense, while retaining it
    # for trees/cameras with sparse or no exact evidence.  Split-generation
    # weights conserve the original evidence count: replacing one parent by
    # two children cannot make ownership appear more certain.
    tree = foliage.tree_instance_id.long()
    valid_tree = dynamic & (tree >= 0)
    exact_tree = valid_tree & exact
    if bool(exact_tree.any()):
        tree_count = int(tree[valid_tree].max().item()) + 1
        exact_support = gate.new_zeros(tree_count)
        generation_weight = torch.pow(
            gate.new_tensor(0.5),
            foliage.split_generation.to(gate.dtype),
        )
        exact_support.scatter_add_(
            0, tree[exact_tree], generation_weight[exact_tree]
        )
        exact_coverage = 1.0 - torch.exp(-exact_support / 512.0)
        fallback_scale = 1.0 - 0.90 * exact_coverage
        nearby = nearby.clone()
        nearby[valid_tree] *= fallback_scale[tree[valid_tree]]
    replacement_group = getattr(foliage, "replacement_group", None)
    if replacement_group is not None:
        replacement_group = torch.as_tensor(
            replacement_group, device=gate.device
        ).reshape(-1)
        if len(replacement_group) != len(foliage):
            raise ValueError(
                "replacement_group must have one value per foliage primitive"
            )
        # A temporal fallback is an interpolation of a canonical cell.  When
        # no physically local canonical group exists, only the calibrated
        # owner camera may activate the leaf.  Letting such ownerless rows
        # appear in neighbouring frames created an unconserved additive fog
        # layer because neither local replacement nor optical-mass
        # conservation had a denominator for them.
        valid_group = replacement_group >= 0
        canonical_mask = getattr(
            foliage, "canonical_crown_mask", None
        )
        if canonical_mask is not None and bool(valid_group.any()):
            canonical_mask = torch.as_tensor(
                canonical_mask, device=gate.device, dtype=torch.bool
            ).reshape(-1)
            if len(canonical_mask) != len(foliage):
                raise ValueError(
                    "canonical_crown_mask must have one value per foliage "
                    "primitive"
                )
            group_count = int(replacement_group[valid_group].max()) + 1
            canonical_present = torch.zeros(
                group_count, dtype=torch.bool, device=gate.device
            )
            canonical_rows = canonical_mask & valid_group
            if bool(canonical_rows.any()):
                canonical_present[
                    replacement_group[canonical_rows].long()
                ] = True
            local_owner = valid_group & canonical_present[
                replacement_group.clamp_min(0).long()
            ]
        else:
            # Synthetic/legacy stubs without layer metadata retain their
            # historical grouped behavior. Production foliage models always
            # expose canonical_crown_mask.
            local_owner = valid_group
        nearby = nearby.clone()
        nearby[dynamic & ~local_owner] = 0
    gate[dynamic] = torch.where(exact, gate.new_tensor(1.0), nearby)[dynamic]
    return gate


def _rotation_matrices_from_quaternions(
    quaternions: torch.Tensor,
) -> torch.Tensor:
    """Return world-from-local matrices for normalized ``wxyz`` quaternions."""
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("quaternions must have shape [N, 4]")
    quaternion = torch.nn.functional.normalize(quaternions, dim=-1)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


def projected_gaussian_cross_section(
    scales: torch.Tensor,
    rotations: torch.Tensor | None = None,
    view_vectors: torch.Tensor | None = None,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Return the view-local perspective cross-section of each 3D Gaussian.

    For covariance ``Sigma`` and a unit viewing direction ``n``, the
    determinant of the covariance projected onto the plane normal to ``n``
    is ``det(Sigma) * n^T inv(Sigma) n``.  Its square root is the projected
    area scale.  Dividing by squared range supplies the perspective factor;
    the camera focal factor is common to every primitive in a render and
    therefore cancels in local optical-mass ratios.

    The legacy world-space estimate remains available when neither rotations
    nor view vectors are supplied.  Production rendering always supplies
    both, because a depth-uncertain exact-ray Gaussian must not claim optical
    area merely from its long axis along the source ray.
    """
    if scales.ndim != 2 or scales.shape[1] != 3:
        raise ValueError("scales must have shape [N, 3]")
    if (rotations is None) != (view_vectors is None):
        raise ValueError(
            "rotations and view_vectors must either both be supplied or omitted"
        )
    safe_scales = scales.clamp_min(float(epsilon))
    if rotations is None:
        return (
            safe_scales.prod(dim=1)
            / safe_scales.min(dim=1).values.clamp_min(float(epsilon))
        )
    if rotations.shape != (len(scales), 4):
        raise ValueError("rotations must have shape [N, 4]")
    if view_vectors.shape != (len(scales), 3):
        raise ValueError("view_vectors must have shape [N, 3]")
    distance = view_vectors.norm(dim=1).clamp_min(float(epsilon))
    direction = view_vectors / distance[:, None]
    rotation = _rotation_matrices_from_quaternions(rotations)
    local_direction = torch.bmm(
        rotation.transpose(1, 2), direction[:, :, None]
    ).squeeze(-1)
    inverse_variance_along_ray = (
        local_direction.square() / safe_scales.square()
    ).sum(dim=1)
    orthographic_area = (
        safe_scales.prod(dim=1)
        * inverse_variance_along_ray.clamp_min(float(epsilon)).sqrt()
    )
    return orthographic_area / distance.square()


def bounded_exact_ray_render_scales(
    foliage,
    scales: torch.Tensor,
    visibility_gate: torch.Tensor | None,
    *,
    maximum_depth_to_tangent_ratio: float = (
        EXACT_RAY_RENDER_MAX_DEPTH_TO_TANGENT_RATIO
    ),
) -> torch.Tensor:
    """Bound only the *rendered* long axis of visible exact-ray leaves.

    ``position_covariance`` remains the metric depth posterior used by the
    geometry factors.  The bound is applied to the EWA footprint only when a
    single-observation ray birth is visible.  The tangent footprint, opacity,
    colour and metric posterior remain unchanged.  This prevents uncertain
    ray depth from becoming an opaque EWA needle when a learned quaternion is
    imperfectly aligned or the leaf is viewed from another camera, without
    pretending that its 3D position became more certain.
    """
    if visibility_gate is None or not len(scales):
        return scales
    limit = float(maximum_depth_to_tangent_ratio)
    if not math.isfinite(limit) or limit <= 1.0:
        raise ValueError(
            "maximum exact-ray fallback depth/tangent ratio must be finite "
            "and greater than one"
        )
    gate = torch.as_tensor(
        visibility_gate, device=scales.device, dtype=scales.dtype
    ).reshape(-1)
    if gate.shape != (len(scales),):
        raise ValueError("visibility_gate must have one value per Gaussian")
    exact_ray = (
        foliage.dynamic_leaf_mask
        & (foliage.initialization_source == 4)
    )
    visible_exact_ray = exact_ray & (gate > 0)
    if not bool(visible_exact_ray.any()):
        return scales
    # Camera-plane quaternary splits contract both tangent axes while leaving
    # the depth axis intact.  The middle local scale is consequently a robust
    # tangent reference; unlike the minimum it is insensitive to a rare
    # binary split of only one tangent direction.  Detaching the ceiling keeps
    # photometric loss from shrinking the physical tangent footprint.
    visible_rows = torch.nonzero(
        visible_exact_ray, as_tuple=False
    ).flatten()
    visible_scales = scales[visible_rows]
    tangent_reference = (
        visible_scales.kthvalue(2, dim=1).values.detach()
    )
    ceiling = tangent_reference * limit
    bounded = scales.clone()
    bounded[visible_rows] = torch.minimum(
        visible_scales, ceiling[:, None]
    )
    return bounded


def local_optical_mass_replacement(
    layer_role: torch.Tensor,
    replacement_group: torch.Tensor,
    opacities: torch.Tensor,
    scales: torch.Tensor,
    *,
    rotations: torch.Tensor | None = None,
    view_vectors: torch.Tensor | None = None,
    cross_section: torch.Tensor | None = None,
    canonical_mask: torch.Tensor | None = None,
    detail_mask: torch.Tensor | None = None,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Measure established dynamic mass relative to its local canonical mass.

    Peak opacity alone is not a conserved quantity when a Gaussian is split:
    two children retain the parent's peak opacity while each receives half of
    its cross-sectional area.  Summing ``tau * area`` instead, where
    ``tau = -log(1 - alpha)``, preserves the local optical mass under the
    renderer's sqrt(2) three-axis split and lets all descendants contribute to
    one replacement decision.

    The returned tensor has one value per non-negative replacement group.
    Values are continuous in ``[0, 1]`` and therefore implement a gradual,
    local hand-off rather than a hard ownership gate.
    """
    roles = layer_role.reshape(-1)
    groups = replacement_group.reshape(-1).long()
    alpha = opacities.reshape(-1)
    if not (
        len(roles)
        == len(groups)
        == len(alpha)
        == int(scales.shape[0])
    ):
        raise ValueError(
            "layer roles, replacement groups, opacities and scales must "
            "describe the same number of volume Gaussians"
        )
    if scales.ndim != 2 or scales.shape[1] != 3:
        raise ValueError("scales must have shape [N, 3]")

    valid_group = groups >= 0
    if not bool(valid_group.any()):
        return alpha.new_zeros(0)
    group_count = int(groups[valid_group].max().item()) + 1
    canonical = (
        valid_group & (roles == LAYER_CANONICAL_CROWN)
        if canonical_mask is None
        else valid_group
        & torch.as_tensor(
            canonical_mask, device=roles.device, dtype=torch.bool
        ).reshape(-1)
    )
    detail = (
        valid_group & (roles == LAYER_DYNAMIC_LEAF)
        if detail_mask is None
        else valid_group
        & torch.as_tensor(
            detail_mask, device=roles.device, dtype=torch.bool
        ).reshape(-1)
    )
    if len(canonical) != len(roles) or len(detail) != len(roles):
        raise ValueError("replacement masks must match the primitive rows")
    if not bool(canonical.any()) or not bool(detail.any()):
        return alpha.new_zeros(group_count)

    safe_alpha = alpha.clamp(0.0, 1.0 - epsilon)
    optical_depth = -torch.log1p(-safe_alpha)
    if cross_section is None:
        cross_section = projected_gaussian_cross_section(
            scales,
            rotations,
            view_vectors,
            epsilon=epsilon,
        )
    else:
        cross_section = torch.as_tensor(
            cross_section, device=alpha.device, dtype=alpha.dtype
        ).reshape(-1)
        if len(cross_section) != len(alpha):
            raise ValueError(
                "cross_section must have one value per Gaussian"
            )
    optical_mass = optical_depth * cross_section

    canonical_mass = alpha.new_zeros(group_count)
    detail_mass = alpha.new_zeros(group_count)
    canonical_mass.scatter_add_(
        0, groups[canonical], optical_mass[canonical]
    )
    detail_mass.scatter_add_(0, groups[detail], optical_mass[detail])
    return (
        detail_mass
        / canonical_mass.clamp_min(epsilon)
    ).clamp(0.0, 1.0)


def replacement_group_pair_indices(
    replacement_group: torch.Tensor,
    canonical_mask: torch.Tensor,
    detail_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enumerate the sparse canonical/detail Cartesian product per group."""
    groups = replacement_group.reshape(-1).long()
    canonical = torch.as_tensor(
        canonical_mask, device=groups.device, dtype=torch.bool
    ).reshape(-1)
    detail = torch.as_tensor(
        detail_mask, device=groups.device, dtype=torch.bool
    ).reshape(-1)
    if len(canonical) != len(groups) or len(detail) != len(groups):
        raise ValueError("replacement pair masks must match group rows")
    valid = groups >= 0
    canonical_rows = torch.nonzero(
        valid & canonical, as_tuple=False
    ).flatten()
    detail_rows = torch.nonzero(valid & detail, as_tuple=False).flatten()
    if not len(canonical_rows) or not len(detail_rows):
        empty = torch.empty(0, dtype=torch.long, device=groups.device)
        return empty, empty
    group_count = int(groups[valid].max().item()) + 1
    canonical_order = torch.argsort(groups[canonical_rows], stable=True)
    detail_order = torch.argsort(groups[detail_rows], stable=True)
    canonical_sorted = canonical_rows[canonical_order]
    detail_sorted = detail_rows[detail_order]
    canonical_count = torch.bincount(
        groups[canonical_sorted], minlength=group_count
    )
    detail_count = torch.bincount(
        groups[detail_sorted], minlength=group_count
    )
    pair_count = canonical_count * detail_count
    pair_groups = torch.repeat_interleave(
        torch.arange(group_count, device=groups.device), pair_count
    )
    total_pairs = int(pair_count.sum().item())
    pair_local = torch.arange(total_pairs, device=groups.device)
    pair_start = torch.cumsum(pair_count, dim=0) - pair_count
    pair_local -= torch.repeat_interleave(pair_start, pair_count)
    canonical_start = torch.cumsum(canonical_count, dim=0) - canonical_count
    detail_start = torch.cumsum(detail_count, dim=0) - detail_count
    pair_canonical = canonical_sorted[
        canonical_start[pair_groups]
        + torch.div(
            pair_local,
            detail_count[pair_groups],
            rounding_mode="floor",
        )
    ]
    pair_detail = detail_sorted[
        detail_start[pair_groups]
        + torch.remainder(pair_local, detail_count[pair_groups])
    ]
    return pair_canonical, pair_detail


def view_depth_local_optical_replacement(
    layer_role: torch.Tensor,
    replacement_group: torch.Tensor,
    opacities: torch.Tensor,
    scales: torch.Tensor,
    projected_xy: torch.Tensor,
    camera_depth: torch.Tensor,
    *,
    focal_x: float,
    focal_y: float,
    cross_section: torch.Tensor,
    depth_radius: torch.Tensor | None = None,
    canonical_mask: torch.Tensor | None = None,
    detail_mask: torch.Tensor | None = None,
    pair_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Return per-primitive envelope retreat in the active camera.

    Replacement groups express persistent 3D locality, but a group-wide
    scalar still lets a detail observation suppress unrelated pixels and
    depths.  The previous implementation collapsed every detail mixture to a
    single centroid and average radius.  Once either branch was split, that
    statistic ceased to describe the covered image support: small canonical
    descendants saw a severe radius-ratio penalty and almost never handed
    optical mass to the detail mixture.

    This implementation evaluates the (usually sparse) canonical/detail
    Cartesian product inside each persistent group.  Detail optical mass is
    distributed over the canonical rows it actually overlaps, and no detail
    mass can be spent more than once.  The resulting attenuation is applied
    in optical-depth space, so the hand-off is invariant to Gaussian splits
    and conserves extinction instead of multiplying peak alpha heuristically.
    """
    roles = layer_role.reshape(-1)
    groups = replacement_group.reshape(-1).long()
    alpha = opacities.reshape(-1)
    xy = torch.as_tensor(
        projected_xy, device=alpha.device, dtype=alpha.dtype
    ).reshape(-1, 2)
    depth = torch.as_tensor(
        camera_depth, device=alpha.device, dtype=alpha.dtype
    ).reshape(-1)
    area = torch.as_tensor(
        cross_section, device=alpha.device, dtype=alpha.dtype
    ).reshape(-1)
    if not (
        len(roles)
        == len(groups)
        == len(alpha)
        == len(scales)
        == len(xy)
        == len(depth)
        == len(area)
    ):
        raise ValueError("view-local replacement arrays must have equal length")
    valid_group = groups >= 0
    if not bool(valid_group.any()):
        return alpha.new_zeros(len(alpha))
    group_count = int(groups[valid_group].max().item()) + 1
    canonical = (
        valid_group & (roles == LAYER_CANONICAL_CROWN)
        if canonical_mask is None
        else valid_group
        & torch.as_tensor(
            canonical_mask, device=roles.device, dtype=torch.bool
        ).reshape(-1)
    )
    detail_role = (
        roles == LAYER_DYNAMIC_LEAF
        if detail_mask is None
        else torch.as_tensor(
            detail_mask, device=roles.device, dtype=torch.bool
        ).reshape(-1)
    )
    if len(canonical) != len(roles) or len(detail_role) != len(roles):
        raise ValueError("replacement masks must match the primitive rows")
    detail = (
        valid_group
        & detail_role
        & (alpha > 0)
        & torch.isfinite(depth)
        & (depth > 0)
        & torch.isfinite(xy).all(dim=1)
    )
    if not bool(canonical.any()) or not bool(detail.any()):
        return alpha.new_zeros(len(alpha))

    optical_depth = -torch.log1p(-alpha.clamp(0.0, 1.0 - epsilon))
    mass = optical_depth * area.clamp_min(0)
    focal = float(max((float(focal_x) * float(focal_y)) ** 0.5, 1.0e-6))
    # ``area`` is a world-space cross-section.  Perspective projected radius
    # therefore contains the formerly missing 1/z factor.  Omitting it made
    # image-space locality depend on the scene's arbitrary world scale.
    projected_radius = (
        torch.sqrt(area.clamp_min(epsilon))
        * focal
        / depth.abs().clamp_min(1.0e-6)
    )
    if depth_radius is None:
        depth_radius_value = scales.max(dim=1).values
    else:
        depth_radius_value = torch.as_tensor(
            depth_radius, device=alpha.device, dtype=alpha.dtype
        ).reshape(-1)
        if len(depth_radius_value) != len(alpha):
            raise ValueError("depth_radius must have one value per Gaussian")
    if pair_indices is None:
        pair_canonical, pair_detail = replacement_group_pair_indices(
            groups, canonical, detail
        )
    else:
        pair_canonical, pair_detail = pair_indices
        pair_canonical = torch.as_tensor(
            pair_canonical, device=groups.device, dtype=torch.long
        ).reshape(-1)
        pair_detail = torch.as_tensor(
            pair_detail, device=groups.device, dtype=torch.long
        ).reshape(-1)
        if len(pair_canonical) != len(pair_detail):
            raise ValueError("replacement pair indices must have equal length")
        valid_pair = (
            canonical[pair_canonical]
            & detail[pair_detail]
            & (groups[pair_canonical] == groups[pair_detail])
        )
        pair_canonical = pair_canonical[valid_pair]
        pair_detail = pair_detail[valid_pair]
    if not len(pair_canonical):
        return alpha.new_zeros(len(alpha))

    pixel_sigma = (
        projected_radius[pair_canonical]
        + projected_radius[pair_detail]
    ).clamp_min(1.0)
    pixel_distance2 = (
        (xy[pair_canonical] - xy[pair_detail]).square().sum(dim=1)
        / pixel_sigma.square()
    )
    depth_sigma = (
        depth_radius_value[pair_canonical]
        + depth_radius_value[pair_detail]
    ).clamp_min(0.01)
    # A hit-ray detail sample owns transmittance behind its measured first-hit
    # depth.  Requiring symmetric depth coincidence kept the broad visual-hull
    # envelope alive immediately behind an exact leaf and recreated the same
    # density twice.  Conversely, a canonical sample *in front* of the hit can
    # be another real leaf, so only that direction is reduced continuously by
    # the posterior depth interval.
    canonical_in_front = (
        depth[pair_detail]
        - depth[pair_canonical]
        - 2.5 * depth_sigma
    ).clamp_min(0.0)
    depth_distance2 = canonical_in_front.square() / (
        2.5 * depth_sigma
    ).square()
    # This is deliberately asymmetric: it measures which fraction of the
    # *detail* optical mass lies inside a canonical footprint.  A broad detail
    # Gaussian can cover several split canonical descendants; its mass is
    # divided among them below, rather than being discarded once per child by
    # a symmetric radius-ratio ceiling.
    canonical_area = projected_radius[pair_canonical].square()
    detail_area = projected_radius[pair_detail].square().clamp_min(epsilon)
    detail_coverage_ceiling = (canonical_area / detail_area).clamp(max=1.0)
    overlap = torch.exp(-0.5 * (pixel_distance2 + depth_distance2))
    overlap *= detail_coverage_ceiling
    valid_projection = (
        torch.isfinite(depth[pair_canonical])
        & (depth[pair_canonical] > 0)
        & torch.isfinite(xy[pair_canonical]).all(dim=1)
        & torch.isfinite(depth[pair_detail])
        & (depth[pair_detail] > 0)
        & torch.isfinite(xy[pair_detail]).all(dim=1)
    )
    overlap = torch.where(valid_projection, overlap, torch.zeros_like(overlap))

    # Allocate each detail row's optical mass once.  When its coverage weights
    # sum to less than one, the uncovered fraction remains detail-only rather
    # than retiring unrelated canonical mass.
    detail_overlap_sum = alpha.new_zeros(len(alpha))
    detail_overlap_sum.scatter_add_(0, pair_detail, overlap)
    allocated_pair_mass = (
        mass[pair_detail]
        * overlap
        / detail_overlap_sum[pair_detail].clamp_min(1.0)
    )
    allocated_canonical_mass = alpha.new_zeros(len(alpha))
    allocated_canonical_mass.scatter_add_(
        0, pair_canonical, allocated_pair_mass
    )
    replaced_mass_fraction = (
        allocated_canonical_mass / mass.clamp_min(epsilon)
    ).clamp(0.0, 1.0)

    # Convert the conserved optical-depth reduction back to the renderer's
    # peak-alpha multiplier.  Multiplying alpha directly is not equivalent
    # once alpha is appreciable.
    target_tau = optical_depth * (1.0 - replaced_mass_fraction)
    target_alpha = -torch.expm1(-target_tau)
    suppression = alpha.new_zeros(len(alpha))
    canonical_rows = torch.nonzero(canonical, as_tuple=False).flatten()
    suppression[canonical_rows] = (
        1.0
        - target_alpha[canonical_rows]
        / alpha[canonical_rows].clamp_min(epsilon)
    )
    # Broad detail must not retire a compact envelope merely because its
    # integrated tau is large. Estimate a split-invariant group projected IoU
    # by spending every detail footprint at most once over the same pair
    # weights. The continuous ceiling guarantees replacement cannot exceed
    # 0.5 below IoU 0.2, while reaching full authority by IoU 0.4. This is the
    # executable boundary-safety invariant used by the six-layer audit.
    projected_area = projected_radius.square()
    allocated_pair_area = (
        projected_area[pair_detail]
        * overlap
        / detail_overlap_sum[pair_detail].clamp_min(1.0)
    )
    canonical_group_area = alpha.new_zeros(group_count)
    detail_group_area = alpha.new_zeros(group_count)
    intersection_group_area = alpha.new_zeros(group_count)
    canonical_group_area.scatter_add_(
        0, groups[canonical], projected_area[canonical]
    )
    detail_group_area.scatter_add_(0, groups[detail], projected_area[detail])
    intersection_group_area.scatter_add_(
        0, groups[pair_canonical], allocated_pair_area
    )
    intersection_group_area = torch.minimum(
        intersection_group_area,
        torch.minimum(canonical_group_area, detail_group_area),
    )
    union_group_area = (
        canonical_group_area
        + detail_group_area
        - intersection_group_area
    ).clamp_min(epsilon)
    projected_group_iou = intersection_group_area / union_group_area
    boundary_safe_ceiling = (2.5 * projected_group_iou).clamp(0.0, 1.0)
    suppression[canonical_rows] = torch.minimum(
        suppression[canonical_rows],
        boundary_safe_ceiling[groups[canonical_rows]],
    )
    # Until replacement is evaluated per pixel inside the mixed CUDA pass,
    # the primitive-level proxy must never perform a majority retirement in
    # one forward. This makes the audited invariant exact: replacement > 0.5
    # is impossible without a future native per-ray ownership path, avoiding
    # another broad-kernel-to-hole failure while persistent multi-view EMA and
    # local topology can continue the hand-off over subsequent observations.
    return suppression.clamp(0.0, 0.5)


def conserve_local_temporal_fallback_mass(
    layer_role: torch.Tensor,
    replacement_group: torch.Tensor,
    opacities: torch.Tensor,
    scales: torch.Tensor,
    visibility_gate: torch.Tensor,
    *,
    rotations: torch.Tensor | None = None,
    view_vectors: torch.Tensor | None = None,
    cross_section: torch.Tensor | None = None,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Prevent neighbouring-frame leaves from becoming an additive fog layer.

    Exact owner-camera leaves are direct observations and are never reduced
    here. A temporal fallback is only an interpolation of its local canonical
    cell, however, so its optical mass may fill the mass not already supplied
    by exact leaves but may not add a second unrestricted volume on top.

    The cap is evaluated per persistent replacement group in split-invariant
    optical mass ``-log(1-alpha) * cross_section``.  The resulting scale is
    detached: it is a measured ownership state, not a gradient path through
    which either branch can manipulate the normalization denominator.
    """
    roles = layer_role.reshape(-1)
    groups = replacement_group.reshape(-1).long()
    alpha = opacities.reshape(-1)
    gate = torch.as_tensor(
        visibility_gate, device=alpha.device, dtype=alpha.dtype
    ).reshape(-1)
    if not (
        len(roles)
        == len(groups)
        == len(alpha)
        == len(gate)
        == int(scales.shape[0])
    ):
        raise ValueError(
            "roles, replacement groups, opacities, scales and visibility "
            "gate must describe the same number of volume Gaussians"
        )
    if scales.ndim != 2 or scales.shape[1] != 3:
        raise ValueError("scales must have shape [N, 3]")

    valid_group = groups >= 0
    canonical = valid_group & (roles == LAYER_CANONICAL_CROWN)
    dynamic = valid_group & (roles == LAYER_DYNAMIC_LEAF)
    exact = dynamic & (gate >= 1.0 - 1.0e-6)
    fallback = dynamic & (gate > 0) & ~exact
    if not bool(canonical.any()) or not bool(fallback.any()):
        return opacities

    group_count = int(groups[valid_group].max().item()) + 1
    safe_alpha = alpha.clamp(0.0, 1.0 - epsilon)
    optical_depth = -torch.log1p(-safe_alpha)
    if cross_section is None:
        cross_section = projected_gaussian_cross_section(
            scales,
            rotations,
            view_vectors,
            epsilon=epsilon,
        )
    else:
        cross_section = torch.as_tensor(
            cross_section, device=alpha.device, dtype=alpha.dtype
        ).reshape(-1)
        if len(cross_section) != len(alpha):
            raise ValueError(
                "cross_section must have one value per Gaussian"
            )
    optical_mass = optical_depth * cross_section

    canonical_mass = alpha.new_zeros(group_count)
    exact_mass = alpha.new_zeros(group_count)
    fallback_mass = alpha.new_zeros(group_count)
    canonical_mass.scatter_add_(
        0, groups[canonical], optical_mass[canonical]
    )
    exact_mass.scatter_add_(0, groups[exact], optical_mass[exact])
    fallback_mass.scatter_add_(
        0, groups[fallback], optical_mass[fallback]
    )

    # A group can outlive its canonical row after explicit replacement.  With
    # no canonical reference there is no mass budget to infer, so leave that
    # fallback unchanged rather than turning a completed replacement into a
    # hole.
    has_reference = canonical_mass > epsilon
    allowance = (canonical_mass - exact_mass).clamp_min(0.0)
    group_scale = torch.ones_like(canonical_mass)
    bounded = has_reference & (fallback_mass > epsilon)
    group_scale[bounded] = (
        allowance[bounded] / fallback_mass[bounded]
    ).clamp(0.0, 1.0)
    group_scale = group_scale.detach()

    adjusted = alpha.clone()
    adjusted_depth = (
        optical_depth[fallback] * group_scale[groups[fallback]]
    )
    adjusted[fallback] = 1.0 - torch.exp(-adjusted_depth)
    return adjusted.reshape_as(opacities)


def _restore_sparse_volume_rows(
    value: torch.Tensor,
    structural_count: int,
    active_volume_indices: torch.Tensor,
    volume_count: int,
) -> torch.Tensor:
    """Restore mixed-rasterizer rows after exact-zero volume compaction."""
    if value.numel() == 0:
        return value
    expected = int(structural_count) + int(len(active_volume_indices))
    if int(value.shape[0]) != expected:
        raise ValueError(
            "Sparse mixed output has an unexpected primitive dimension: "
            f"{value.shape[0]} != {expected}"
        )
    volume_shape = (int(volume_count),) + tuple(value.shape[1:])
    restored_volume = value.new_zeros(volume_shape)
    if len(active_volume_indices):
        restored_volume.index_copy_(
            0,
            active_volume_indices,
            value[int(structural_count) :],
        )
    return torch.cat(
        [value[: int(structural_count)], restored_volume], dim=0
    )


@dataclass
class HybridRenderOutput:
    render: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor
    normal_world: torch.Tensor
    median_depth: torch.Tensor
    distortion: torch.Tensor
    radii: torch.Tensor
    means2d: torch.Tensor | None
    structural_count: int
    responsibility: torch.Tensor | None = None
    gate_responsibility: torch.Tensor | None = None
    surface_alpha: torch.Tensor | None = None
    volume_alpha: torch.Tensor | None = None
    surface_depth: torch.Tensor | None = None
    volume_depth: torch.Tensor | None = None
    surface_means2d: torch.Tensor | None = None
    volume_means2d: torch.Tensor | None = None
    volume_replacement: torch.Tensor | None = None
    # ``volume_replacement_candidate`` is deliberately distinct from the
    # applied state.  It is only a cheap same-tree/projected-depth candidate
    # used to focus the scheduled per-ray audit; it may never retire optical
    # mass by itself.
    volume_replacement_candidate: torch.Tensor | None = None


class VolumetricFoliageModel(nn.Module):
    """Layered foliage 3DGS with persistent geometry-evidence metadata."""

    def __init__(
        self,
        sh_degree: int,
        *,
        dynamic_rank: int = 4,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.sh_degree = int(sh_degree)
        self.dynamic_rank = int(dynamic_rank)
        device = torch.device(device)
        coefficients = (self.sh_degree + 1) ** 2
        self.xyz = nn.Parameter(torch.empty(0, 3, device=device))
        self.log_scales = nn.Parameter(torch.empty(0, 3, device=device))
        self.quaternions = nn.Parameter(torch.empty(0, 4, device=device))
        self.opacity_logits = nn.Parameter(torch.empty(0, 1, device=device))
        self.features = nn.Parameter(
            torch.empty(0, coefficients, 3, device=device)
        )
        self.deformation_basis = nn.Parameter(
            torch.empty(0, self.dynamic_rank, 3, device=device)
        )
        self.dynamic_feature_basis = nn.Parameter(
            torch.empty(0, self.dynamic_rank, 3, device=device)
        )
        self.dynamic_opacity_basis = nn.Parameter(
            torch.empty(0, self.dynamic_rank, 1, device=device)
        )
        self.register_buffer(
            "primitive_role", torch.empty(0, dtype=torch.int8, device=device)
        )
        self.register_buffer(
            "layer_role", torch.empty(0, dtype=torch.int8, device=device)
        )
        self.register_buffer(
            "track_id", torch.empty(0, dtype=torch.int64, device=device)
        )
        self.register_buffer(
            "tree_instance_id",
            torch.empty(0, dtype=torch.int32, device=device),
        )
        self.register_buffer(
            "support_camera_ids",
            torch.empty(0, 0, dtype=torch.int32, device=device),
        )
        self.register_buffer(
            "observation_camera_ids",
            torch.empty(0, 0, dtype=torch.int32, device=device),
        )
        self.register_buffer(
            "observation_uv",
            torch.empty(0, 0, 2, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "observation_depth",
            torch.empty(0, 0, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "support_view_count",
            torch.empty(0, dtype=torch.int16, device=device),
        )
        self.register_buffer(
            "support_sequence_count",
            torch.empty(0, dtype=torch.int16, device=device),
        )
        self.register_buffer(
            "split_generation",
            torch.empty(0, dtype=torch.int16, device=device),
        )
        self.register_buffer(
            "track_linearity", torch.empty(0, device=device)
        )
        self.register_buffer(
            "static_skeleton_confidence", torch.empty(0, device=device)
        )
        self.register_buffer(
            "occupancy_probability", torch.empty(0, device=device)
        )
        self.register_buffer(
            "position_covariance", torch.empty(0, 3, 3, device=device)
        )
        self.register_buffer(
            "reprojection_error", torch.empty(0, device=device)
        )
        self.register_buffer(
            "ray_depth_nll", torch.empty(0, device=device)
        )
        self.register_buffer(
            "free_space_violation_count",
            torch.empty(0, dtype=torch.int16, device=device),
        )
        self.register_buffer(
            "unknown_view_count",
            torch.empty(0, dtype=torch.int16, device=device),
        )
        self.register_buffer(
            "initialization_source",
            torch.empty(0, dtype=torch.int8, device=device),
        )
        self.register_buffer(
            "initialization_center", torch.empty(0, 3, device=device)
        )
        self.register_buffer(
            "scale_ceiling", torch.empty(0, 3, device=device)
        )
        self.register_buffer(
            "replacement_group",
            torch.empty(0, dtype=torch.int64, device=device),
        )
        # ``layer_role`` remains checkpoint-compatible: role 0 is every
        # persistent static crown primitive and role 2 is the deprecated
        # sequence-owned residual.  This orthogonal bit separates the broad
        # persistent envelope from fused multi-view static leaf clusters
        # without inventing another time-conditioned renderer role.
        self.register_buffer(
            "static_detail",
            torch.empty(0, dtype=torch.bool, device=device),
        )
        self.register_buffer(
            "replacement_overlap_ema",
            torch.empty(0, device=device),
        )
        self.register_buffer(
            "replacement_observation_count",
            torch.empty(0, dtype=torch.int16, device=device),
        )
        self.register_buffer(
            "replacement_camera_signature",
            torch.empty(0, dtype=torch.int64, device=device),
        )
        # A local optical hand-off is a persistent, reversible mass state,
        # not a renderer-only alpha multiplier.  The reference mass follows
        # legitimate hit/opacity learning; ``handoff_retired_fraction`` says
        # which part is currently supplied by verified static detail.
        self.register_buffer(
            "handoff_reference_mass", torch.empty(0, device=device)
        )
        self.register_buffer(
            "handoff_retired_fraction", torch.empty(0, device=device)
        )
        # Topology provenance and child verification are independent from
        # inherited support-camera *candidates*.  A displaced child cannot
        # claim the parent's UV/depth/evidence row as fresh proof.
        self.register_buffer(
            "lineage_id", torch.empty(0, dtype=torch.int64, device=device)
        )
        self.register_buffer(
            "parent_lineage_id",
            torch.empty(0, dtype=torch.int64, device=device),
        )
        self.register_buffer(
            "birth_iteration",
            torch.empty(0, dtype=torch.int32, device=device),
        )
        self.register_buffer(
            "verification_state",
            torch.empty(0, dtype=torch.int8, device=device),
        )
        self.register_buffer(
            "verified_camera_ids",
            torch.empty(0, 0, dtype=torch.int32, device=device),
        )
        self.register_buffer(
            "verified_camera_count",
            torch.empty(0, dtype=torch.int16, device=device),
        )
        self.register_buffer(
            "verified_sequence_count",
            torch.empty(0, dtype=torch.int16, device=device),
        )
        self.register_buffer(
            "evidence_primitive_id",
            torch.empty(0, dtype=torch.int64, device=device),
        )
        self.register_buffer(
            "candidate_evidence_primitive_id",
            torch.empty(0, dtype=torch.int64, device=device),
        )
        # Topology changes replace the metadata buffers and invalidate this
        # non-persistent acceleration cache.  Keeping the sparse same-group
        # pair list avoids sorting close to one million rows on every render.
        self._replacement_pair_cache: tuple[
            torch.Tensor, torch.Tensor
        ] | None = None

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def scales(self) -> torch.Tensor:
        return self.log_scales.exp()

    @property
    def opacities(self) -> torch.Tensor:
        return self.opacity_logits.sigmoid().squeeze(-1)

    def integrated_optical_mass(self) -> torch.Tensor:
        """Return a view-independent projected extinction mass proxy.

        Peak alpha is not conserved when the learned covariance changes.
        The product of optical depth and the two-axis Gaussian cross-section
        is.  Training snapshots this quantity before each Adam update and
        reprojects the learned mass onto the final, physically constrained
        covariance afterwards.  This keeps scale refinement from silently
        creating either holes or broad opaque curtains.
        """
        area = projected_gaussian_cross_section(self.scales)
        tau = -torch.log1p(-self.opacities.clamp(0.0, 1.0 - 1.0e-6))
        return tau * area

    @torch.no_grad()
    def restore_integrated_optical_mass(
        self,
        target_mass: torch.Tensor,
        *,
        minimum_opacity: float = 1.0e-6,
        maximum_opacity: float = 1.0 - 1.0e-6,
    ) -> dict[str, float]:
        """Compensate peak alpha after covariance/scale changes.

        ``target_mass`` may include the current step's opacity-gradient
        update, but it must be measured against the *pre-scale-update* area.
        Thus opacity still learns optical existence while scale, split and
        screen-bandwidth refinement cannot change integrated mass by accident.
        """
        target = torch.as_tensor(
            target_mass,
            device=self.xyz.device,
            dtype=self.xyz.dtype,
        ).reshape(-1)
        if target.shape != (len(self),):
            raise ValueError("target optical mass must match foliage rows")
        area = projected_gaussian_cross_section(self.scales).clamp_min(1e-12)
        tau = target.clamp_min(0.0) / area
        alpha = (-torch.expm1(-tau)).clamp(
            float(minimum_opacity), float(maximum_opacity)
        )
        before = self.opacities.detach()
        self.opacity_logits.copy_(torch.logit(alpha)[:, None])
        relative = (
            (alpha - before).abs()
            / before.clamp_min(float(minimum_opacity))
        )
        return {
            "target_mass": float(target.sum()),
            "realized_mass": float(self.integrated_optical_mass().sum()),
            "mean_relative_alpha_compensation": float(relative.mean())
            if len(relative)
            else 0.0,
            "maximum_relative_alpha_compensation": float(relative.max())
            if len(relative)
            else 0.0,
        }

    @property
    def normalized_quaternions(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self.quaternions, dim=-1)

    @property
    def static_skeleton_mask(self) -> torch.Tensor:
        return self.layer_role == LAYER_STATIC_SKELETON

    @property
    def dynamic_leaf_mask(self) -> torch.Tensor:
        return self.layer_role == LAYER_DYNAMIC_LEAF

    @property
    def canonical_crown_mask(self) -> torch.Tensor:
        return self.layer_role == LAYER_CANONICAL_CROWN

    @property
    def persistent_envelope_mask(self) -> torch.Tensor:
        return self.canonical_crown_mask & ~self.static_detail

    @property
    def static_leaf_mask(self) -> torch.Tensor:
        return self.canonical_crown_mask & self.static_detail

    @property
    def detail_leaf_mask(self) -> torch.Tensor:
        return self.static_leaf_mask | self.dynamic_leaf_mask

    def conditioned_state(
        self,
        temporal_code: torch.Tensor | None,
        *,
        include_dynamic: bool,
        base_geometry_gradient_gate: torch.Tensor | None = None,
        dynamic_geometry_gradient_gate: torch.Tensor | None = None,
        base_appearance_gradient_gate: torch.Tensor | None = None,
        dynamic_appearance_gradient_gate: torch.Tensor | None = None,
        base_opacity_gradient_gate: torch.Tensor | None = None,
        dynamic_opacity_gradient_gate: torch.Tensor | None = None,
        conditioned_visibility_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a conditioned state with explicit base/residual ownership.

        A sequence-local primitive has two physically different parameter
        families.  Its base xyz/colour/opacity belongs to the calibrated
        observation that created it, whereas its low-rank deformation,
        feature and opacity residuals are precisely the parameters intended
        to interpolate neighbouring frames.  Applying one gradient gate after
        adding those terms makes that distinction impossible: either fallback
        views corrupt the owner state or the temporal residual never learns.
        Route the two families before composition while leaving the forward
        state bit-identical.
        """
        xyz = _identity_with_gradient_gate(
            self.xyz, base_geometry_gradient_gate
        )
        features = _identity_with_gradient_gate(
            self.features, base_appearance_gradient_gate
        )
        opacity_logits = _identity_with_gradient_gate(
            self.opacity_logits.squeeze(-1),
            base_opacity_gradient_gate,
        )
        dynamic = self.dynamic_leaf_mask
        if include_dynamic and bool(dynamic.any()):
            # A sequence-local leaf stores its exact observed offset, but its
            # reference frame is the nearby canonical crown group.  Carry the
            # canonical group's learned displacement into every dynamic
            # member before applying the low-rank temporal residual.  The
            # former absolute dynamic bases evolved independently from the
            # crown, creating two incompatible geometries and broad
            # replacement halos around tree/facade boundaries.
            groups = self.replacement_group.long()
            canonical = self.canonical_crown_mask & (groups >= 0)
            dynamic_group = dynamic & (groups >= 0)
            if bool(canonical.any()) and bool(dynamic_group.any()):
                group_count = int(groups[canonical].max()) + 1
                current_sum = xyz.new_zeros(group_count, 3)
                initial_sum = xyz.new_zeros(group_count, 3)
                count = xyz.new_zeros(group_count)
                current_sum = current_sum.index_add(
                    0, groups[canonical], xyz[canonical]
                )
                initial_sum = initial_sum.index_add(
                    0,
                    groups[canonical],
                    self.initialization_center[canonical],
                )
                count = count.index_add(
                    0,
                    groups[canonical],
                    torch.ones_like(groups[canonical], dtype=xyz.dtype),
                )
                displacement = (
                    current_sum - initial_sum
                ) / count.clamp_min(1)[:, None]
                valid_dynamic = dynamic_group & (
                    groups < group_count
                )
                xyz = xyz.clone()
                xyz[valid_dynamic] = (
                    xyz[valid_dynamic]
                    + displacement[groups[valid_dynamic]]
                )
        if temporal_code is not None and include_dynamic and bool(dynamic.any()):
            code = temporal_code.to(device=xyz.device, dtype=xyz.dtype)
            if code.shape != (self.dynamic_rank,):
                raise ValueError(
                    f"temporal code must have shape {(self.dynamic_rank,)}, "
                    f"got {tuple(code.shape)}"
                )
            deformation_basis = _identity_with_gradient_gate(
                self.deformation_basis,
                dynamic_geometry_gradient_gate,
            )
            feature_basis = _identity_with_gradient_gate(
                self.dynamic_feature_basis,
                dynamic_appearance_gradient_gate,
            )
            opacity_basis = _identity_with_gradient_gate(
                self.dynamic_opacity_basis,
                dynamic_opacity_gradient_gate,
            )
            deformation_delta = torch.einsum(
                "nrc,r->nc", deformation_basis, code
            )
            feature_delta = torch.einsum(
                "nrc,r->nc", feature_basis, code
            )
            opacity_delta = torch.einsum(
                "nrc,r->nc", opacity_basis, code
            ).squeeze(-1)
            # The canonical cell itself owns the shared low-frequency motion
            # and appearance state.  Sequence-local rows remain bounded
            # birth/death residuals, but their visible temporal estimates are
            # pooled into the canonical replacement group.  This closes the
            # former half-measure where dynamic rows inherited canonical
            # displacement while the canonical row never inherited the
            # learned sequence/time deformation.
            if conditioned_visibility_gate is not None:
                visibility = torch.as_tensor(
                    conditioned_visibility_gate,
                    device=xyz.device,
                    dtype=xyz.dtype,
                ).reshape(-1).detach()
                if visibility.shape != (len(self),):
                    raise ValueError(
                        "conditioned_visibility_gate must have one value per "
                        "volume Gaussian"
                    )
                groups = self.replacement_group.long()
                canonical = self.canonical_crown_mask & (groups >= 0)
                visible_dynamic = (
                    dynamic & (groups >= 0) & (visibility > 0)
                )
                if bool(canonical.any()) and bool(visible_dynamic.any()):
                    group_count = int(groups[canonical].max()) + 1
                    visible_dynamic &= groups < group_count
                    if bool(visible_dynamic.any()):
                        group = groups[visible_dynamic]
                        weight = visibility[visible_dynamic]
                        denominator = xyz.new_zeros(group_count)
                        denominator.index_add_(0, group, weight)

                        def group_average(value: torch.Tensor) -> torch.Tensor:
                            total = value.new_zeros(
                                (group_count, value.shape[-1])
                            )
                            total.index_add_(
                                0,
                                group,
                                value[visible_dynamic] * weight[:, None],
                            )
                            return total / denominator.clamp_min(1e-8)[
                                :, None
                            ]

                        valid_canonical = canonical & (
                            denominator[groups.clamp_max(group_count - 1)] > 0
                        )
                        canonical_group = groups[valid_canonical]
                        xyz = xyz.clone()
                        xyz[valid_canonical] += group_average(
                            deformation_delta
                        )[canonical_group]
                        features = features.clone()
                        features[valid_canonical, 0] += group_average(
                            feature_delta
                        )[canonical_group]
                        opacity_logits = opacity_logits.clone()
                        opacity_group_delta = group_average(
                            opacity_delta[:, None]
                        )[:, 0]
                        opacity_logits[valid_canonical] += (
                            opacity_group_delta[canonical_group]
                        )
            xyz = xyz + deformation_delta * dynamic[:, None]
            features = features.clone()
            features[:, 0] = features[:, 0] + feature_delta * dynamic[:, None]
            opacity_logits = opacity_logits + opacity_delta * dynamic
        opacities = opacity_logits.sigmoid()
        if not include_dynamic and bool(dynamic.any()):
            opacities = opacities * (~dynamic).to(opacities.dtype)
        return xyz, features, opacities

    @torch.no_grad()
    def initialize_from_volume_state(self, payload: dict) -> int:
        """Initialize from independent SfM/visual-hull evidence."""
        if len(self):
            raise RuntimeError("Volumetric foliage is already initialized")
        if payload.get("version") != "independent_sfm_semantic_canopy_volume_v1":
            raise RuntimeError("Unsupported independent foliage volume state")
        device, dtype = self.xyz.device, self.xyz.dtype
        xyz = payload["centers"].to(device=device, dtype=dtype)
        scales = payload["scales"].to(device=device, dtype=dtype).clamp_min(1e-6)
        colors = payload["colors"].to(device=device, dtype=dtype).clamp(0, 1)
        coefficients = (self.sh_degree + 1) ** 2
        features = torch.zeros(
            len(xyz), coefficients, 3, device=device, dtype=dtype
        )
        features[:, 0] = (colors - 0.5) / 0.28209479177387814
        opacity = payload["opacities"].to(device=device, dtype=dtype)
        count = len(xyz)
        primitive_role = payload.get(
            "primitive_role", torch.zeros(count, dtype=torch.int8)
        ).to(device=device, dtype=torch.int8)
        track_linearity = payload.get(
            "track_linearity", torch.zeros(count)
        ).to(device=device, dtype=dtype)
        layer_role = payload.get("layer_role")
        if layer_role is None:
            layer_role = torch.full(
                (count,),
                LAYER_CANONICAL_CROWN,
                dtype=torch.int8,
                device=device,
            )
            skeleton = (
                (primitive_role == PRIMITIVE_SFM_TRACK)
                & (track_linearity >= 2.0)
            )
            layer_role[skeleton] = LAYER_STATIC_SKELETON
        else:
            layer_role = layer_role.to(device=device, dtype=torch.int8)
        support_camera_ids = payload.get("support_camera_ids")
        if support_camera_ids is None:
            support_camera_ids = torch.full(
                (count, 0), -1, dtype=torch.int32
            )
        observation_capacity = int(support_camera_ids.shape[1])
        observation_camera_ids = payload.get(
            "observation_camera_ids",
            torch.full(
                (count, observation_capacity), -1, dtype=torch.int32
            ),
        )
        observation_uv = payload.get(
            "observation_uv",
            torch.full(
                (count, observation_capacity, 2),
                float("nan"),
                dtype=torch.float32,
            ),
        )
        observation_depth = payload.get(
            "observation_depth",
            torch.full(
                (count, observation_capacity),
                float("nan"),
                dtype=torch.float32,
            ),
        )
        track_id = payload.get("track_id")
        if track_id is None:
            track_id = torch.full((count,), -1, dtype=torch.int64)
            track = primitive_role.cpu() == PRIMITIVE_SFM_TRACK
            track_id[track] = torch.arange(
                int(track.sum()), dtype=torch.int64
            )
        position_covariance = payload.get(
            "position_covariance", torch.diag_embed(scales.square())
        )
        self._replace(
            xyz=xyz,
            log_scales=scales.log(),
            quaternions=payload["quaternions"].to(device=device, dtype=dtype),
            opacity_logits=torch.logit(opacity.clamp(1e-6, 1 - 1e-6)),
            features=features,
            deformation_basis=torch.zeros(
                count, self.dynamic_rank, 3, device=device, dtype=dtype
            ),
            dynamic_feature_basis=torch.zeros(
                count, self.dynamic_rank, 3, device=device, dtype=dtype
            ),
            dynamic_opacity_basis=torch.zeros(
                count, self.dynamic_rank, 1, device=device, dtype=dtype
            ),
        )
        metadata = {
            "primitive_role": primitive_role,
            "layer_role": layer_role,
            "track_id": track_id.to(device=device, dtype=torch.int64),
            "tree_instance_id": payload.get(
                "tree_instance_id",
                torch.full((count,), -1, dtype=torch.int32),
            ).to(device=device, dtype=torch.int32),
            "support_camera_ids": support_camera_ids.to(
                device=device, dtype=torch.int32
            ),
            "observation_camera_ids": observation_camera_ids.to(
                device=device, dtype=torch.int32
            ),
            "observation_uv": observation_uv.to(
                device=device, dtype=dtype
            ),
            "observation_depth": observation_depth.to(
                device=device, dtype=dtype
            ),
            "support_view_count": payload.get(
                "support_view_count", torch.zeros(count, dtype=torch.int16)
            ).to(device=device, dtype=torch.int16),
            "support_sequence_count": payload.get(
                "support_sequence_count", torch.zeros(count, dtype=torch.int16)
            ).to(device=device, dtype=torch.int16),
            "split_generation": payload.get(
                "split_generation", torch.zeros(count, dtype=torch.int16)
            ).to(device=device, dtype=torch.int16),
            "track_linearity": track_linearity,
            "static_skeleton_confidence": payload.get(
                "static_skeleton_confidence", torch.zeros(count)
            ).to(device=device, dtype=dtype),
            "occupancy_probability": payload.get(
                "occupancy_probability", torch.ones(count)
            ).to(device=device, dtype=dtype),
            "position_covariance": position_covariance.to(
                device=device, dtype=dtype
            ),
            "reprojection_error": payload.get(
                "reprojection_error", torch.full((count,), float("nan"))
            ).to(device=device, dtype=dtype),
            "ray_depth_nll": payload.get(
                "ray_depth_nll", torch.zeros(count)
            ).to(device=device, dtype=dtype),
            "free_space_violation_count": payload.get(
                "free_space_violation_count",
                torch.zeros(count, dtype=torch.int16),
            ).to(device=device, dtype=torch.int16),
            "unknown_view_count": payload.get(
                "unknown_view_count",
                torch.zeros(count, dtype=torch.int16),
            ).to(device=device, dtype=torch.int16),
            "initialization_source": payload.get(
                "initialization_source", primitive_role
            ).to(device=device, dtype=torch.int8),
            "initialization_center": payload.get(
                "initialization_center", xyz
            ).to(device=device, dtype=dtype).detach().clone(),
            "scale_ceiling": payload.get(
                "scale_ceiling",
                torch.where(
                    payload.get(
                        "initialization_source", primitive_role
                    ).to(device=device, dtype=torch.int8)[:, None]
                    == 4,
                    scales * 1.10,
                    torch.full_like(scales, float("inf")),
                ),
            ).to(device=device, dtype=dtype),
            "static_detail": payload.get(
                "static_detail", torch.zeros(count, dtype=torch.bool)
            ).to(device=device, dtype=torch.bool),
            "replacement_overlap_ema": payload.get(
                "replacement_overlap_ema", torch.zeros(count)
            ).to(device=device, dtype=dtype),
            "replacement_observation_count": payload.get(
                "replacement_observation_count",
                torch.zeros(count, dtype=torch.int16),
            ).to(device=device, dtype=torch.int16),
            "replacement_camera_signature": payload.get(
                "replacement_camera_signature",
                torch.zeros(count, dtype=torch.int64),
            ).to(device=device, dtype=torch.int64),
            "handoff_reference_mass": payload.get(
                "handoff_reference_mass",
                -torch.log1p(
                    -opacity.reshape(-1).to(device=device, dtype=dtype)
                    .clamp(0.0, 1.0 - 1.0e-6)
                )
                * projected_gaussian_cross_section(scales),
            ).to(device=device, dtype=dtype),
            "handoff_retired_fraction": payload.get(
                "handoff_retired_fraction", torch.zeros(count)
            ).to(device=device, dtype=dtype),
            "lineage_id": payload.get(
                "lineage_id", torch.arange(count, dtype=torch.int64)
            ).to(device=device, dtype=torch.int64),
            "parent_lineage_id": payload.get(
                "parent_lineage_id",
                torch.full((count,), -1, dtype=torch.int64),
            ).to(device=device, dtype=torch.int64),
            "birth_iteration": payload.get(
                "birth_iteration", torch.full((count,), -1, dtype=torch.int32)
            ).to(device=device, dtype=torch.int32),
            # Seed rows and legacy checkpoints were verified before topology
            # mutation.  Only newly created rows enter the unverified state.
            "verification_state": payload.get(
                "verification_state",
                torch.full(
                    (count,), VERIFICATION_VERIFIED, dtype=torch.int8
                ),
            ).to(device=device, dtype=torch.int8),
            "verified_camera_ids": payload.get(
                "verified_camera_ids", observation_camera_ids
            ).to(device=device, dtype=torch.int32),
            "verified_camera_count": payload.get(
                "verified_camera_count",
                (observation_camera_ids >= 0).sum(dim=1).to(torch.int16),
            ).to(device=device, dtype=torch.int16),
            "verified_sequence_count": payload.get(
                "verified_sequence_count",
                payload.get(
                    "support_sequence_count",
                    torch.zeros(count, dtype=torch.int16),
                ),
            ).to(device=device, dtype=torch.int16),
        }
        ray_offsets = payload.get("ray_evidence", {}).get("offsets")
        evidence_primitive_id = torch.full(
            (count,), -1, dtype=torch.int64, device=device
        )
        if ray_offsets is not None:
            ray_primitive_count = max(int(len(ray_offsets)) - 1, 0)
            if ray_primitive_count > count:
                raise RuntimeError(
                    "Ray-evidence primitive table is larger than the "
                    "foliage seed state"
                )
            evidence_primitive_id[:ray_primitive_count] = torch.arange(
                ray_primitive_count,
                dtype=torch.int64,
                device=device,
            )
        metadata["evidence_primitive_id"] = payload.get(
            "evidence_primitive_id", evidence_primitive_id
        ).to(device=device, dtype=torch.int64)
        metadata["candidate_evidence_primitive_id"] = payload.get(
            "candidate_evidence_primitive_id",
            metadata["evidence_primitive_id"],
        ).to(device=device, dtype=torch.int64)
        static_detail = metadata["static_detail"]
        envelope_indices = torch.nonzero(
            (layer_role == LAYER_CANONICAL_CROWN) & ~static_detail,
            as_tuple=False,
        ).flatten()
        static_detail_indices = torch.nonzero(
            (layer_role == LAYER_CANONICAL_CROWN) & static_detail,
            as_tuple=False,
        ).flatten()
        dynamic_indices = torch.nonzero(
            layer_role == LAYER_DYNAMIC_LEAF, as_tuple=False
        ).flatten()
        supplied_replacement_group = payload.get("replacement_group")
        if supplied_replacement_group is None:
            # Legacy seeds did not preserve a replacement-cell identity.
            # Canonical cells can still receive stable ids, but inventing a
            # dynamic association from a coarse tree label is unsafe: a
            # single crown component can span many metres and multiple depth
            # layers. New production seeds persist the local association.
            replacement_group = torch.full(
                (count,), -1, dtype=torch.int64, device=device
            )
            replacement_group[envelope_indices] = torch.arange(
                len(envelope_indices), dtype=torch.int64, device=device
            )
        else:
            replacement_group = torch.as_tensor(
                supplied_replacement_group,
                dtype=torch.int64,
                device=device,
            ).reshape(-1)
            if len(replacement_group) != count:
                raise RuntimeError(
                    "replacement_group does not match foliage seed length"
                )
            expected_canonical = torch.arange(
                len(envelope_indices), dtype=torch.int64, device=device
            )
            if not torch.equal(
                replacement_group[envelope_indices], expected_canonical
            ):
                raise RuntimeError(
                    "Canonical replacement groups must be contiguous and "
                    "row-stable"
                )
            dynamic_groups = replacement_group[dynamic_indices]
            if bool(
                (
                    (dynamic_groups < -1)
                    | (dynamic_groups >= len(envelope_indices))
                ).any()
            ):
                raise RuntimeError(
                    "Dynamic replacement group has no canonical owner"
                )
            detail_groups = replacement_group[static_detail_indices]
            if bool(
                (
                    (detail_groups < -1)
                    | (detail_groups >= len(envelope_indices))
                ).any()
            ):
                raise RuntimeError(
                    "Static detail replacement group has no envelope owner"
                )
        metadata["replacement_group"] = replacement_group
        self._replace_buffers(**metadata)
        return int(len(xyz))

    def _replace(self, **values: torch.Tensor) -> None:
        for name, value in values.items():
            setattr(self, name, nn.Parameter(value.requires_grad_(True)))

    def _replace_buffers(self, **values: torch.Tensor) -> None:
        for name, value in values.items():
            setattr(self, name, value)
        if {"replacement_group", "layer_role", "static_detail"} & set(values):
            self._replacement_pair_cache = None

    def replacement_pair_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached sparse envelope/detail pairs for local hand-off."""
        if self._replacement_pair_cache is None:
            self._replacement_pair_cache = replacement_group_pair_indices(
                self.replacement_group,
                self.persistent_envelope_mask,
                self.detail_leaf_mask,
            )
        return self._replacement_pair_cache

    @staticmethod
    @torch.no_grad()
    def _nearest_reference_rows_bounded(
        query_xyz: torch.Tensor,
        reference_xyz: torch.Tensor,
        reference_rows: torch.Tensor,
        *,
        maximum_pairwise_entries: int = 8_388_608,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find exact nearest rows without materializing an O(NxM) matrix.

        Ray-driven birth needs only the nearest scaffold identity.  The old
        2,048 x 65,536 ``cdist`` tile occupied 512 MiB before CUDA workspace
        and failed at the first topology event on an otherwise valid shared
        GPU.  This tiled reduction preserves the same Euclidean argmin and
        first-row tie rule while bounding the pairwise tensor to 32 MiB for
        float32 at the default limit.
        """

        query_xyz = torch.as_tensor(query_xyz).reshape(-1, 3)
        reference_xyz = torch.as_tensor(reference_xyz).reshape(-1, 3)
        reference_rows = torch.as_tensor(
            reference_rows,
            device=reference_xyz.device,
            dtype=torch.long,
        ).reshape(-1)
        if query_xyz.device != reference_xyz.device:
            raise ValueError("Nearest-reference tensors must share a device")
        if not len(reference_rows):
            raise ValueError("Nearest-reference search requires candidates")
        if int(maximum_pairwise_entries) <= 0:
            raise ValueError("maximum_pairwise_entries must be positive")
        if not len(query_xyz):
            return (
                query_xyz.new_empty((0,)),
                reference_rows.new_empty((0,)),
            )

        maximum_pairwise_entries = int(maximum_pairwise_entries)
        # At most 2,048 queries per outer tile keeps the number of reference
        # launches modest for the production 2,048-birth event while still
        # covering callers with a larger diagnostic batch.
        query_batch_size = min(
            len(query_xyz), maximum_pairwise_entries, 2_048
        )
        best_distance_parts = []
        best_row_parts = []
        for query_begin in range(0, len(query_xyz), query_batch_size):
            query = query_xyz[
                query_begin : query_begin + query_batch_size
            ]
            reference_batch_size = max(
                1, maximum_pairwise_entries // max(len(query), 1)
            )
            best_distance = query.new_full((len(query),), float("inf"))
            # Match torch.min's first-candidate behavior even if every
            # distance is non-finite. Strict ``<`` below also preserves the
            # first reference row for exact finite ties across tiles.
            best_row = reference_rows[0].expand(len(query)).clone()
            for reference_begin in range(
                0, len(reference_rows), reference_batch_size
            ):
                rows = reference_rows[
                    reference_begin : reference_begin
                    + reference_batch_size
                ]
                distance = torch.cdist(query, reference_xyz[rows])
                local_distance, local_index = distance.min(dim=1)
                better = local_distance < best_distance
                best_distance[better] = local_distance[better]
                best_row[better] = rows[local_index[better]]
            best_distance_parts.append(best_distance)
            best_row_parts.append(best_row)
        return torch.cat(best_distance_parts), torch.cat(best_row_parts)

    @torch.no_grad()
    def associate_static_detail_replacement_groups(
        self,
        rows: torch.Tensor | None = None,
        *,
        query_batch_size: int = 128,
    ) -> dict[str, int | float | str]:
        """Attach unowned static detail to its nearest same-tree envelope.

        Cross-view ray births are genuine new geometry, but leaving every one
        at ``replacement_group=-1`` permanently exempts it from local optical
        conservation.  The tree instance is inherited from its nearest static
        scaffold.  Within that instance, the nearest persistent envelope cell
        supplies only a candidate identity; actual retirement still requires
        view/ray overlap in :func:`view_depth_local_optical_replacement`.
        Thus no world-distance threshold becomes a brittle visibility gate.
        """
        target = self.static_leaf_mask & (self.replacement_group < 0)
        if rows is not None:
            selected = torch.zeros_like(target)
            selected[
                torch.as_tensor(
                    rows, device=self.xyz.device, dtype=torch.long
                ).reshape(-1)
            ] = True
            target &= selected
        target_rows = torch.nonzero(target, as_tuple=False).flatten()
        owners = self.persistent_envelope_mask & (
            self.replacement_group >= 0
        )
        if not len(target_rows) or not bool(owners.any()):
            return {
                "contract": "nearest_same_tree_group_then_view_ray_overlap",
                "candidates": int(len(target_rows)),
                "assigned": 0,
                "unassigned_without_tree_owner": int(len(target_rows)),
                "mean_world_distance": 0.0,
                "maximum_world_distance": 0.0,
            }
        assigned_rows = []
        assigned_groups = []
        assigned_distances = []
        target_trees = self.tree_instance_id[target_rows].long()
        for tree_id in torch.unique(target_trees).tolist():
            if int(tree_id) < 0:
                continue
            local_targets = target_rows[target_trees == int(tree_id)]
            local_owners = torch.nonzero(
                owners & (self.tree_instance_id == int(tree_id)),
                as_tuple=False,
            ).flatten()
            if not len(local_owners):
                continue
            for begin in range(0, len(local_targets), int(query_batch_size)):
                batch = local_targets[begin : begin + int(query_batch_size)]
                nearest_distance, owner_rows = (
                    self._nearest_reference_rows_bounded(
                        self.xyz.detach()[batch],
                        self.xyz.detach(),
                        local_owners,
                    )
                )
                assigned_rows.append(batch)
                assigned_groups.append(self.replacement_group[owner_rows])
                assigned_distances.append(nearest_distance)
        if assigned_rows:
            assigned_rows_tensor = torch.cat(assigned_rows)
            assigned_groups_tensor = torch.cat(assigned_groups)
            distances = torch.cat(assigned_distances)
            self.replacement_group[assigned_rows_tensor] = (
                assigned_groups_tensor
            )
            self._replacement_pair_cache = None
            assigned_count = int(len(assigned_rows_tensor))
            mean_distance = float(distances.mean())
            maximum_distance = float(distances.max())
        else:
            assigned_count = 0
            mean_distance = 0.0
            maximum_distance = 0.0
        return {
            "contract": "nearest_same_tree_group_then_view_ray_overlap",
            "candidates": int(len(target_rows)),
            "assigned": assigned_count,
            "unassigned_without_tree_owner": int(
                len(target_rows) - assigned_count
            ),
            "mean_world_distance": mean_distance,
            "maximum_world_distance": maximum_distance,
        }

    @property
    def metadata_names(self) -> tuple[str, ...]:
        return (
            "primitive_role",
            "layer_role",
            "track_id",
            "tree_instance_id",
            "support_camera_ids",
            "observation_camera_ids",
            "observation_uv",
            "observation_depth",
            "support_view_count",
            "support_sequence_count",
            "split_generation",
            "track_linearity",
            "static_skeleton_confidence",
            "occupancy_probability",
            "position_covariance",
            "reprojection_error",
            "ray_depth_nll",
            "free_space_violation_count",
            "unknown_view_count",
            "initialization_source",
            "initialization_center",
            "scale_ceiling",
            "replacement_group",
            "static_detail",
            "replacement_overlap_ema",
            "replacement_observation_count",
            "replacement_camera_signature",
            "handoff_reference_mass",
            "handoff_retired_fraction",
            "lineage_id",
            "parent_lineage_id",
            "birth_iteration",
            "verification_state",
            "verified_camera_ids",
            "verified_camera_count",
            "verified_sequence_count",
            "evidence_primitive_id",
            "candidate_evidence_primitive_id",
        )

    @torch.no_grad()
    def append_dynamic_leaves(
        self,
        indices: torch.Tensor,
        *,
        opacity_scale: float = 0.35,
        scale_factor: float = 0.65,
        seed: int = 1701,
    ) -> int:
        """Clone evidence-backed crown/track points into a dynamic residual layer."""
        indices = torch.as_tensor(
            indices, device=self.xyz.device, dtype=torch.long
        ).unique()
        if not indices.numel():
            return 0
        generator = torch.Generator(device=self.xyz.device)
        generator.manual_seed(int(seed))
        parameter_values = {}
        for name in (
            "xyz",
            "log_scales",
            "quaternions",
            "opacity_logits",
            "features",
            "deformation_basis",
            "dynamic_feature_basis",
            "dynamic_opacity_basis",
        ):
            value = getattr(self, name).detach()
            clone = value[indices].clone()
            if name == "log_scales":
                clone = clone + math.log(float(scale_factor))
            elif name == "opacity_logits":
                opacity = clone.sigmoid() * float(opacity_scale)
                clone = torch.logit(opacity.clamp(1e-6, 1 - 1e-6))
            elif name == "deformation_basis":
                clone.normal_(0.0, 0.002, generator=generator)
            parameter_values[name] = torch.cat([value, clone], dim=0)
        metadata = {}
        for name in self.metadata_names:
            value = getattr(self, name)
            clone = value[indices].clone()
            if name == "layer_role":
                clone.fill_(LAYER_DYNAMIC_LEAF)
            elif name == "scale_ceiling":
                clone *= float(scale_factor)
            elif name == "static_detail":
                clone.fill_(False)
            elif name in {
                "replacement_overlap_ema",
                "replacement_observation_count",
                "replacement_camera_signature",
            }:
                clone.zero_()
            elif name in {
                "evidence_primitive_id",
                "candidate_evidence_primitive_id",
            }:
                # Ray/depth intervals describe the canonical evidence cell.
                # A sequence-conditioned residual must not claim that same
                # independent existence observation.
                clone.fill_(-1)
            metadata[name] = torch.cat([value, clone], dim=0)
        self._replace(**parameter_values)
        self._replace_buffers(**metadata)
        return int(indices.numel())

    @torch.no_grad()
    def append_static_ray_births(
        self,
        centers: torch.Tensor,
        *,
        colors: torch.Tensor | None = None,
        support_camera_ids: torch.Tensor | None = None,
        support_sequence_count: torch.Tensor | None = None,
        initial_scale: float = 0.04,
        initial_opacity: float = 0.02,
        birth_iteration: int = -1,
    ) -> dict[str, torch.Tensor | int]:
        """Append cross-sequence uncovered-hit consensus as static leaves."""
        centers = torch.as_tensor(
            centers, device=self.xyz.device, dtype=self.xyz.dtype
        ).reshape(-1, 3)
        if not len(centers):
            return {
                "appended": 0,
                "_new_to_old": torch.arange(
                    len(self), device=self.xyz.device
                ),
                "_new_start": len(self),
            }
        if not len(self):
            raise RuntimeError("Static ray births require a foliage scaffold")
        if initial_scale <= 0 or not 0 < initial_opacity < 1:
            raise ValueError("Static ray birth scale/opacity is invalid")
        if colors is not None:
            colors = torch.as_tensor(
                colors, device=self.xyz.device, dtype=self.xyz.dtype
            ).reshape(-1, 3)
            if len(colors) != len(centers):
                raise ValueError("Static ray-birth colors must align with centers")
        if support_camera_ids is not None:
            support_camera_ids = torch.as_tensor(
                support_camera_ids,
                device=self.xyz.device,
                dtype=torch.int32,
            ).reshape(len(centers), -1)
        if support_sequence_count is not None:
            support_sequence_count = torch.as_tensor(
                support_sequence_count,
                device=self.xyz.device,
                dtype=torch.int16,
            ).reshape(-1)
            if len(support_sequence_count) != len(centers):
                raise ValueError(
                    "Static ray-birth sequence support must align with centers"
                )
        reference_pool = torch.nonzero(
            ~self.dynamic_leaf_mask, as_tuple=False
        ).flatten()
        best_distance, parent = self._nearest_reference_rows_bounded(
            centers,
            self.xyz.detach(),
            reference_pool,
        )
        old_count = len(self)
        parameter_values = {}
        for name in (
            "xyz",
            "log_scales",
            "quaternions",
            "opacity_logits",
            "features",
            "deformation_basis",
            "dynamic_feature_basis",
            "dynamic_opacity_basis",
        ):
            value = getattr(self, name).detach()
            child = value[parent].clone()
            if name == "xyz":
                child = centers.clone()
            elif name == "log_scales":
                child.fill_(math.log(float(initial_scale)))
            elif name == "opacity_logits":
                child.fill_(torch.logit(value.new_tensor(initial_opacity)))
            elif name == "features":
                child[:, 1:] = 0
                if colors is not None:
                    valid_color = torch.isfinite(colors).all(dim=1)
                    child[valid_color, 0] = (
                        colors[valid_color].clamp(0.0, 1.0) - 0.5
                    ) / 0.28209479177387814
            elif name in {
                "deformation_basis",
                "dynamic_feature_basis",
                "dynamic_opacity_basis",
            }:
                child.zero_()
            parameter_values[name] = torch.cat([value, child], dim=0)
        metadata = {}
        for name in self.metadata_names:
            value = getattr(self, name)
            child = value[parent].clone()
            if name == "layer_role":
                child.fill_(LAYER_CANONICAL_CROWN)
            elif name == "static_detail":
                child.fill_(True)
            elif name in {
                "replacement_group",
                "track_id",
                "evidence_primitive_id",
                "candidate_evidence_primitive_id",
            }:
                child.fill_(-1)
            elif name == "lineage_id":
                next_lineage = (
                    int(self.lineage_id.max()) + 1
                    if len(self.lineage_id)
                    else 0
                )
                child = torch.arange(
                    next_lineage,
                    next_lineage + len(centers),
                    device=self.xyz.device,
                    dtype=torch.int64,
                )
            elif name == "parent_lineage_id":
                child = self.lineage_id[parent].clone()
            elif name == "birth_iteration":
                child.fill_(int(birth_iteration))
            elif name == "verification_state":
                child.fill_(VERIFICATION_UNVERIFIED)
            elif name in {
                "verified_camera_count",
                "verified_sequence_count",
            }:
                child.zero_()
            elif name == "verified_camera_ids":
                child.fill_(-1)
            elif name == "handoff_reference_mass":
                birth_scale = centers.new_full((len(centers), 3), initial_scale)
                birth_tau = -math.log1p(-float(initial_opacity))
                child = projected_gaussian_cross_section(birth_scale) * birth_tau
            elif name == "handoff_retired_fraction":
                child.zero_()
            elif name == "initialization_source":
                child.fill_(6)
            elif name == "initialization_center":
                child = centers.clone()
            elif name == "scale_ceiling":
                child.fill_(float(initial_scale) * 2.0)
            elif name == "position_covariance":
                child = torch.eye(
                    3, device=self.xyz.device, dtype=self.xyz.dtype
                )[None].repeat(len(centers), 1, 1)
                child *= float(initial_scale) ** 2
            elif name == "support_view_count":
                if support_camera_ids is None:
                    child.fill_(2)
                else:
                    child.copy_((support_camera_ids >= 0).sum(dim=1))
            elif name == "support_sequence_count":
                if support_sequence_count is None:
                    child.fill_(2)
                else:
                    child.copy_(support_sequence_count)
            elif name in {
                "split_generation",
                "free_space_violation_count",
                "unknown_view_count",
                "replacement_observation_count",
                "replacement_camera_signature",
            }:
                child.zero_()
            elif name in {
                "track_linearity",
                "static_skeleton_confidence",
                "reprojection_error",
                "ray_depth_nll",
                "replacement_overlap_ema",
            }:
                child.zero_()
            elif name == "support_camera_ids":
                child.fill_(-1)
                if support_camera_ids is not None:
                    width = min(child.shape[1], support_camera_ids.shape[1])
                    child[:, :width] = support_camera_ids[:, :width]
            elif name == "observation_camera_ids":
                child.fill_(-1)
            elif name in {"observation_uv", "observation_depth"}:
                child.fill_(float("nan"))
            metadata[name] = torch.cat([value, child], dim=0)
        new_to_old = torch.cat(
            [
                torch.arange(old_count, device=self.xyz.device),
                parent,
            ]
        )
        self._replace(**parameter_values)
        self._replace_buffers(**metadata)
        association = self.associate_static_detail_replacement_groups(
            torch.arange(old_count, len(self), device=self.xyz.device)
        )
        return {
            "appended": int(len(centers)),
            "replacement_group_association": association,
            "_new_to_old": new_to_old,
            "_new_start": old_count,
        }

    @torch.no_grad()
    def split(
        self,
        indices: torch.Tensor,
        *,
        shrink: float = math.sqrt(2.0),
        allow_static_skeleton: bool = False,
        birth_iteration: int = -1,
    ) -> dict[str, int]:
        """Replace selected volumes with two role-preserving children."""
        indices = torch.as_tensor(
            indices, device=self.xyz.device, dtype=torch.long
        ).reshape(-1)
        return self.split_adaptive(
            indices,
            torch.full_like(indices, 2),
            shrink_override=float(shrink),
            allow_static_skeleton=allow_static_skeleton,
            birth_iteration=birth_iteration,
        )

    @torch.no_grad()
    def split_adaptive(
        self,
        indices: torch.Tensor,
        child_counts: torch.Tensor,
        *,
        shrink_override: float | None = None,
        allow_static_skeleton: bool = False,
        split_plane_normals: torch.Tensor | None = None,
        birth_iteration: int = -1,
    ) -> dict[str, int]:
        """Replace selected volumes with adaptive role-preserving children."""
        remove = torch.zeros(
            len(self), dtype=torch.bool, device=self.xyz.device
        )
        return self.replace_and_split_adaptive(
            remove,
            indices,
            child_counts,
            shrink_override=shrink_override,
            allow_static_skeleton=allow_static_skeleton,
            split_plane_normals=split_plane_normals,
            birth_iteration=birth_iteration,
        )

    @torch.no_grad()
    def replace_and_split_adaptive(
        self,
        remove: torch.Tensor,
        indices: torch.Tensor,
        child_counts: torch.Tensor,
        *,
        shrink_override: float | None = None,
        allow_static_skeleton: bool = False,
        split_plane_normals: torch.Tensor | None = None,
        birth_iteration: int = -1,
    ) -> dict[str, int]:
        """Replace broad volumes with two or four oriented children.

        A fixed binary split consumes one topology event even when a
        primitive is many times wider than the requested screen bandwidth.
        Four children tile the parent's two dominant local axes and reduce
        its projected scale by two in one mutation.  ``child_count - 1`` is
        the true model-growth cost and callers budget that cost explicitly.

        The default geometric shrink is ``sqrt(child_count)``.  Each child
        therefore has ``1 / child_count`` of the parent's projected
        cross-section and retains the parent optical depth, preserving
        integrated optical mass.  Exact-camera renderer bases may additionally
        supply one camera-plane normal per parent.  Those rows are subdivided
        only in that image plane: their uncertain depth extent and posterior
        covariance are not falsely tightened by an optical-bandwidth update.
        ``shrink_override`` exists only for the legacy binary public method.
        """
        remove = torch.as_tensor(
            remove, device=self.xyz.device, dtype=torch.bool
        ).reshape(-1)
        if remove.shape != (len(self),):
            raise ValueError("replace-and-split remove mask shape mismatch")
        # A canopy contradiction is not allowed to delete the independent
        # trunk/branch owner. Static rows can still be selected explicitly as
        # split parents when ``allow_static_skeleton`` is enabled.
        remove = remove & ~self.static_skeleton_mask
        indices = torch.as_tensor(
            indices, device=self.xyz.device, dtype=torch.long
        ).reshape(-1)
        child_counts = torch.as_tensor(
            child_counts, device=self.xyz.device, dtype=torch.long
        ).reshape(-1)
        if len(indices) != len(child_counts):
            raise ValueError(
                "adaptive split child_counts must match selected parents"
            )
        if len(torch.unique(indices)) != len(indices):
            raise ValueError("adaptive split parents must be unique")
        if bool(((child_counts != 2) & (child_counts != 4)).any()):
            raise ValueError("adaptive volume splits support 2 or 4 children")
        if indices.numel() and bool(remove[indices].any()):
            raise ValueError(
                "replace-and-split parents cannot also be retired"
            )
        if split_plane_normals is None:
            parent_plane_normals = torch.full(
                (len(indices), 3),
                float("nan"),
                device=self.xyz.device,
                dtype=self.xyz.dtype,
            )
        else:
            parent_plane_normals = torch.as_tensor(
                split_plane_normals,
                device=self.xyz.device,
                dtype=self.xyz.dtype,
            ).reshape(-1, 3)
            if len(parent_plane_normals) != len(indices):
                raise ValueError(
                    "split_plane_normals must match selected parents"
                )
        if indices.numel() and not allow_static_skeleton:
            retained = ~self.static_skeleton_mask[indices]
            indices = indices[retained]
            child_counts = child_counts[retained]
            parent_plane_normals = parent_plane_normals[retained]
        if not indices.numel():
            kept = torch.nonzero(~remove, as_tuple=False).flatten()
            pruned = int(remove.sum())
            if pruned:
                self._replace(
                    **{
                        name: getattr(self, name).detach()[kept]
                        for name in (
                            "xyz",
                            "log_scales",
                            "quaternions",
                            "opacity_logits",
                            "features",
                            "deformation_basis",
                            "dynamic_feature_basis",
                            "dynamic_opacity_basis",
                        )
                    }
                )
                self._replace_buffers(
                    **{
                        name: getattr(self, name)[kept]
                        for name in self.metadata_names
                    }
                )
            return {
                "split_parents": 0,
                "children": 0,
                "net_growth": 0,
                "binary_parents": 0,
                "quaternary_parents": 0,
                "camera_plane_parents": 0,
                "pruned": pruned,
                "_new_to_old": kept,
                "_child_start": int(len(kept)),
            }
        keep = ~remove
        keep[indices] = False
        scales = self.scales.detach()[indices]
        axis_order = torch.argsort(scales, dim=-1, descending=True)
        plane_norm = parent_plane_normals.norm(dim=1)
        optical_plane_parent = (
            torch.isfinite(parent_plane_normals).all(dim=1)
            & (plane_norm > 1e-8)
        )
        child_parent = torch.repeat_interleave(
            torch.arange(len(indices), device=self.xyz.device),
            child_counts,
        )
        child_start = torch.repeat_interleave(
            torch.cumsum(child_counts, dim=0) - child_counts,
            child_counts,
        )
        child_slot = (
            torch.arange(
                int(child_counts.sum()),
                device=self.xyz.device,
            )
            - child_start
        )
        child_local = torch.zeros(
            int(child_counts.sum()),
            3,
            device=self.xyz.device,
            dtype=scales.dtype,
        )
        child_axis_order = axis_order[child_parent]
        child_scales = scales[child_parent]
        first_axis = child_axis_order[:, 0]
        second_axis = child_axis_order[:, 1]
        is_four = child_counts[child_parent] == 4
        first_sign = torch.where(
            is_four,
            torch.where(child_slot < 2, -1.0, 1.0),
            torch.where(child_slot == 0, -1.0, 1.0),
        ).to(scales.dtype)
        second_sign = torch.where(
            is_four,
            torch.where(
                (child_slot == 0) | (child_slot == 2),
                -1.0,
                1.0,
            ),
            0.0,
        ).to(scales.dtype)
        first_offset = (
            0.35
            * child_scales.gather(1, first_axis[:, None])[:, 0]
            * first_sign
        )
        second_offset = (
            0.35
            * child_scales.gather(1, second_axis[:, None])[:, 0]
            * second_sign
        )
        child_local.scatter_(
            1, first_axis[:, None], first_offset[:, None]
        )
        child_local.scatter_add_(
            1, second_axis[:, None], second_offset[:, None]
        )
        quaternion = self.normalized_quaternions.detach()[indices]
        w, x, y, z = quaternion.unbind(-1)
        rotation = torch.stack(
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
                2 * (x * y + w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z - w * x),
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ],
            dim=-1,
        ).reshape(-1, 3, 3)
        if bool(optical_plane_parent.any()):
            unit_plane_normal = torch.zeros_like(parent_plane_normals)
            unit_plane_normal[optical_plane_parent] = (
                parent_plane_normals[optical_plane_parent]
                / plane_norm[optical_plane_parent, None]
            )
            # Rotation columns are local Gaussian axes in world space.  The
            # two axes least aligned with the calibrated camera depth axis
            # span the best available image-plane subdivision directions.
            plane_alignment = torch.abs(
                torch.bmm(
                    unit_plane_normal[:, None, :], rotation
                )[:, 0]
            )
            tangent_axis_order = torch.argsort(
                plane_alignment, dim=1, descending=False
            )
            axis_order[optical_plane_parent] = tangent_axis_order[
                optical_plane_parent
            ]
            child_axis_order = axis_order[child_parent]
            first_axis = child_axis_order[:, 0]
            second_axis = child_axis_order[:, 1]
            # ``child_local`` was initially assembled in geometric
            # longest-axis order. Rebuild it after substituting the calibrated
            # image-plane order; otherwise a quaternary optical split can
            # project both nominal axes onto the same line.
            child_local.zero_()
            first_offset = (
                0.35
                * child_scales.gather(1, first_axis[:, None])[:, 0]
                * first_sign
            )
            second_offset = (
                0.35
                * child_scales.gather(1, second_axis[:, None])[:, 0]
                * second_sign
            )
            child_local.scatter_(
                1, first_axis[:, None], first_offset[:, None]
            )
            child_local.scatter_add_(
                1, second_axis[:, None], second_offset[:, None]
            )
        child_rotation = rotation[child_parent]
        offset = torch.bmm(
            child_rotation, child_local[:, :, None]
        ).squeeze(-1)
        optical_plane_child = optical_plane_parent[child_parent]
        if bool(optical_plane_child.any()):
            child_plane_normal = unit_plane_normal[child_parent]
            plane_offset = offset[optical_plane_child]
            plane_normal = child_plane_normal[optical_plane_child]
            # Projection makes the child displacement exactly depth-neutral
            # in the owner camera, even when a nearly isotropic parent's
            # quaternion is not perfectly aligned with that camera.
            projected = plane_offset - plane_normal * (
                plane_offset * plane_normal
            ).sum(dim=1, keepdim=True)
            desired_length = plane_offset.norm(dim=1, keepdim=True)
            projected_length = projected.norm(dim=1, keepdim=True)
            projected = projected * (
                desired_length / projected_length.clamp_min(1e-8)
            )
            offset[optical_plane_child] = projected
        parent_shrink = (
            torch.full_like(
                child_counts,
                float(shrink_override),
                dtype=scales.dtype,
            )
            if shrink_override is not None
            else child_counts.to(scales.dtype).sqrt()
        )
        child_shrink = parent_shrink[child_parent]
        child_axis_shrink = torch.full_like(child_scales, 1.0)
        first_axis_shrink = torch.where(
            optical_plane_child,
            torch.full_like(child_shrink, 2.0),
            child_shrink,
        )
        child_axis_shrink.scatter_(
            1, first_axis[:, None], first_axis_shrink[:, None]
        )
        child_axis_shrink.scatter_(
            1,
            second_axis[:, None],
            torch.where(
                is_four,
                child_shrink,
                torch.ones_like(child_shrink),
            )[:, None],
        )
        # A geometric split contracts all three axes.  An exact-ray optical
        # split contracts only one (binary) or two (quaternary) image-plane
        # axes and preserves its depth extent.
        child_axis_shrink[~optical_plane_child] = child_shrink[
            ~optical_plane_child, None
        ]
        parameter_values = {}
        for name in (
            "xyz",
            "log_scales",
            "quaternions",
            "opacity_logits",
            "features",
            "deformation_basis",
            "dynamic_feature_basis",
            "dynamic_opacity_basis",
        ):
            value = getattr(self, name).detach()
            child = value[indices].repeat_interleave(
                child_counts, dim=0
            )
            if name == "xyz":
                child += offset
            elif name == "log_scales":
                child -= child_axis_shrink.log()
            elif name == "opacity_logits":
                # Preserve integrated projected optical depth, not only the
                # transmittance of hypothetical coincident children. Each
                # child has 1/shrink^2 of the parent's projected area, so its
                # optical depth carries shrink^2/N of the parent.
                parent_alpha = child.sigmoid().clamp(1e-6, 1 - 1e-6)
                parent_tau = -torch.log1p(-parent_alpha)
                child_tau = (
                    parent_tau
                    * child_shrink[:, None].square()
                    / child_counts[child_parent, None].to(parent_tau.dtype)
                )
                opacity = -torch.expm1(-child_tau)
                child = torch.logit(opacity.clamp(1e-6, 1 - 1e-6))
            parameter_values[name] = torch.cat([value[keep], child], dim=0)
        metadata = {
            name: torch.cat(
                [
                    getattr(self, name)[keep],
                    getattr(self, name)[indices].repeat_interleave(
                        child_counts, dim=0
                    ),
                ],
                dim=0,
            )
            for name in self.metadata_names
        }
        child_start = int(keep.sum())
        child_xyz = parameter_values["xyz"][child_start:]
        metadata["initialization_center"][child_start:] = child_xyz.detach()
        geometric_child = ~optical_plane_child
        child_covariance = metadata["position_covariance"][child_start:]
        child_covariance[geometric_child] /= (
            child_shrink[geometric_child].square()[:, None, None]
        )
        metadata["scale_ceiling"][child_start:] /= child_axis_shrink
        # The support/observation tables below are retained only as candidate
        # owner rays for re-verification.  They are not proof at the displaced
        # child centre: the explicit state/counts are reset, the immutable
        # parent evidence id is dropped, and higher-order SH is removed.
        child_rows = slice(child_start, None)
        parent_lineage = self.lineage_id[indices].repeat_interleave(
            child_counts
        )
        next_lineage = (
            int(self.lineage_id.max()) + 1 if len(self.lineage_id) else 0
        )
        child_total = int(child_counts.sum())
        metadata["parent_lineage_id"][child_rows] = parent_lineage
        metadata["lineage_id"][child_rows] = torch.arange(
            next_lineage,
            next_lineage + child_total,
            device=self.xyz.device,
            dtype=torch.int64,
        )
        metadata["birth_iteration"][child_rows] = int(birth_iteration)
        metadata["verification_state"][child_rows] = (
            VERIFICATION_UNVERIFIED
        )
        metadata["verified_camera_ids"][child_rows] = -1
        metadata["verified_camera_count"][child_rows] = 0
        metadata["verified_sequence_count"][child_rows] = 0
        inherited_evidence = self.evidence_primitive_id[indices]
        inherited_candidate = self.candidate_evidence_primitive_id[indices]
        metadata["candidate_evidence_primitive_id"][child_rows] = torch.where(
            inherited_evidence >= 0,
            inherited_evidence,
            inherited_candidate,
        ).repeat_interleave(child_counts)
        metadata["evidence_primitive_id"][child_rows] = -1
        metadata["replacement_camera_signature"][child_rows] = 0
        metadata["replacement_overlap_ema"][child_rows] = 0
        metadata["replacement_observation_count"][child_rows] = 0
        parameter_values["features"][child_rows, 1:] = 0
        # Reference mass follows the actual conserved child mass rather than
        # duplicating the parent's scalar once per child.
        child_alpha = parameter_values["opacity_logits"][child_rows].sigmoid()
        child_mass = -torch.log1p(
            -child_alpha.squeeze(-1).clamp(0.0, 1.0 - 1.0e-6)
        ) * projected_gaussian_cross_section(
            parameter_values["log_scales"][child_rows].exp()
        )
        retained_fraction = (
            1.0 - metadata["handoff_retired_fraction"][child_rows]
        ).clamp_min(1.0e-4)
        metadata["handoff_reference_mass"][child_rows] = (
            child_mass / retained_fraction
        )
        # Discrete support remains a candidate-count prior, not verification.
        for name in (
            "support_view_count",
            "support_sequence_count",
            "free_space_violation_count",
            "unknown_view_count",
            "replacement_observation_count",
        ):
            value = metadata[name][child_start:]
            metadata[name][child_start:] = torch.div(
                value,
                child_counts[child_parent].to(value.dtype),
                rounding_mode="floor",
            )
        metadata["support_view_count"][child_start:].clamp_(min=1)
        metadata["support_sequence_count"][child_start:].clamp_(min=1)
        metadata["support_sequence_count"][child_start:] = torch.minimum(
            metadata["support_sequence_count"][child_start:],
            metadata["support_view_count"][child_start:],
        )
        metadata["split_generation"][child_start:] += 1
        metadata["occupancy_probability"][child_start:] = (
            0.5
            * metadata["occupancy_probability"][child_start:]
            + 0.25
        ).clamp(0, 1)
        new_to_old = torch.cat(
            [
                torch.nonzero(keep, as_tuple=False).flatten(),
                indices.repeat_interleave(child_counts),
            ]
        )
        self._replace(**parameter_values)
        self._replace_buffers(**metadata)
        return {
            "split_parents": int(indices.numel()),
            "children": int(child_counts.sum()),
            "net_growth": int(child_counts.sum() - len(indices)),
            "binary_parents": int((child_counts == 2).sum()),
            "quaternary_parents": int((child_counts == 4).sum()),
            "camera_plane_parents": int(optical_plane_parent.sum()),
            "pruned": int(remove.sum()),
            "_new_to_old": new_to_old,
            # Private mutation metadata lets the trainer initialize newly
            # created optical children from their immutable owner raster.
            # It is consumed before the JSON event is recorded.
            "_child_start": child_start,
        }

    @torch.no_grad()
    def prune(self, remove: torch.Tensor) -> tuple[int, torch.Tensor]:
        remove = torch.as_tensor(
            remove, device=self.xyz.device, dtype=torch.bool
        )
        if remove.shape != (len(self),):
            raise ValueError("prune mask shape mismatch")
        remove = remove & ~self.static_skeleton_mask
        keep = ~remove
        count = int(remove.sum())
        if not count:
            return 0, torch.arange(len(self), device=self.xyz.device)
        new_to_old = torch.nonzero(keep, as_tuple=False).flatten()
        self._replace(
            **{
                name: getattr(self, name).detach()[keep]
                for name in (
                    "xyz",
                    "log_scales",
                    "quaternions",
                    "opacity_logits",
                    "features",
                    "deformation_basis",
                    "dynamic_feature_basis",
                    "dynamic_opacity_basis",
                )
            }
        )
        self._replace_buffers(
            **{
                name: getattr(self, name)[keep]
                for name in self.metadata_names
            }
        )
        return count, new_to_old

    def capture(self) -> dict:
        return {
            "sh_degree": self.sh_degree,
            "dynamic_rank": self.dynamic_rank,
            "xyz": self.xyz.detach(),
            "log_scales": self.log_scales.detach(),
            "quaternions": self.quaternions.detach(),
            "opacity_logits": self.opacity_logits.detach(),
            "features": self.features.detach(),
            "deformation_basis": self.deformation_basis.detach(),
            "dynamic_feature_basis": self.dynamic_feature_basis.detach(),
            "dynamic_opacity_basis": self.dynamic_opacity_basis.detach(),
            **{
                name: getattr(self, name).detach()
                for name in self.metadata_names
            },
        }

    def restore(self, payload: dict) -> None:
        if int(payload["sh_degree"]) != self.sh_degree:
            raise RuntimeError("Foliage SH degree mismatch")
        if int(payload.get("dynamic_rank", self.dynamic_rank)) != self.dynamic_rank:
            raise RuntimeError("Foliage dynamic rank mismatch")
        device = self.xyz.device
        count = int(payload["xyz"].shape[0])
        dtype = payload["xyz"].dtype
        dynamic_defaults = {
            "deformation_basis": torch.zeros(
                count, self.dynamic_rank, 3, dtype=dtype
            ),
            "dynamic_feature_basis": torch.zeros(
                count, self.dynamic_rank, 3, dtype=dtype
            ),
            "dynamic_opacity_basis": torch.zeros(
                count, self.dynamic_rank, 1, dtype=dtype
            ),
        }
        self._replace(
            **{
                name: payload.get(name, dynamic_defaults.get(name)).to(
                    device=device
                )
                for name in (
                    "xyz",
                    "log_scales",
                    "quaternions",
                    "opacity_logits",
                    "features",
                    "deformation_basis",
                    "dynamic_feature_basis",
                    "dynamic_opacity_basis",
                )
            }
        )
        metadata_defaults = {
            "primitive_role": torch.zeros(count, dtype=torch.int8),
            "layer_role": torch.zeros(count, dtype=torch.int8),
            "track_id": torch.full((count,), -1, dtype=torch.int64),
            "tree_instance_id": torch.full(
                (count,), -1, dtype=torch.int32
            ),
            "support_camera_ids": torch.empty(count, 0, dtype=torch.int32),
            "observation_camera_ids": torch.empty(
                count, 0, dtype=torch.int32
            ),
            "observation_uv": torch.empty(count, 0, 2),
            "observation_depth": torch.empty(count, 0),
            "support_view_count": torch.zeros(count, dtype=torch.int16),
            "support_sequence_count": torch.zeros(count, dtype=torch.int16),
            "split_generation": torch.zeros(count, dtype=torch.int16),
            "track_linearity": torch.zeros(count),
            "static_skeleton_confidence": torch.zeros(count),
            "occupancy_probability": torch.ones(count),
            "position_covariance": torch.diag_embed(
                payload["log_scales"].exp().square()
            ),
            "reprojection_error": torch.full((count,), float("nan")),
            "ray_depth_nll": torch.zeros(count),
            "free_space_violation_count": torch.zeros(
                count, dtype=torch.int16
            ),
            "unknown_view_count": torch.zeros(
                count, dtype=torch.int16
            ),
            "initialization_source": torch.zeros(count, dtype=torch.int8),
            "initialization_center": payload["xyz"].clone(),
            "scale_ceiling": torch.where(
                payload.get(
                    "initialization_source",
                    torch.zeros(count, dtype=torch.int8),
                )[:, None]
                == 4,
                payload["log_scales"].exp() * 1.10,
                torch.full_like(
                    payload["log_scales"], float("inf")
                ),
            ),
            "replacement_group": torch.full(
                (count,), -1, dtype=torch.int64
            ),
            "static_detail": torch.zeros(count, dtype=torch.bool),
            "replacement_overlap_ema": torch.zeros(count),
            "replacement_observation_count": torch.zeros(
                count, dtype=torch.int16
            ),
            "replacement_camera_signature": torch.zeros(
                count, dtype=torch.int64
            ),
            "handoff_reference_mass": (
                -torch.log1p(
                    -payload["opacity_logits"].sigmoid().reshape(-1)
                    .clamp(0.0, 1.0 - 1.0e-6)
                )
                * projected_gaussian_cross_section(
                    payload["log_scales"].exp()
                )
            ),
            "handoff_retired_fraction": torch.zeros(count),
            "lineage_id": torch.arange(count, dtype=torch.int64),
            "parent_lineage_id": torch.full(
                (count,), -1, dtype=torch.int64
            ),
            "birth_iteration": torch.full(
                (count,), -1, dtype=torch.int32
            ),
            "verification_state": torch.full(
                (count,), VERIFICATION_VERIFIED, dtype=torch.int8
            ),
            "verified_camera_ids": payload.get(
                "observation_camera_ids",
                torch.empty(count, 0, dtype=torch.int32),
            ),
            "verified_camera_count": (
                payload.get(
                    "observation_camera_ids",
                    torch.empty(count, 0, dtype=torch.int32),
                )
                >= 0
            ).sum(dim=1).to(torch.int16),
            "verified_sequence_count": payload.get(
                "support_sequence_count",
                torch.zeros(count, dtype=torch.int16),
            ),
            "evidence_primitive_id": torch.full(
                (count,), -1, dtype=torch.int64
            ),
            "candidate_evidence_primitive_id": payload.get(
                "evidence_primitive_id",
                torch.full((count,), -1, dtype=torch.int64),
            ),
        }
        self._replace_buffers(
            **{
                name: payload.get(name, metadata_defaults[name]).to(
                    device=device
                )
                for name in self.metadata_names
            }
        )


def render_hybrid(
    camera,
    structural,
    foliage: VolumetricFoliageModel,
    *,
    background: torch.Tensor,
    structural_thickness_ratio: float = 0.02,
    structural_scale_ceiling: float | None = None,
    radius_clip: float = 0.0,
    surface_gate: torch.Tensor | None = None,
    surface_gate_indices: torch.Tensor | None = None,
    surface_gate_atlas: torch.Tensor | None = None,
    audit_fields: torch.Tensor | None = None,
    temporal_code: torch.Tensor | None = None,
    include_dynamic: bool = False,
    volume_means_override: torch.Tensor | None = None,
    volume_opacity_scale: float | torch.Tensor = 1.0,
    volume_role_mask: torch.Tensor | None = None,
    volume_gate: torch.Tensor | None = None,
    volume_gradient_gate: torch.Tensor | None = None,
    volume_geometry_gradient_gate: torch.Tensor | None = None,
    volume_appearance_gradient_gate: torch.Tensor | None = None,
    volume_opacity_gradient_gate: torch.Tensor | None = None,
    volume_dynamic_geometry_gradient_gate: torch.Tensor | None = None,
    volume_dynamic_appearance_gradient_gate: torch.Tensor | None = None,
    volume_dynamic_opacity_gradient_gate: torch.Tensor | None = None,
    exact_ray_render_aspect_limit: float = (
        EXACT_RAY_RENDER_MAX_DEPTH_TO_TANGENT_RATIO
    ),
    optical_replacement_policy: str = "view_depth_local",
    compact_zero_volume: bool = True,
    structural_trainable_start: int | None = None,
) -> HybridRenderOutput:
    """Rasterise exact 2D surfels and 3D volumes with native mixed CUDA.

    The structural branch is immutable by default.  ``structural_trainable_start``
    keeps that contract for the prefix while allowing an appended residual
    suffix to receive the mixed-kernel gradients.

    ``volume_gradient_gate`` remains the backwards-compatible default for all
    volume parameters.  Base geometry/appearance/opacity gates can be
    overridden independently, and the three ``volume_dynamic_*`` gates route
    only the low-rank temporal residuals.  This preserves exact observation
    ownership without making sequence interpolation read-only.
    """
    del structural_thickness_ratio, radius_clip
    if optical_replacement_policy not in {
        "view_depth_local",
        "group_projected",
        "disabled",
    }:
        raise ValueError(
            "optical_replacement_policy must be view_depth_local, "
            "group_projected or disabled"
        )
    from diff_surfel_rasterization import (
        GaussianRasterizationSettings,
        MixedGaussianRasterizer,
    )
    from utils.sh_utils import eval_sh

    structural_count = int(len(structural.get_xyz))
    structural_xyz = structural.get_xyz
    structural_tangent = structural.get_scaling
    structural_rotation = structural.get_rotation
    # A training-time Chart atlas may own geometry for a sparse subset of
    # native 2D surfels.  It returns full, row-aligned tensors so the mixed
    # CUDA kernel still sees one ordinary 2DGS array and preserves exact
    # same-tile/same-depth sorting with the 3D EWA branch.
    chart_provider = getattr(
        structural, "_chart_geometry_provider", None
    )
    if chart_provider is not None:
        chart_geometry = chart_provider.geometry_override(structural)
        if chart_geometry is not None:
            (
                structural_xyz,
                structural_tangent,
                structural_rotation,
            ) = chart_geometry
            expected = (structural_count, 3)
            if structural_xyz.shape != expected:
                raise RuntimeError(
                    "Chart geometry override no longer aligns with surface rows"
                )
    if structural_scale_ceiling is not None:
        structural_tangent = structural_tangent.clamp_max(
            float(structural_scale_ceiling)
        )
    def trainable_suffix(value: torch.Tensor) -> torch.Tensor:
        if structural_trainable_start is None:
            return value.detach()
        start = int(structural_trainable_start)
        if start < 0 or start > structural_count:
            raise ValueError(
                "structural_trainable_start must be within the structural model"
            )
        if start == 0:
            return value
        if start == structural_count:
            return value.detach()
        return torch.cat([value[:start].detach(), value[start:]], dim=0)

    surface_means = trainable_suffix(structural_xyz)
    surface_scales = trainable_suffix(structural_tangent)
    surface_rotations = trainable_suffix(structural_rotation)
    surface_base_opacity = trainable_suffix(
        structural.get_opacity
    ).reshape(-1)
    if surface_gate is None:
        surface_gate = torch.ones_like(surface_base_opacity)
    if surface_gate.shape != surface_base_opacity.shape:
        raise ValueError(
            f"surface_gate has shape {tuple(surface_gate.shape)}, expected "
            f"{tuple(surface_base_opacity.shape)}"
        )
    # Direct multiplicative gates initialize to an exactly representable 1.
    # Clamping is intentionally outside the parameter state so the parent
    # opacity remains immutable and bit-identical at zero step.
    surface_opacity = surface_base_opacity * surface_gate.clamp(0.0, 1.0)
    if surface_gate_indices is None:
        surface_gate_indices = torch.full(
            (structural_count,),
            -1,
            dtype=torch.int32,
            device=surface_means.device,
        )
    else:
        surface_gate_indices = surface_gate_indices.to(
            device=surface_means.device, dtype=torch.int32
        )
    if surface_gate_atlas is None:
        surface_gate_atlas = surface_means.new_empty((0, 0, 0))
    if surface_gate_indices.shape != (structural_count,):
        raise ValueError("surface_gate_indices must match structural count")
    geometry_gradient_gate = (
        volume_gradient_gate
        if volume_geometry_gradient_gate is None
        else volume_geometry_gradient_gate
    )
    appearance_gradient_gate = (
        volume_gradient_gate
        if volume_appearance_gradient_gate is None
        else volume_appearance_gradient_gate
    )
    opacity_gradient_gate = (
        volume_gradient_gate
        if volume_opacity_gradient_gate is None
        else volume_opacity_gradient_gate
    )
    dynamic_geometry_gradient_gate = (
        geometry_gradient_gate
        if volume_dynamic_geometry_gradient_gate is None
        else volume_dynamic_geometry_gradient_gate
    )
    dynamic_appearance_gradient_gate = (
        appearance_gradient_gate
        if volume_dynamic_appearance_gradient_gate is None
        else volume_dynamic_appearance_gradient_gate
    )
    dynamic_opacity_gradient_gate = (
        opacity_gradient_gate
        if volume_dynamic_opacity_gradient_gate is None
        else volume_dynamic_opacity_gradient_gate
    )
    volume_means, volume_features, volume_opacities = (
        foliage.conditioned_state(
            temporal_code,
            include_dynamic=include_dynamic,
            base_geometry_gradient_gate=geometry_gradient_gate,
            dynamic_geometry_gradient_gate=(
                dynamic_geometry_gradient_gate
            ),
            base_appearance_gradient_gate=appearance_gradient_gate,
            dynamic_appearance_gradient_gate=(
                dynamic_appearance_gradient_gate
            ),
            base_opacity_gradient_gate=opacity_gradient_gate,
            dynamic_opacity_gradient_gate=(
                dynamic_opacity_gradient_gate
            ),
            conditioned_visibility_gate=volume_gate,
        )
    )
    if volume_means_override is not None:
        if volume_means_override.shape != volume_means.shape:
            raise ValueError(
                "volume_means_override must match the foliage xyz shape"
            )
        volume_means = volume_means_override.to(
            device=volume_means.device, dtype=volume_means.dtype
        )
    volume_opacities = volume_opacities * torch.as_tensor(
        volume_opacity_scale,
        device=volume_opacities.device,
        dtype=volume_opacities.dtype,
    )
    volume_active = torch.ones(
        len(volume_opacities),
        dtype=torch.bool,
        device=volume_opacities.device,
    )
    if volume_role_mask is not None:
        volume_role_mask = torch.as_tensor(
            volume_role_mask,
            device=volume_opacities.device,
            dtype=torch.bool,
        ).reshape(-1)
        if volume_role_mask.shape != (len(volume_opacities),):
            raise ValueError("volume_role_mask must match foliage rows")
        volume_opacities = volume_opacities * volume_role_mask.to(
            volume_opacities.dtype
        )
        if compact_zero_volume:
            volume_active &= volume_role_mask
    if compact_zero_volume and isinstance(
        volume_opacity_scale, (int, float)
    ) and float(volume_opacity_scale) == 0.0:
        volume_active.zero_()
    if compact_zero_volume and not include_dynamic and len(volume_active):
        volume_active &= ~foliage.dynamic_leaf_mask
    if volume_gate is not None:
        volume_gate = torch.as_tensor(
            volume_gate,
            device=volume_opacities.device,
            dtype=volume_opacities.dtype,
        ).reshape(-1)
        if len(volume_gate) != len(volume_opacities):
            raise ValueError("volume_gate must have one value per volume Gaussian")
        volume_opacities = volume_opacities * volume_gate.clamp(0, 1)
        # A zero gate has exactly zero forward value and zero parameter
        # gradient.  Removing it before CUDA preprocessing is therefore
        # function- and gradient-equivalent, while avoiding projection,
        # binning and sorting of roughly one million sequence-local leaves in
        # every unrelated camera.
        if compact_zero_volume:
            volume_active &= volume_gate > 0
    volume_scales = _identity_with_gradient_gate(
        foliage.scales, geometry_gradient_gate
    )
    volume_rotations = _identity_with_gradient_gate(
        foliage.normalized_quaternions, geometry_gradient_gate
    )
    volume_scales = bounded_exact_ray_render_scales(
        foliage,
        volume_scales,
        volume_gate,
        maximum_depth_to_tangent_ratio=(
            exact_ray_render_aspect_limit
        ),
    )
    volume_view_vectors = (
        volume_means - camera.camera_center.reshape(1, 3)
    )
    replacement_per_row_full = volume_opacities.new_zeros(
        len(volume_opacities)
    )
    replacement_per_row_candidate_full = volume_opacities.new_zeros(
        len(volume_opacities)
    )
    replacement_detail = foliage.detail_leaf_mask
    replacement_envelope = foliage.persistent_envelope_mask
    replacement_active = (
        bool(foliage.static_leaf_mask.any())
        or (include_dynamic and volume_gate is not None)
    )
    if replacement_active and len(volume_opacities):
        groups = foliage.replacement_group
        valid_detail = replacement_detail & (groups >= 0)
        if bool(valid_detail.any()) and bool(replacement_envelope.any()):
            # Projected optical mass is needed only for persistent canonical
            # rows and dynamic rows visible in the current camera.  Computing
            # quaternion matrices for every sequence-local leaf (and doing it
            # once for fallback conservation and again for replacement) made
            # cost scale with the whole multi-sequence model rather than the
            # active render.  Build the exact same cross-section once on the
            # sparse ownership set and share it between both consumers.
            optical_rows = (
                (replacement_envelope & (groups >= 0))
                | (
                    valid_detail
                    & (
                        torch.ones_like(valid_detail)
                        if volume_gate is None
                        else volume_gate > 0
                    )
                )
            )
            optical_indices = torch.nonzero(
                optical_rows, as_tuple=False
            ).flatten()
            optical_cross_section = volume_scales.new_zeros(
                len(volume_scales)
            )
            optical_cross_section[optical_indices] = (
                projected_gaussian_cross_section(
                    volume_scales[optical_indices],
                    volume_rotations[optical_indices],
                    volume_view_vectors[optical_indices],
                )
            )
            # Project each anisotropic covariance onto the active camera's
            # optical axis. ``max(scale)`` is not a depth interval: for a
            # tangent-wide but depth-thin leaf it falsely overlaps geometry
            # behind the crown and suppresses neighbouring building detail.
            optical_depth_radius = volume_scales.new_zeros(
                len(volume_scales)
            )
            camera_depth_axis = camera.world_view_transform[:3, 2].to(
                device=volume_scales.device,
                dtype=volume_scales.dtype,
            )
            camera_depth_axis = camera_depth_axis / (
                camera_depth_axis.norm().clamp_min(1.0e-8)
            )
            optical_rotation = _rotation_matrices_from_quaternions(
                volume_rotations[optical_indices]
            )
            local_depth_axis = torch.bmm(
                optical_rotation.transpose(1, 2),
                camera_depth_axis.expand(len(optical_indices), -1)[
                    :, :, None
                ],
            ).squeeze(-1)
            optical_depth_radius[optical_indices] = torch.sqrt(
                (
                    local_depth_axis.square()
                    * volume_scales[optical_indices].square()
                ).sum(dim=1).clamp_min(1.0e-12)
            )
            if include_dynamic and volume_gate is not None:
                volume_opacities = conserve_local_temporal_fallback_mass(
                    foliage.layer_role,
                    groups,
                    volume_opacities,
                    volume_scales,
                    volume_gate,
                    rotations=volume_rotations,
                    view_vectors=volume_view_vectors,
                    cross_section=optical_cross_section,
                )
            if compact_zero_volume:
                volume_active &= volume_opacities.reshape(-1) > 0
            # Use a split-invariant local optical-mass ratio. A single
            # maximum opacity ignored both footprint and accumulated dynamic
            # descendants, so exact ray/RGB births remained hidden behind a
            # much larger canonical crown.
            if optical_replacement_policy == "view_depth_local":
                homogeneous = torch.cat(
                    [
                        volume_means,
                        torch.ones_like(volume_means[:, :1]),
                    ],
                    dim=1,
                )
                camera_points = homogeneous @ camera.world_view_transform.to(
                    device=volume_means.device,
                    dtype=volume_means.dtype,
                )
                camera_depth = camera_points[:, 2]
                inverse_depth = camera_depth.clamp_min(1.0e-8).reciprocal()
                projected_xy = torch.stack(
                    [
                        float(camera.focal_x)
                        * camera_points[:, 0]
                        * inverse_depth
                        + float(camera.cx),
                        float(camera.focal_y)
                        * camera_points[:, 1]
                        * inverse_depth
                        + float(camera.cy),
                    ],
                    dim=1,
                )
                replacement_per_row = view_depth_local_optical_replacement(
                    foliage.layer_role,
                    groups,
                    volume_opacities,
                    volume_scales,
                    projected_xy,
                    camera_depth,
                    focal_x=float(camera.focal_x),
                    focal_y=float(camera.focal_y),
                    cross_section=optical_cross_section,
                    depth_radius=optical_depth_radius,
                    canonical_mask=replacement_envelope,
                    detail_mask=replacement_detail,
                    pair_indices=foliage.replacement_pair_indices(),
                ).detach()
                replacement_per_row_candidate_full = replacement_per_row
                replacement = None
            elif optical_replacement_policy == "group_projected":
                replacement = local_optical_mass_replacement(
                    foliage.layer_role,
                    groups,
                    volume_opacities,
                    volume_scales,
                    rotations=volume_rotations,
                    view_vectors=volume_view_vectors,
                    cross_section=optical_cross_section,
                    canonical_mask=replacement_envelope,
                    detail_mask=replacement_detail,
                ).detach()
                replacement_per_row = None
            else:
                replacement = None
                replacement_per_row = volume_opacities.new_zeros(
                    len(volume_opacities)
                )
            # Replacement is a measured optical-mass state transition, not an
            # escape gradient. Letting gradients pass through the ratio made
            # either branch improve its loss by manipulating the ownership
            # denominator rather than its rendered colour or geometry. Dynamic
            # mass still changes the next forward hand-off after an owner-camera
            # photometric update.
            group_count = (
                len(replacement)
                if replacement is not None
                else int(groups.max().item()) + 1
            )
            valid_canonical = (
                replacement_envelope
                & (groups >= 0)
                & (groups < group_count)
            )
            volume_opacities = volume_opacities.clone()
            if replacement_per_row is not None:
                # Static production uses a persistent mass state updated only
                # from the scheduled per-pixel T_before*alpha audit.  The
                # primitive-centre proxy is merely a same-tree candidate and
                # must never dig a transient hole in the training render.
                if include_dynamic or not bool(
                    foliage.static_leaf_mask.any()
                ):
                    replacement_per_row_full = replacement_per_row
                    volume_opacities[valid_canonical] *= (
                        1.0 - replacement_per_row[valid_canonical]
                    )
            else:
                volume_opacities[valid_canonical] *= (
                    1.0 - replacement[groups[valid_canonical]]
                )
                replacement_per_row_full[valid_canonical] = replacement[
                    groups[valid_canonical]
                ]
                replacement_per_row_candidate_full[valid_canonical] = (
                    replacement[groups[valid_canonical]]
                )
    active_volume_indices = torch.nonzero(
        volume_active, as_tuple=False
    ).flatten()
    full_volume_means = volume_means
    volume_means = volume_means[active_volume_indices]
    volume_features = volume_features[active_volume_indices]
    volume_opacities = volume_opacities[active_volume_indices]
    volume_scales = volume_scales[active_volume_indices]
    volume_rotations = volume_rotations[active_volume_indices]

    def sh_colors(
        xyz: torch.Tensor,
        features: torch.Tensor,
        degree: int,
    ) -> torch.Tensor:
        if xyz.shape[0] == 0:
            return xyz.new_empty((0, 3))
        coefficients = features.transpose(1, 2).reshape(
            -1, 3, features.shape[1]
        )
        direction = xyz - camera.camera_center.reshape(1, 3)
        direction = torch.nn.functional.normalize(direction, dim=-1)
        return torch.clamp_min(eval_sh(degree, coefficients, direction) + 0.5, 0)

    degree = min(int(structural.active_sh_degree), int(foliage.sh_degree))
    surface_colors = sh_colors(
        surface_means,
        trainable_suffix(structural.get_features),
        degree,
    )
    volume_colors = sh_colors(volume_means, volume_features, degree)
    colors = torch.cat([surface_colors, volume_colors], dim=0)
    opacities = torch.cat(
        [surface_opacity, volume_opacities],
        dim=0,
    ).reshape(-1, 1)

    surface_means2D = torch.zeros_like(
        surface_means,
        requires_grad=(
            surface_gate.requires_grad
            or structural_trainable_start is not None
        ),
    )
    # Retain a full-row leaf tensor for the training/topology API.  The
    # indexed tensor passed to CUDA scatters its backward gradient into these
    # rows, while inactive exact-zero rows remain zero without any manual
    # optimizer bookkeeping.
    volume_means2D = torch.zeros_like(
        full_volume_means, requires_grad=True
    )
    active_volume_means2D = volume_means2D[active_volume_indices]
    try:
        surface_means2D.retain_grad()
        volume_means2D.retain_grad()
    except RuntimeError:
        pass
    tanfovx = math.tan(float(camera.FoVx) * 0.5)
    tanfovy = math.tan(float(camera.FoVy) * 0.5)
    settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=background,
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=degree,
        campos=camera.camera_center,
        prefiltered=False,
        debug=False,
    )
    (
        rgb,
        radii,
        allmap,
        responsibility,
        gate_responsibility,
    ) = MixedGaussianRasterizer(settings)(
        surface_means,
        surface_means2D,
        surface_scales,
        surface_rotations,
        volume_means,
        active_volume_means2D,
        volume_scales,
        volume_rotations,
        colors,
        opacities,
        surface_gate_indices=surface_gate_indices,
        surface_gate_atlas=surface_gate_atlas,
        audit_fields=audit_fields,
    )
    radii = _restore_sparse_volume_rows(
        radii,
        structural_count,
        active_volume_indices,
        len(full_volume_means),
    )
    responsibility = _restore_sparse_volume_rows(
        responsibility,
        structural_count,
        active_volume_indices,
        len(full_volume_means),
    )
    alpha_chw = allmap[1:2]
    depth = torch.nan_to_num(
        allmap[0:1] / alpha_chw.clamp_min(1e-8), 0.0, 0.0
    )
    normal_camera = allmap[2:5]
    normal_world = (
        normal_camera.permute(1, 2, 0)
        @ camera.world_view_transform[:3, :3].T
    ).permute(2, 0, 1)
    median_depth = torch.nan_to_num(allmap[5:6], 0.0, 0.0)
    surface_alpha = allmap[7:8]
    volume_alpha = allmap[8:9]
    surface_depth = torch.nan_to_num(
        allmap[9:10] / surface_alpha.clamp_min(1e-8), 0.0, 0.0
    )
    volume_depth = torch.nan_to_num(
        allmap[10:11] / volume_alpha.clamp_min(1e-8), 0.0, 0.0
    )
    return HybridRenderOutput(
        render=rgb,
        alpha=alpha_chw,
        depth=depth,
        normal_world=normal_world,
        median_depth=median_depth,
        distortion=allmap[6:7],
        radii=radii,
        means2d=torch.cat([surface_means2D, volume_means2D], dim=0),
        structural_count=structural_count,
        responsibility=(
            responsibility if responsibility.numel() else None
        ),
        gate_responsibility=(
            gate_responsibility
            if gate_responsibility.numel()
            else None
        ),
        surface_alpha=surface_alpha,
        volume_alpha=volume_alpha,
        surface_depth=surface_depth,
        volume_depth=volume_depth,
        surface_means2d=surface_means2D,
        volume_means2d=volume_means2D,
        volume_replacement=replacement_per_row_full,
        volume_replacement_candidate=(
            replacement_per_row_candidate_full
        ),
    )
