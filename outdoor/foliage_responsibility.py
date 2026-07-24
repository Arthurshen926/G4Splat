"""Contribution-normalized legacy canopy responsibility and risk audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from outdoor.foliage_view_graph import sequence_id
from outdoor.hybrid_gaussian_renderer import render_hybrid


RESPONSIBILITY_VERSION = "native_mixed_forward_contribution_audit_v2"
FIELD_NAMES = (
    "canopy",
    "rigid",
    "sky",
    "boundary",
    "canopy_residual",
)


def rigid_safe_replacement_mask(
    aggregate_rigid: torch.Tensor,
    maximum_view_rigid: torch.Tensor,
    maximum: float,
) -> torch.Tensor:
    """Protect geometry that projects onto a rigid region in any audit view."""
    return (aggregate_rigid <= float(maximum)) & (
        maximum_view_rigid <= float(maximum)
    )


@dataclass
class LegacyResponsibilityAudit:
    candidate_indices: torch.Tensor
    statistics: dict[str, torch.Tensor]
    selected_view_indices: list[int]
    selected_view_names: list[str]
    thresholds: dict[str, float | int]

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": RESPONSIBILITY_VERSION,
                "candidate_indices": self.candidate_indices.cpu(),
                "statistics": {
                    key: value.cpu() for key, value in self.statistics.items()
                },
                "selected_view_indices": self.selected_view_indices,
                "selected_view_names": self.selected_view_names,
                "thresholds": self.thresholds,
                "field_names": FIELD_NAMES,
            },
            path,
        )
        return path


def _select_diverse_views(views, task_fields, *, count: int) -> list[int]:
    records = []
    for index, view in enumerate(views):
        canopy = task_fields.canopy_fraction(
            view.image_name,
            (view.image_height, view.image_width),
        )
        records.append(
            {
                "index": index,
                "canopy": canopy,
                "sequence": sequence_id(view.image_name),
                "center": view.camera_center.detach().cpu(),
            }
        )
    ranked = sorted(records, key=lambda row: (-row["canopy"], row["index"]))
    selected: list[dict] = []
    sequence_counts: dict[str, int] = {}
    while ranked and len(selected) < int(count):
        best_position = 0
        best_score = -1.0
        for position, row in enumerate(ranked):
            if selected:
                distance = min(
                    torch.linalg.vector_norm(row["center"] - old["center"]).item()
                    for old in selected
                )
            else:
                distance = float("inf")
            diversity = min(1.0, distance / 0.5)
            novelty = 1.0 / (1.0 + sequence_counts.get(row["sequence"], 0))
            score = row["canopy"] * (0.4 + 0.6 * diversity) * novelty
            if score > best_score:
                best_score = score
                best_position = position
        chosen = ranked.pop(best_position)
        selected.append(chosen)
        sequence_counts[chosen["sequence"]] = (
            sequence_counts.get(chosen["sequence"], 0) + 1
        )
    return [int(row["index"]) for row in selected]


@torch.no_grad()
def audit_legacy_surface_responsibility(
    views,
    structural,
    empty_foliage,
    task_fields,
    *,
    background: torch.Tensor,
    view_count: int = 24,
    minimum_support_views: int = 3,
    minimum_support_sequences: int = 2,
    mandatory_view_indices: tuple[int, ...] = (),
    minimum_canopy_responsibility: float = 0.60,
    maximum_rigid_responsibility: float = 0.25,
    support_contribution: float = 0.25,
    residual_quantile: float = 0.60,
    risky_radius: int = 128,
    giant_radius: int = 256,
    near_depth: float = 1.5,
) -> LegacyResponsibilityAudit:
    """Accumulate real forward weights ``T_before * alpha`` per primitive.

    Every semantic numerator is divided by the same accumulated contribution
    denominator. A view counts as support only when the primitive contributes
    a material amount inside canopy pixels, and support sequences are counted
    independently. Rigid protection is enforced both on the aggregate and on
    every individual audit view so an unsafe projection cannot be diluted by
    otherwise canopy-dominant views.
    """
    primitive_count = int(structural.get_xyz.shape[0])
    if primitive_count == 0:
        raise RuntimeError("Cannot audit an empty structural model")
    selected = _select_diverse_views(
        views, task_fields, count=min(view_count, len(views))
    )
    selected = list(
        dict.fromkeys(
            selected
            + [
                int(index)
                for index in mandatory_view_indices
                if 0 <= int(index) < len(views)
            ]
        )
    )
    device = structural.get_xyz.device
    sums = torch.zeros(
        primitive_count, len(FIELD_NAMES) + 1, device=device
    )
    support_views = torch.zeros(
        primitive_count, dtype=torch.int16, device=device
    )
    radius_ge_risky = torch.zeros_like(support_views)
    radius_ge_giant = torch.zeros_like(support_views)
    maximum_radius = torch.zeros(
        primitive_count, dtype=torch.int32, device=device
    )
    minimum_depth = torch.full(
        (primitive_count,), float("inf"), device=device
    )
    maximum_view_rigid = torch.zeros(
        primitive_count, dtype=torch.float32, device=device
    )
    sequence_support: dict[str, torch.Tensor] = {}
    per_view_residual = []

    for index in tqdm(selected, desc="native contribution audit"):
        view = views[index]
        fields = task_fields.fields(
            view.image_name,
            (view.image_height, view.image_width),
            device,
        )
        base = render_hybrid(
            view,
            structural,
            empty_foliage,
            background=background,
        )
        target = view.original_image.to(device)
        residual = torch.sqrt(
            (base.render - target).square().mean(dim=0) + 1e-6
        )
        per_view_residual.append(
            float(
                (
                    residual * fields["p_canopy"]
                ).sum().div(fields["p_canopy"].sum().clamp_min(1.0)).item()
            )
        )
        audit_fields = torch.stack(
            [
                fields["p_canopy"],
                fields["p_rigid"],
                fields["p_sky"],
                fields["p_boundary_uncertain"],
                fields["p_canopy"] * residual,
            ],
            dim=0,
        ).contiguous()
        package = render_hybrid(
            view,
            structural,
            empty_foliage,
            background=background,
            audit_fields=audit_fields,
        )
        responsibility = package.responsibility
        if responsibility is None or responsibility.shape != sums.shape:
            raise RuntimeError("Mixed rasterizer responsibility contract failed")
        sums += responsibility
        view_contribution = responsibility[:, 0]
        view_rigid = torch.where(
            view_contribution > 0,
            responsibility[:, 2] / view_contribution.clamp_min(1e-8),
            torch.zeros_like(view_contribution),
        )
        maximum_view_rigid = torch.maximum(maximum_view_rigid, view_rigid)
        canopy_contribution = responsibility[:, 1]
        supported = canopy_contribution >= float(support_contribution)
        support_views += supported.to(torch.int16)
        sequence = sequence_id(view.image_name)
        if sequence not in sequence_support:
            sequence_support[sequence] = torch.zeros(
                primitive_count, dtype=torch.bool, device=device
            )
        sequence_support[sequence] |= supported

        radii = package.radii[:primitive_count]
        maximum_radius = torch.maximum(maximum_radius, radii)
        radius_ge_risky += (radii >= int(risky_radius)).to(torch.int16)
        radius_ge_giant += (radii >= int(giant_radius)).to(torch.int16)
        rotation = torch.as_tensor(
            view.R, device=device, dtype=structural.get_xyz.dtype
        )
        translation = torch.as_tensor(
            view.T, device=device, dtype=structural.get_xyz.dtype
        )
        depth = (structural.get_xyz.detach() @ rotation + translation)[:, 2]
        depth = torch.where(
            radii > 0, depth, torch.full_like(depth, float("inf"))
        )
        minimum_depth = torch.minimum(minimum_depth, depth)

    denominator = sums[:, 0].clamp_min(1e-8)
    normalized = sums[:, 1:] / denominator[:, None]
    support_sequences = torch.stack(
        list(sequence_support.values()), dim=0
    ).sum(dim=0).to(torch.int16)
    finite = sums[:, 0] > 0
    canopy = normalized[:, 0]
    rigid = normalized[:, 1]
    canopy_residual = normalized[:, 4]
    rigid_safe = rigid_safe_replacement_mask(
        rigid, maximum_view_rigid, maximum_rigid_responsibility
    )
    eligible_residual = canopy_residual[
        finite
        & (canopy >= float(minimum_canopy_responsibility))
        & rigid_safe
    ]
    if eligible_residual.numel():
        residual_threshold = float(
            torch.quantile(eligible_residual, float(residual_quantile)).item()
        )
    else:
        residual_threshold = float("inf")
    selected_count = max(len(selected), 1)
    risk = (
        (maximum_radius >= int(giant_radius))
        | (
            radius_ge_risky.float() / selected_count
            >= 0.05
        )
        | (minimum_depth < float(near_depth))
    )
    candidate = (
        finite
        & (canopy >= float(minimum_canopy_responsibility))
        & rigid_safe
        & (canopy_residual >= residual_threshold)
        & (support_views >= int(minimum_support_views))
        & (support_sequences >= int(minimum_support_sequences))
        & risk
    )
    indices = torch.nonzero(candidate, as_tuple=False).flatten()
    if indices.numel() == 0:
        raise RuntimeError(
            "Contribution-normalized audit found no safe replacement "
            "candidate; refusing to relax semantic/sequence gates"
        )
    statistics = {
        "total_contribution": sums[:, 0],
        "canopy_responsibility": canopy,
        "rigid_responsibility": rigid,
        "maximum_view_rigid_responsibility": maximum_view_rigid,
        "sky_responsibility": normalized[:, 2],
        "boundary_responsibility": normalized[:, 3],
        "canopy_residual": canopy_residual,
        "support_views": support_views,
        "support_sequences": support_sequences,
        "maximum_projected_radius": maximum_radius,
        "risky_radius_view_count": radius_ge_risky,
        "giant_radius_view_count": radius_ge_giant,
        "minimum_camera_depth": minimum_depth,
        "candidate": candidate,
    }
    return LegacyResponsibilityAudit(
        candidate_indices=indices,
        statistics=statistics,
        selected_view_indices=selected,
        selected_view_names=[str(views[index].image_name) for index in selected],
        thresholds={
            "minimum_support_views": int(minimum_support_views),
            "minimum_support_sequences": int(minimum_support_sequences),
            "minimum_canopy_responsibility": float(
                minimum_canopy_responsibility
            ),
            "maximum_rigid_responsibility": float(
                maximum_rigid_responsibility
            ),
            "maximum_per_view_rigid_responsibility": float(
                maximum_rigid_responsibility
            ),
            "support_contribution": float(support_contribution),
            "residual_quantile": float(residual_quantile),
            "residual_threshold": residual_threshold,
            "risky_radius": int(risky_radius),
            "giant_radius": int(giant_radius),
            "near_depth": float(near_depth),
            "selected_view_mean_canopy_residual": float(
                sum(per_view_residual) / max(len(per_view_residual), 1)
            ),
            "mandatory_view_count": len(tuple(mandatory_view_indices)),
        },
    )
