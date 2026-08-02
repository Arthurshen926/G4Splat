#!/usr/bin/env python
"""Render and evaluate the authoritative native mixed Teacher directly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SURFEL_ROOT = REPO_ROOT / "2d-gaussian-splatting"
sys.path[:0] = [str(REPO_ROOT), str(SURFEL_ROOT)]

from arguments import ModelParams  # noqa: E402
from outdoor.hybrid_teacher_api import (  # noqa: E402
    load_hybrid_teacher,
    teacher_branch_validity,
)
from outdoor.evidence_store import sha256_file  # noqa: E402
from outdoor.lazy_scene import LazyScene, rgb_source_contract  # noqa: E402
from outdoor.scene_contract import validate_disjoint_contracts  # noqa: E402
from matcha.cambridge_masks import CambridgeMaskLookup  # noqa: E402
from scene import GaussianModel  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402


def _region(prediction, target, weight):
    weight = weight.clamp(0, 1)
    mass = weight.sum().clamp_min(1)
    mae = ((prediction - target).abs().mean(0) * weight).sum() / mass
    mse = ((prediction - target).square().mean(0) * weight).sum() / mass
    return {
        "psnr": float(-10 * torch.log10(mse.clamp_min(1e-12))),
        "mae": float(mae),
    }


def _protocol_metrics(prediction, target, mask=None):
    """Historical Cambridge rgb_metrics_v3 scalar definition."""
    if mask is not None:
        if not bool(mask.any()):
            return None
        prediction = prediction[:, mask]
        target = target[:, mask]
    residual = prediction - target
    mse = residual.square().mean().clamp_min(1e-12)
    c1, c2 = 0.01**2, 0.03**2
    mean_prediction = prediction.mean()
    mean_target = target.mean()
    variance_prediction = (
        (prediction - mean_prediction).square().mean()
    )
    variance_target = (target - mean_target).square().mean()
    covariance = (
        (prediction - mean_prediction) * (target - mean_target)
    ).mean()
    global_ssim = (
        (2 * mean_prediction * mean_target + c1)
        * (2 * covariance + c2)
        / (
            (mean_prediction.square() + mean_target.square() + c1)
            * (variance_prediction + variance_target + c2)
        )
    )
    return {
        "psnr": float(-10 * torch.log10(mse)),
        "ssim": float(global_ssim),
        "mae": float(residual.abs().mean()),
        "rmse": float(mse.sqrt()),
    }


def _historical_uint8_raster(image: torch.Tensor) -> torch.Tensor:
    """Reproduce the PNG raster consumed by the historical Cambridge metric.

    The retained 18.756 report was computed after the renderer clamped its
    output, multiplied it by 255 and cast it to uint8.  Computing directly on
    an unclamped floating-point Teacher render is useful as an optimization
    diagnostic, but it is not the same measurement.  Keep both protocols
    explicit instead of attaching a historical-comparability label to the
    in-memory tensor.
    """
    finite = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    return torch.floor(finite.clamp(0.0, 1.0) * 255.0) / 255.0


def _high_frequency_metrics(prediction, target, mask=None):
    """Measure edge fidelity that RGB PSNR can hide.

    Metrics are evaluated on luminance Sobel/Laplacian responses.  Region
    masks are eroded by one pixel so semantic-mask boundaries do not become
    artificial image edges.
    """
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("high-frequency inputs must be aligned CHW images")
    device, dtype = prediction.device, prediction.dtype
    luminance = torch.tensor(
        [0.299, 0.587, 0.114], device=device, dtype=dtype
    ).reshape(3, 1, 1)
    prediction_luma = (prediction * luminance).sum(0)[None, None]
    target_luma = (target * luminance).sum(0)[None, None]
    kernels = torch.tensor(
        [
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        ],
        device=device,
        dtype=dtype,
    )[:, None]
    prediction_response = torch.nn.functional.conv2d(
        prediction_luma, kernels, padding=1
    )[0]
    target_response = torch.nn.functional.conv2d(
        target_luma, kernels, padding=1
    )[0]
    prediction_response[:2] /= 8.0
    target_response[:2] /= 8.0
    prediction_response[2] /= 4.0
    target_response[2] /= 4.0

    height, width = prediction.shape[-2:]
    if mask is None:
        valid = torch.ones(
            height, width, device=device, dtype=torch.bool
        )
    else:
        valid = torch.as_tensor(
            mask, device=device, dtype=torch.bool
        ).reshape(height, width)
        valid = (
            torch.nn.functional.conv2d(
                valid[None, None].float(),
                torch.ones(1, 1, 3, 3, device=device),
                padding=1,
            )[0, 0]
            == 9
        )
    valid[[0, -1], :] = False
    valid[:, [0, -1]] = False
    if not bool(valid.any()):
        return None
    prediction_gradient = prediction_response[:2, valid]
    target_gradient = target_response[:2, valid]
    gradient_residual = prediction_gradient - target_gradient
    prediction_energy = prediction_gradient.square().sum(0).sqrt()
    target_energy = target_gradient.square().sum(0).sqrt()
    cosine = (
        (prediction_gradient * target_gradient).sum()
        / (
            prediction_gradient.square().sum().sqrt()
            * target_gradient.square().sum().sqrt()
        ).clamp_min(1e-12)
    )
    return {
        "gradient_mae": float(gradient_residual.abs().mean()),
        "gradient_cosine": float(cosine),
        "edge_energy_ratio": float(
            prediction_energy.sum()
            / target_energy.sum().clamp_min(1e-12)
        ),
        "laplacian_mae": float(
            (
                prediction_response[2, valid]
                - target_response[2, valid]
            )
            .abs()
            .mean()
        ),
    }


def _tree_boundary_masks(
    tree_keep: torch.Tensor,
    radius_pixels: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Split the tree/non-tree interface into two diagnostic bands.

    ``tree_keep`` follows the Cambridge mask convention: true pixels are
    outside the tree class.  The two returned masks deliberately remain
    independent of the static-valid mask so callers can intersect the exact
    evaluation validity contract afterwards.

    The automatic radius is proportional to the shorter image side.  This
    keeps the diagnostic comparable across render resolutions without making
    it a training gate or a hand-tuned threshold for one camera.
    """
    tree_keep = torch.as_tensor(tree_keep, dtype=torch.bool)
    if tree_keep.ndim != 2:
        raise ValueError("tree_keep must be a 2D mask")
    if radius_pixels is None:
        radius_pixels = max(
            2, int(round(min(tree_keep.shape) * 0.015))
        )
    radius_pixels = int(radius_pixels)
    if radius_pixels < 1:
        raise ValueError("tree boundary radius must be positive")
    kernel_size = 2 * radius_pixels + 1
    tree = ~tree_keep
    dilated_tree = (
        torch.nn.functional.max_pool2d(
            tree[None, None].float(),
            kernel_size=kernel_size,
            stride=1,
            padding=radius_pixels,
        )[0, 0]
        > 0
    )
    dilated_non_tree = (
        torch.nn.functional.max_pool2d(
            tree_keep[None, None].float(),
            kernel_size=kernel_size,
            stride=1,
            padding=radius_pixels,
        )[0, 0]
        > 0
    )
    inside = tree & dilated_non_tree
    outside = tree_keep & dilated_tree
    return inside, outside, radius_pixels


def _database_view_set_contract(
    selected_indices,
    total_view_count: int,
    *,
    expected_historical_view_count: int = 1487,
) -> dict[str, object]:
    """Describe whether an aggregate covers the historical database split.

    Matching cameras, RGB rasters and scalar formulas is necessary but not
    sufficient for comparison with a retained all-view mean.  The exact view
    set is part of the metric protocol as well.
    """
    selected = [int(value) for value in selected_indices]
    total_view_count = int(total_view_count)
    complete_current_split = (
        selected == list(range(total_view_count))
    )
    historical_view_count_match = (
        total_view_count == int(expected_historical_view_count)
    )
    return {
        "selected_view_count": len(selected),
        "dataset_view_count": total_view_count,
        "expected_historical_view_count": int(
            expected_historical_view_count
        ),
        "complete_current_split": complete_current_split,
        "historical_view_count_match": historical_view_count_match,
        "exact_historical_database_view_set": bool(
            complete_current_split and historical_view_count_match
        ),
    }


def _historical_static_comparison_is_valid(
    *,
    raster_contract_match: bool,
    database_view_set_contract: dict[str, object],
    evaluation_mode: str,
    semantic_conditioning_source: str,
) -> bool:
    """Return whether the canonical render matches the retained 2DGS metric.

    A canonical-only invocation and the canonical branch of a deployment-valid
    hybrid invocation are the same comparison scope. Conditioned database
    codes, oracle routing, rigid-only incomplete maps and view subsets are not.
    """
    canonical_scope = (
        evaluation_mode == "canonical"
        or (
            evaluation_mode == "hybrid"
            and semantic_conditioning_source == "rendered"
        )
    )
    return bool(
        raster_contract_match
        and database_view_set_contract.get(
            "exact_historical_database_view_set", False
        )
        and canonical_scope
    )


def _projected_radius_diagnostics(
    radii: torch.Tensor,
    *,
    large_radius_pixels: float = 24.0,
) -> dict[str, object]:
    """Summarize visible screen footprints relevant to mixed depth order."""
    flat = torch.as_tensor(radii).detach().float().reshape(-1)
    visible = torch.isfinite(flat) & (flat > 0)
    source_rows = torch.nonzero(visible, as_tuple=False).flatten()
    values = flat[visible]
    if not len(values):
        return {
            "visible_count": 0,
            "median_pixels": 0.0,
            "p95_pixels": 0.0,
            "p99_pixels": 0.0,
            "maximum_pixels": 0.0,
            "large_radius_pixels": float(large_radius_pixels),
            "large_count": 0,
            "large_fraction": 0.0,
            "largest_rows": [],
        }
    largest_count = min(8, int(len(values)))
    largest_values, largest_order = torch.topk(
        values, k=largest_count, largest=True, sorted=True
    )
    quantiles = torch.quantile(
        values, values.new_tensor([0.5, 0.95, 0.99])
    )
    large = values > float(large_radius_pixels)
    return {
        "visible_count": int(len(values)),
        "median_pixels": float(quantiles[0]),
        "p95_pixels": float(quantiles[1]),
        "p99_pixels": float(quantiles[2]),
        "maximum_pixels": float(values.max()),
        "large_radius_pixels": float(large_radius_pixels),
        "large_count": int(large.sum()),
        "large_fraction": float(large.float().mean()),
        "largest_rows": [
            {
                "row": int(source_rows[int(order)]),
                "radius_pixels": float(value),
            }
            for value, order in zip(largest_values, largest_order)
        ],
    }


def _route_evaluation_scene_artifacts(dataset, output: Path) -> Path:
    """Keep read-only evaluation from mutating the Teacher model directory."""
    teacher_model_path = Path(dataset.model_path).expanduser().resolve()
    evaluation_output = Path(output).expanduser().resolve()
    evaluation_output.mkdir(parents=True, exist_ok=True)
    dataset.model_path = str(evaluation_output)
    return teacher_model_path


def _mean_protocol(rows, mode, region, predicate=None):
    values = [
        row[mode]["protocol"][region]
        for row in rows
        if row[mode]["protocol"].get(region) is not None
        and (predicate is None or predicate(row))
    ]
    if not values:
        return None
    return {
        metric: float(np.mean([value[metric] for value in values]))
        for metric in ("psnr", "ssim", "mae", "rmse")
    } | {"evaluated_view_count": len(values)}


def _mean_high_frequency(rows, mode, region):
    values = [
        row[mode]["high_frequency"][region]
        for row in rows
        if row[mode]["high_frequency"].get(region) is not None
    ]
    if not values:
        return None
    return {
        metric: float(np.mean([value[metric] for value in values]))
        for metric in (
            "gradient_mae",
            "gradient_cosine",
            "edge_energy_ratio",
            "laplacian_mae",
        )
    } | {"evaluated_view_count": len(values)}


def _conditioned_visit_counts(
    state: dict, view_count: int
) -> tuple[np.ndarray, dict]:
    """Recover which per-view conditioned codes were actually optimized.

    A checkpoint can contain initialized codes for every database image long
    before the biased conditioned schedule has visited every image.  Treating
    those untouched codes as a trained branch made short-run all-view metrics
    both pessimistic and methodologically ambiguous.  The immutable schedule
    stored in the checkpoint is the authoritative visitation record.
    """
    view_count = max(int(view_count), 0)
    counts = np.zeros(view_count, dtype=np.int64)
    schedule = state.get("schedules", {}).get("conditioned")
    contract = state.get("training_contract", {})
    total = int(
        contract.get(
            "schedule_horizon",
            contract.get("iterations", state.get("iterations", 0)),
        )
    )
    checkpoint = int(state.get("iteration", 0))
    activation = contract.get("branch_activation", {})
    dynamic_iteration = activation.get("dynamic_iteration")
    dynamic_start = activation.get("dynamic")
    explicit = state.get("conditioned_visit_counts")
    if explicit is not None:
        if torch.is_tensor(explicit):
            explicit = explicit.detach().cpu().numpy()
        explicit = np.asarray(explicit)
        if (
            explicit.ndim != 1
            or len(explicit) != view_count
            or not np.issubdtype(explicit.dtype, np.integer)
            or bool((explicit < 0).any())
        ):
            raise RuntimeError(
                "Checkpoint conditioned visit ledger is invalid"
            )
        counts = explicit.astype(np.int64, copy=True)
        trained = int((counts > 0).sum())
        provenance = state.get(
            "conditioned_visit_provenance", {}
        )
        return counts, {
            "available": True,
            "source": "explicit_runtime_ledger",
            "provenance": provenance,
            "schedule_horizon": total,
            "active_conditioned_steps": int(counts.sum()),
            "trained_view_count": trained,
            "untrained_view_count": view_count - trained,
            "trained_view_fraction": trained / max(view_count, 1),
            "minimum_visits": int(counts.min()) if len(counts) else 0,
            "median_visits": (
                float(np.median(counts[counts > 0]))
                if bool((counts > 0).any())
                else 0.0
            ),
            "maximum_visits": (
                int(counts.max()) if len(counts) else 0
            ),
        }
    if (
        schedule is None
        or total <= 0
        or checkpoint <= 0
        or (dynamic_iteration is None and dynamic_start is None)
    ):
        return counts, {
            "available": False,
            "reason": "checkpoint_has_no_conditioned_schedule_contract",
            "trained_view_count": 0,
            "untrained_view_count": int(view_count),
        }
    schedule = np.asarray(schedule, dtype=np.int64).reshape(-1)
    stop = min(checkpoint, len(schedule))
    if dynamic_iteration is not None:
        # The immutable contract stores one-based activation iterations,
        # while schedules are indexed from zero.
        start = min(max(int(dynamic_iteration) - 1, 0), stop)
        activation_source = "absolute_iteration"
    else:
        # Legacy checkpoints enabled iteration i exactly when
        # i / total > dynamic_start.
        start = min(
            max(int(np.floor(total * float(dynamic_start))), 0),
            stop,
        )
        activation_source = "legacy_fraction"
    # The conditioned branch is deliberately disabled during the terminal
    # canonical-polish phase.  Legacy schedule-only checkpoints did not store
    # an explicit ledger, so exclude that suffix rather than claiming those
    # camera codes received optimizer updates.
    resolved_phases = state.get(
        "resolved_phase_schedule",
        contract.get("resolved_phase_schedule", ()),
    )
    previous_endpoint = 0
    canonical_polish_begin = stop
    for name, endpoint in resolved_phases:
        if str(name) == "canonical_polish":
            canonical_polish_begin = min(
                max(int(previous_endpoint), 0), stop
            )
            break
        previous_endpoint = int(endpoint)
    effective_stop = min(stop, canonical_polish_begin)
    active = schedule[start:effective_stop]
    if active.size and (
        int(active.min()) < 0 or int(active.max()) >= int(view_count)
    ):
        raise RuntimeError(
            "Conditioned schedule contains an out-of-range camera index"
        )
    if active.size:
        counts += np.bincount(active, minlength=int(view_count))[:view_count]
    trained = int((counts > 0).sum())
    return counts, {
        "available": True,
        "source": "legacy_schedule_reconstruction",
        "schedule_horizon": total,
        "activation_source": activation_source,
        "dynamic_iteration": (
            int(dynamic_iteration)
            if dynamic_iteration is not None
            else start + 1
        ),
        "dynamic_start_fraction": (
            float(dynamic_start) if dynamic_start is not None else None
        ),
        "active_schedule_begin": start,
        "active_schedule_end": effective_stop,
        "active_conditioned_steps": int(active.size),
        "trained_view_count": trained,
        "untrained_view_count": int(view_count) - trained,
        "trained_view_fraction": trained / max(int(view_count), 1),
        "minimum_visits": int(counts.min()) if len(counts) else 0,
        "median_visits": (
            float(np.median(counts[counts > 0]))
            if bool((counts > 0).any())
            else 0.0
        ),
        "maximum_visits": int(counts.max()) if len(counts) else 0,
    }


def _evaluation_indices(
    *,
    indices: str | None,
    evaluate_all: bool,
    localization_query: bool,
    view_count: int,
) -> list[int]:
    """Resolve an explicit, split-aware evaluation view set.

    Database diagnostics retain the historical 408--410 default.  A
    localization query has a different camera list, so silently applying
    those database indices is invalid; its default is the complete query
    split.
    """
    if evaluate_all or (localization_query and indices is None):
        resolved = list(range(int(view_count)))
    else:
        specification = indices
        if specification is None:
            specification = "408,409,410"
        resolved = [
            int(value)
            for value in specification.split(",")
            if value.strip()
        ]
    invalid = [
        index
        for index in resolved
        if index < 0 or index >= int(view_count)
    ]
    if invalid:
        raise IndexError(
            "Evaluation indices are outside the selected split: "
            f"view_count={view_count}, invalid={invalid[:8]}"
        )
    return resolved


def _query_representation_contract(
    *, localization_query: bool, evaluation_mode: str
) -> dict:
    """Declare exactly which map representation a query render measures.

    Query cameras must never access fitted per-database-image conditioned
    codes.  A rigid-only render is nevertheless a valid *stage diagnostic*:
    it measures whether the surface scaffold generalizes before the canonical
    canopy/volume branch is enabled.  Treating it as a complete localization
    map would be wrong, so the distinction is made machine-readable here.
    """
    if not localization_query:
        return {
            "scope": "database_fit",
            "complete_localization_map": False,
            "conditioned_database_code_used": (
                evaluation_mode == "hybrid"
            ),
        }
    if evaluation_mode == "hybrid":
        raise RuntimeError(
            "Localization queries cannot use hybrid rendering because it "
            "would access fitted per-database-image conditioned codes. Use "
            "canonical for the complete map or explicit rigid for an "
            "incomplete surface-stage diagnostic."
        )
    if evaluation_mode == "rigid":
        return {
            "scope": "rigid_surface_stage_diagnostic",
            "complete_localization_map": False,
            "conditioned_database_code_used": False,
            "omitted_components": [
                "canonical_canopy_volume",
                "sequence_time_conditioned_volume",
            ],
        }
    if evaluation_mode == "canonical":
        return {
            "scope": "complete_canonical_localization_map",
            "complete_localization_map": True,
            "conditioned_database_code_used": False,
            "omitted_components": [
                "sequence_time_conditioned_volume",
            ],
        }
    raise ValueError(
        f"Unsupported localization query evaluation mode: {evaluation_mode}"
    )


def _assert_resolution_contract(
    state: dict,
    views,
    *,
    allow_override: bool,
) -> None:
    expected = state.get("training_contract", {}).get(
        "render_resolution"
    )
    if not expected or not views:
        return
    actual = {
        "width": int(views[0].image_width),
        "height": int(views[0].image_height),
    }
    expected = {
        "width": int(expected["width"]),
        "height": int(expected["height"]),
    }
    if actual != expected and not allow_override:
        raise RuntimeError(
            "Evaluation resolution differs from the immutable training "
            f"contract: trained={expected}, evaluation={actual}. Pass "
            "--allow-resolution-override only for explicitly non-comparable "
            "visualization."
        )


def _assert_rgb_source_contract(
    state: dict,
    dataset,
    *,
    allow_override: bool,
    actual: dict | None = None,
) -> tuple[dict, bool]:
    expected = state.get("training_contract", {}).get("rgb_source")
    if actual is None:
        actual = rgb_source_contract(dataset)
    if not expected:
        return actual, False
    identity_fields = [
        "image_root",
        "image_count",
        "name_set_sha256",
        "producer_manifest_sha256",
        "target_storage",
        "canonical_image_size_wh",
    ]
    # Checkpoints produced before the byte-level RGB binding remain readable,
    # but every new checkpoint must prove that the exact raster bytes (not
    # merely the set of filenames) are unchanged at evaluation.
    if expected.get("content_mapping_sha256") is not None:
        identity_fields.extend(("content_mapping_sha256", "content_bytes"))
    matched = all(expected.get(key) == actual.get(key) for key in identity_fields)
    if not matched and not allow_override:
        raise RuntimeError(
            "Evaluation RGB targets differ from the immutable training "
            f"contract: trained={expected}, evaluation={actual}. Pass "
            "--allow-resolution-override only for explicitly non-comparable "
            "visualization."
        )
    return actual, matched


def _assert_camera_geometry_contract(
    state: dict,
    camera_contract_path: Path,
    *,
    allow_override: bool,
) -> tuple[str, bool]:
    expected = state.get("training_contract", {}).get(
        "camera_geometry_sha256"
    )
    actual = json.loads(
        camera_contract_path.read_text(encoding="utf-8")
    )["camera_geometry_sha256"]
    if expected is None:
        return actual, False
    matched = str(expected) == str(actual)
    if not matched and not allow_override:
        raise RuntimeError(
            "Evaluation camera geometry differs from the immutable training "
            f"contract: trained={expected}, evaluation={actual}. Pass "
            "--allow-resolution-override only for explicitly non-comparable "
            "visualization."
        )
    return actual, matched


def _assert_semantic_contract(
    semantic_contract_path: Path,
    tree_mask_pickle: Path | None,
) -> tuple[dict, str]:
    """Bind protocol regions to the exact mask artifact used by evidence.

    RGB targets and masks intentionally live in different stores.  Accepting
    an arbitrary ``masks.pkl`` next to the resized RGBs previously allowed a
    three-channel base mask to be mistaken for the four-channel tree mask.
    """
    contract = json.loads(
        semantic_contract_path.read_text(encoding="utf-8")
    )
    if tree_mask_pickle is None:
        recorded = contract.get("tree_mask_pickle")
        if not recorded:
            raise RuntimeError(
                "Semantic contract does not record a tree-mask path; pass "
                "--tree-mask-pickle explicitly"
            )
        tree_mask_pickle = Path(recorded)
    tree_mask_pickle = Path(tree_mask_pickle)
    if not tree_mask_pickle.is_file():
        raise FileNotFoundError(
            f"Evaluation tree mask does not exist: {tree_mask_pickle}"
        )
    expected = contract.get("input_hashes", {}).get("tree_mask_pickle")
    if not expected:
        raise RuntimeError(
            "Semantic contract does not bind a tree-mask SHA256"
        )
    actual = sha256_file(tree_mask_pickle)
    if actual != expected:
        raise RuntimeError(
            "Evaluation tree mask differs from the immutable semantic "
            f"contract: expected={expected}, actual={actual}, "
            f"path={tree_mask_pickle}"
        )
    canopy = contract.get("class_availability", {}).get("canopy")
    if canopy != "tree_mask_index_3":
        raise RuntimeError(
            "Semantic contract does not define canopy as tree mask index 3"
        )
    return contract, actual


def _assert_query_rgb_compatibility(
    state: dict, evaluation_rgb_source: dict
) -> None:
    """Require query RGBs to use the training rasterization convention.

    A query split must differ in image identity, so it cannot match the
    training RGB content hash.  Resolution, storage and resize convention are
    nevertheless controlled variables and must remain identical.
    """
    expected = state.get("training_contract", {}).get("rgb_source", {})
    fields = (
        "target_storage",
        "canonical_image_size_wh",
    )
    differences = {
        field: {
            "trained": expected.get(field),
            "query": evaluation_rgb_source.get(field),
        }
        for field in fields
        if expected.get(field) != evaluation_rgb_source.get(field)
    }
    if differences:
        raise RuntimeError(
            "Localization-query RGBs do not use the training raster "
            f"convention: {differences}"
        )


def _assert_localization_query_contract(
    database_contract_path: Path,
    query_contract_path: Path,
    *,
    dataset_source: Path,
    view_names: list[str],
) -> dict:
    """Prove that an evaluated query trajectory is calibrated and disjoint."""
    database_contract_path = database_contract_path.resolve()
    query_contract_path = query_contract_path.resolve()
    disjointness = validate_disjoint_contracts(
        database_contract_path, query_contract_path
    )
    query = json.loads(query_contract_path.read_text(encoding="utf-8"))
    expected_dataset = str(dataset_source.resolve())
    if str(Path(query.get("dataset", "")).resolve()) != expected_dataset:
        raise RuntimeError(
            "Query camera contract is bound to a different dataset: "
            f"contract={query.get('dataset')}, evaluation={expected_dataset}"
        )
    if query.get("split") != "localization_query":
        raise RuntimeError(
            "Query camera contract must declare split=localization_query"
        )
    if query.get("camera_policy") != "calibrated_fixed":
        raise RuntimeError(
            "Query camera contract must use calibrated_fixed cameras"
        )
    # ``Scene`` follows the original 2DGS convention and stores camera names
    # without their raster suffix, whereas the immutable camera manifest keeps
    # the exact on-disk filename.  Compare the canonical camera key, but still
    # reject duplicate keys so this cannot hide an ambiguous manifest.
    def camera_key(name: str) -> str:
        path = Path(str(name))
        return str(path.with_suffix("")) if path.suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
        } else str(path)

    recorded_name_list = [
        camera_key(str(record["image_name"]))
        for record in query.get("records", [])
    ]
    actual_name_list = [camera_key(str(name)) for name in view_names]
    recorded_names = set(recorded_name_list)
    actual_names = set(actual_name_list)
    if len(recorded_names) != len(recorded_name_list):
        raise RuntimeError(
            "Query camera contract contains duplicate canonical camera names"
        )
    if len(actual_names) != len(actual_name_list):
        raise RuntimeError(
            "Evaluated query cameras contain duplicate canonical camera names"
        )
    if recorded_names != actual_names:
        missing = sorted(recorded_names - actual_names)[:5]
        unexpected = sorted(actual_names - recorded_names)[:5]
        raise RuntimeError(
            "Query camera contract and evaluated camera names differ: "
            f"recorded={len(recorded_names)}, actual={len(actual_names)}, "
            f"missing={missing}, unexpected={unexpected}"
        )
    if int(query.get("image_count", -1)) != len(view_names):
        raise RuntimeError(
            "Query camera contract image count differs from evaluation"
        )
    return {
        "database_contract": str(database_contract_path),
        "database_contract_sha256": sha256_file(database_contract_path),
        "query_contract": str(query_contract_path),
        "query_contract_sha256": sha256_file(query_contract_path),
        "disjointness": disjointness,
        "query_camera_policy": query["camera_policy"],
        "query_image_count": len(view_names),
        "camera_name_comparison": "path_without_raster_suffix",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    parser.set_defaults(data_device="cpu", white_background=True)
    parser.add_argument("--teacher-state", type=Path, required=True)
    parser.add_argument("--semantic-contract", type=Path, required=True)
    parser.add_argument(
        "--tree-mask-pickle",
        type=Path,
        help=(
            "Optional explicit tree-mask artifact. By default the evaluator "
            "uses the content-addressed path recorded by --semantic-contract."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--indices",
        default=None,
        help=(
            "Comma-separated view indices. Database diagnostics default to "
            "408,409,410; localization queries default to the complete "
            "disjoint query split."
        ),
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument(
        "--print-full-json",
        action="store_true",
        help=(
            "Print every per-view row to stdout. By default the complete "
            "payload is written to metrics.json and stdout stays concise."
        ),
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=("auto", "rigid", "canonical", "hybrid"),
        default="auto",
        help=(
            "Rigid evaluates the native 2D surfel/sky scaffold with all 3D "
            "volumes disabled. Canonical evaluates the mixed localization map "
            "without any per-image conditioned code. Hybrid reports both "
            "canonical and conditioned database-view rendering. Auto reads "
            "the checkpoint branch state for a database fit and resolves to "
            "canonical for a localization query. An explicit rigid query is "
            "allowed only as an incomplete surface-stage diagnostic; hybrid "
            "query rendering is forbidden."
        ),
    )
    parser.add_argument(
        "--evaluation-split",
        choices=("database_fit", "localization_query"),
        default="database_fit",
        help=(
            "Localization-query evaluation requires disjoint calibrated "
            "camera contracts. Canonical is the complete-map protocol; "
            "explicit rigid is a labelled surface-stage diagnostic."
        ),
    )
    parser.add_argument("--database-contract", type=Path)
    parser.add_argument("--query-contract", type=Path)
    parser.add_argument(
        "--semantic-conditioning-source",
        choices=("rendered", "oracle"),
        default="rendered",
        help=(
            "Conditioned appearance routing. 'rendered' derives soft "
            "surface/volume/sky ownership from the mixed render and is the "
            "deployment-valid default. 'oracle' uses GT protocol masks only "
            "as a diagnostic upper bound."
        ),
    )
    parser.add_argument("--view-cache-size", type=int, default=2)
    parser.add_argument(
        "--allow-projected-optical-footprint-repair",
        action="store_true",
        help=(
            "Explicitly evaluate the exact retained v39 state with the v40 "
            "view-projected optical-mass and non-owner exact-ray render-"
            "footprint repair. This is a labelled causal reinterpretation, "
            "not a render-equivalent migration."
        ),
    )
    parser.add_argument(
        "--allow-static-foliage-render-repair",
        action="store_true",
        help=(
            "Explicitly reinterpret the exact v131/v40 checkpoint with the "
            "v41 no-screen-grid appearance and view/depth-local optical "
            "handoff. Intended only for labelled zero-training diagnosis."
        ),
    )
    parser.add_argument(
        "--exact-ray-render-aspect-limit",
        type=float,
        default=4.0,
        help=(
            "Render-only maximum depth/tangent scale ratio for a visible "
            "single-observation exact-ray leaf. Metric position covariance, "
            "tangent footprint, colour and opacity are unchanged."
        ),
    )
    parser.add_argument(
        "--optical-replacement-policy",
        choices=("view_depth_local", "group_projected", "disabled"),
        default="view_depth_local",
        help=(
            "Envelope-to-detail handoff. view_depth_local additionally "
            "requires active-camera projection and depth-interval overlap."
        ),
    )
    parser.add_argument(
        "--allow-resolution-override",
        action="store_true",
        help=(
            "Permit non-comparable visualization at a resolution different "
            "from the training contract."
        ),
    )
    args = parser.parse_args()
    dataset = model.extract(args)
    localization_query = args.evaluation_split == "localization_query"
    if localization_query:
        if (
            args.database_contract is None
            or args.query_contract is None
        ):
            raise ValueError(
                "Localization-query evaluation requires --database-contract "
                "and --query-contract"
            )
    elif (
        args.database_contract is not None
        or args.query_contract is not None
    ):
        raise ValueError(
            "Camera disjointness contracts are valid only with "
            "--evaluation-split localization_query"
        )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    teacher_model_path = _route_evaluation_scene_artifacts(
        dataset, output
    )
    # Resolve and hash the RGB source before parsing all camera records.  This
    # turns a misspelled ``--images`` directory into an immediate contract
    # error instead of spending time constructing 1k+ calibrated cameras only
    # to fail when the first raster is opened.
    evaluation_rgb_source = rgb_source_contract(dataset)
    teacher = load_hybrid_teacher(
        args.teacher_state,
        sh_degree=dataset.sh_degree,
        allow_projected_optical_footprint_repair=(
            args.allow_projected_optical_footprint_repair
        ),
        allow_static_foliage_render_repair=(
            args.allow_static_foliage_render_repair
        ),
    )
    evaluation_rgb_source, rgb_source_contract_match = (
        _assert_rgb_source_contract(
            teacher.state,
            dataset,
            allow_override=(
                args.allow_resolution_override or localization_query
            ),
            actual=evaluation_rgb_source,
        )
    )
    if localization_query:
        _assert_query_rgb_compatibility(
            teacher.state, evaluation_rgb_source
        )
    # Validate the semantic artifact before constructing 1k+ camera objects.
    # RGB folders commonly contain a different three-channel ``masks.pkl``;
    # the contract-recorded four-channel tree mask is authoritative.
    semantic_contract, evaluation_tree_mask_sha256 = (
        _assert_semantic_contract(
            args.semantic_contract, args.tree_mask_pickle
        )
    )
    tree_mask_pickle = Path(semantic_contract["tree_mask_pickle"])
    if args.tree_mask_pickle is not None:
        tree_mask_pickle = args.tree_mask_pickle
    scene = LazyScene(
        dataset,
        GaussianModel(dataset.sh_degree),
        image_cache_size=args.view_cache_size,
    )
    views = scene.getTrainCameras()
    _assert_resolution_contract(
        teacher.state,
        views,
        allow_override=args.allow_resolution_override,
    )
    (
        evaluation_camera_geometry_sha256,
        camera_geometry_contract_match,
    ) = _assert_camera_geometry_contract(
        teacher.state,
        Path(dataset.model_path) / "camera_intrinsics_contract.json",
        allow_override=(
            args.allow_resolution_override or localization_query
        ),
    )
    training_profile = str(teacher.state.get("training_profile", "unknown"))
    branch_validity = teacher_branch_validity(teacher.state)
    evaluation_mode = args.evaluation_mode
    if evaluation_mode == "auto":
        if localization_query:
            evaluation_mode = "canonical"
        else:
            evaluation_mode = (
                "hybrid" if branch_validity["conditioned"] else "rigid"
            )
    query_representation_contract = _query_representation_contract(
        localization_query=localization_query,
        evaluation_mode=evaluation_mode,
    )
    if evaluation_mode == "hybrid" and not branch_validity["conditioned"]:
        raise RuntimeError(
            "Requested hybrid evaluation for a checkpoint whose conditioned "
            f"branch is not trained ({branch_validity['reason']}). Use "
            "--evaluation-mode auto/rigid or evaluate a later checkpoint."
        )
    modes = (
        ("canonical", "conditioned")
        if evaluation_mode == "hybrid"
        else ("canonical",)
    )
    if evaluation_mode == "hybrid":
        conditioned_visit_counts, conditioned_sampling_coverage = (
            _conditioned_visit_counts(teacher.state, len(views))
        )
    else:
        conditioned_visit_counts = np.zeros(len(views), dtype=np.int64)
        conditioned_sampling_coverage = {
            "available": False,
            "reason": "conditioned_render_not_evaluated",
            "trained_view_count": 0,
            "untrained_view_count": len(views),
        }
    query_contract_validation = None
    if localization_query:
        query_contract_validation = _assert_localization_query_contract(
            args.database_contract,
            args.query_contract,
            dataset_source=Path(dataset.source_path),
            view_names=[str(view.image_name) for view in views],
        )
    # Bind region metrics to the exact four-channel mask artifact from the
    # evidence contract.  RGB directories may contain only the historical
    # three-channel base masks.
    # The tree pickle preserves base channels 0..2 bit-for-bit and appends
    # channel 3. Materialize all four once per camera. The previous evaluator
    # loaded/resized the same 2017-view masks through four lookup objects.
    protocol_masks = CambridgeMaskLookup(
        Path(dataset.source_path),
        tree_mask_pickle,
        mask_indices=[0, 1, 2, 3],
    )
    indices = _evaluation_indices(
        indices=args.indices,
        evaluate_all=args.all,
        localization_query=localization_query,
        view_count=len(views),
    )
    rows = []
    with torch.no_grad():
        for index in indices:
            view = views[index]
            keep = protocol_masks.get_index_masks(
                view.image_name,
                (0, 1, 2, 3),
                (view.image_height, view.image_width),
                torch.device("cuda"),
            )
            object_keep, sky_keep, distortion_keep, tree_keep = keep
            p_transient = (~object_keep).float()
            p_sky = (~sky_keep).float() * (1.0 - p_transient)
            p_canopy = (
                (~tree_keep).float()
                * (1.0 - p_transient)
                * (1.0 - p_sky)
            )
            (
                tree_boundary_inside,
                tree_boundary_outside,
                tree_boundary_radius,
            ) = _tree_boundary_masks(tree_keep)
            task = {
                "p_rigid": (
                    object_keep
                    & sky_keep
                    & distortion_keep
                    & tree_keep
                ).float(),
                "p_canopy": p_canopy,
                "p_sky": p_sky,
            }
            target = view.original_image.cuda(non_blocking=True)
            # Canonical reconstruction must remain deployable and must not
            # receive ground-truth semantic routing. The current Teacher API
            # ignores ``task`` in canonical mode, but passing it here made
            # that non-leakage property depend on an undocumented callee
            # detail and could silently invalidate future evaluations.
            canonical = teacher.render(
                view,
                task=None,
                conditioned=False,
                surface_only=(evaluation_mode == "rigid"),
                exact_ray_render_aspect_limit=(
                    args.exact_ray_render_aspect_limit
                ),
                optical_replacement_policy=(
                    args.optical_replacement_policy
                ),
            )
            conditioned = (
                teacher.render(
                    view,
                    task=(
                        task
                        if args.semantic_conditioning_source == "oracle"
                        else None
                    ),
                    conditioned=True,
                    exact_ray_render_aspect_limit=(
                        args.exact_ray_render_aspect_limit
                    ),
                    optical_replacement_policy=(
                        args.optical_replacement_policy
                    ),
                )
                if evaluation_mode == "hybrid"
                else None
            )
            row = {
                "index": index,
                "image_name": str(view.image_name),
                "conditioned_training_visits": int(
                    conditioned_visit_counts[index]
                ),
                "tree_boundary_radius_pixels": tree_boundary_radius,
            }
            render_by_mode = {"canonical": canonical}
            if conditioned is not None:
                render_by_mode["conditioned"] = conditioned
            for name, render in render_by_mode.items():
                prediction = render["rgb"]
                historical_prediction = _historical_uint8_raster(prediction)
                historical_target = _historical_uint8_raster(target)
                mse = (prediction - target).square().mean()
                row[name] = {
                    "psnr": float(
                        -10 * torch.log10(mse.clamp_min(1e-12))
                    ),
                    "ssim": float(ssim(prediction, target)),
                    "mae": float((prediction - target).abs().mean()),
                    "rigid": _region(
                        prediction, target, task["p_rigid"]
                    ),
                    "canopy": _region(
                        prediction, target, task["p_canopy"]
                    ),
                    "projected_footprint": {
                        "surface": _projected_radius_diagnostics(
                            render["surface_radii"]
                        ),
                        "volume": _projected_radius_diagnostics(
                            render["volume_radii"]
                        ),
                    },
                }
                static_keep = object_keep & sky_keep & distortion_keep
                dynamic_keep = object_keep & distortion_keep
                boundary_inside_static = (
                    static_keep & tree_boundary_inside
                )
                boundary_outside_static = (
                    static_keep & tree_boundary_outside
                )
                row.setdefault(
                    "region_pixel_counts",
                    {
                        "tree_boundary_inside_static": int(
                            boundary_inside_static.sum()
                        ),
                        "tree_boundary_outside_static": int(
                            boundary_outside_static.sum()
                        ),
                    },
                )
                row[name]["protocol"] = {
                    "raw": _protocol_metrics(prediction, target),
                    "dynamic_valid": _protocol_metrics(
                        prediction, target, dynamic_keep
                    ),
                    "static_valid": _protocol_metrics(
                        prediction, target, static_keep
                    ),
                    "non_tree_static": _protocol_metrics(
                        prediction,
                        target,
                        static_keep & tree_keep,
                    ),
                    "tree_static": _protocol_metrics(
                        prediction,
                        target,
                        static_keep & ~tree_keep,
                    ),
                    "tree_boundary_inside_static": _protocol_metrics(
                        prediction,
                        target,
                        boundary_inside_static,
                    ),
                    "tree_boundary_outside_static": _protocol_metrics(
                        prediction,
                        target,
                        boundary_outside_static,
                    ),
                }
                row[name]["historical_uint8_protocol"] = {
                    "raw": _protocol_metrics(
                        historical_prediction, historical_target
                    ),
                    "dynamic_valid": _protocol_metrics(
                        historical_prediction,
                        historical_target,
                        dynamic_keep,
                    ),
                    "static_valid": _protocol_metrics(
                        historical_prediction,
                        historical_target,
                        static_keep,
                    ),
                    "non_tree_static": _protocol_metrics(
                        historical_prediction,
                        historical_target,
                        static_keep & tree_keep,
                    ),
                    "tree_static": _protocol_metrics(
                        historical_prediction,
                        historical_target,
                        static_keep & ~tree_keep,
                    ),
                    "tree_boundary_inside_static": _protocol_metrics(
                        historical_prediction,
                        historical_target,
                        boundary_inside_static,
                    ),
                    "tree_boundary_outside_static": _protocol_metrics(
                        historical_prediction,
                        historical_target,
                        boundary_outside_static,
                    ),
                }
                row[name]["high_frequency"] = {
                    "raw": _high_frequency_metrics(
                        prediction, target
                    ),
                    "static_valid": _high_frequency_metrics(
                        prediction, target, static_keep
                    ),
                    "non_tree_static": _high_frequency_metrics(
                        prediction,
                        target,
                        static_keep & tree_keep,
                    ),
                    "tree_static": _high_frequency_metrics(
                        prediction,
                        target,
                        static_keep & ~tree_keep,
                    ),
                    "tree_boundary_inside_static": (
                        _high_frequency_metrics(
                            prediction,
                            target,
                            boundary_inside_static,
                        )
                    ),
                    "tree_boundary_outside_static": (
                        _high_frequency_metrics(
                            prediction,
                            target,
                            boundary_outside_static,
                        )
                    ),
                }
            rows.append(row)
            if not args.render_only or not args.all:
                images = {
                    "gt": target,
                    "canonical": canonical["rgb"],
                    "error_x4": (
                        (
                            conditioned["rgb"]
                            if conditioned is not None
                            else canonical["rgb"]
                        )
                        - target
                    ).abs()
                    * 4,
                    "surface_alpha": canonical[
                        "surface_alpha"
                    ].expand(3, -1, -1),
                    "canonical_volume_alpha": canonical[
                        "volume_alpha"
                    ].expand(3, -1, -1),
                }
                if conditioned is not None:
                    images["conditioned"] = conditioned["rgb"]
                    images["volume_alpha"] = conditioned[
                        "volume_alpha"
                    ].expand(3, -1, -1)
                for name, value in images.items():
                    pixels = (
                        value.clamp(0, 1)
                        .mul(255)
                        .byte()
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy()
                    )
                    Image.fromarray(pixels).save(
                        output / f"{index:05d}_{name}.png"
                    )
            scene.release_images()
            protocol_masks._resized_mask_cache.clear()
    aggregate = {
        mode: {
            metric: float(
                np.mean([row[mode][metric] for row in rows])
            )
            for metric in ("psnr", "ssim", "mae")
        }
        for mode in modes
    }
    protocol_aggregate = {
        mode: {
            region: _mean_protocol(rows, mode, region)
            for region in (
                "raw",
                "dynamic_valid",
                "static_valid",
                "non_tree_static",
                "tree_static",
                "tree_boundary_inside_static",
                "tree_boundary_outside_static",
            )
        }
        for mode in modes
    }
    historical_uint8_protocol_aggregate = {
        mode: {
            region: _mean_protocol(
                [
                    {
                        **row,
                        mode: {
                            "protocol": row[mode][
                                "historical_uint8_protocol"
                            ]
                        },
                    }
                    for row in rows
                ],
                mode,
                region,
            )
            for region in (
                "raw",
                "dynamic_valid",
                "static_valid",
                "non_tree_static",
                "tree_static",
                "tree_boundary_inside_static",
                "tree_boundary_outside_static",
            )
        }
        for mode in modes
    }
    high_frequency_aggregate = {
        mode: {
            region: _mean_high_frequency(rows, mode, region)
            for region in (
                "raw",
                "static_valid",
                "non_tree_static",
                "tree_static",
                "tree_boundary_inside_static",
                "tree_boundary_outside_static",
            )
        }
        for mode in modes
    }
    conditioned_trained_protocol_aggregate = (
        {
            region: _mean_protocol(
                rows,
                "conditioned",
                region,
                predicate=lambda row: (
                    row["conditioned_training_visits"] > 0
                ),
            )
            for region in (
                "raw",
                "dynamic_valid",
                "static_valid",
                "non_tree_static",
                "tree_static",
                "tree_boundary_inside_static",
                "tree_boundary_outside_static",
            )
        }
        if "conditioned" in modes
        else None
    )
    evaluated_conditioned_trained = sum(
        row["conditioned_training_visits"] > 0 for row in rows
    )
    database_view_set_contract = _database_view_set_contract(
        indices, len(views)
    )
    historical_raster_contract_match = bool(
        not localization_query
        and rgb_source_contract_match
        and camera_geometry_contract_match
        and evaluation_rgb_source.get("target_storage")
        == "shared_uint8_torch_bilinear_align_corners_false"
        and bool(views)
        and int(views[0].image_width) == 640
        and int(views[0].image_height) == 360
    )
    historical_18_756_comparable = (
        _historical_static_comparison_is_valid(
            raster_contract_match=historical_raster_contract_match,
            database_view_set_contract=database_view_set_contract,
            evaluation_mode=evaluation_mode,
            semantic_conditioning_source=(
                args.semantic_conditioning_source
            ),
        )
    )
    payload = {
        "protocol": "native-hybrid-teacher-direct-evaluation-v2",
        # Bind a metric artifact to the exact evaluator implementation.  A
        # teacher-state hash alone cannot detect a later mask, aggregation or
        # image-routing repair being reported under an old metrics.json.
        "evaluation_implementation_sha256": sha256_file(
            Path(__file__).resolve()
        ),
        "evaluation_dependency_hashes": {
            "evaluator": sha256_file(Path(__file__).resolve()),
            "hybrid_teacher_api": sha256_file(
                REPO_ROOT / "outdoor/hybrid_teacher_api.py"
            ),
            "hybrid_renderer": sha256_file(
                REPO_ROOT / "outdoor/hybrid_gaussian_renderer.py"
            ),
            "mixed_forward_cuda": sha256_file(
                SURFEL_ROOT
                / "submodules/diff-surfel-rasterization/"
                "cuda_rasterizer/forward.cu"
            ),
        },
        "teacher_state": str(args.teacher_state.resolve()),
        "teacher_state_sha256": sha256_file(
            args.teacher_state.resolve()
        ),
        "evaluation_scene_artifacts": {
            "teacher_model_path": str(teacher_model_path),
            "evaluation_model_path": str(output),
            "teacher_model_path_mutated": False,
        },
        "training_profile": training_profile,
        "checkpoint_iteration": int(teacher.state.get("iteration", 0)),
        "checkpoint_phase": str(teacher.state.get("phase", "unknown")),
        "render_implementation_validation": teacher.state.get(
            "_render_implementation_validation", {}
        ),
        "branch_validity": branch_validity,
        "evaluation_mode": evaluation_mode,
        "valid_render_modes": list(modes),
        "conditioned_valid": (
            evaluation_mode == "hybrid"
            and bool(branch_validity["conditioned"])
        ),
        "conditioned_sampling_coverage": {
            **conditioned_sampling_coverage,
            "evaluated_trained_view_count": int(
                evaluated_conditioned_trained
            ),
            "evaluated_untrained_view_count": int(
                len(rows) - evaluated_conditioned_trained
            ),
            "all_evaluated_views_trained": bool(
                len(rows) > 0
                and evaluated_conditioned_trained == len(rows)
            ),
        },
        "query_contract_validation": query_contract_validation,
        "query_representation_contract": query_representation_contract,
        "semantic_conditioning_source": (
            args.semantic_conditioning_source
            if evaluation_mode == "hybrid"
            else "not_applicable"
        ),
        "ground_truth_semantics_used_for_rendering": bool(
            evaluation_mode == "hybrid"
            and args.semantic_conditioning_source == "oracle"
        ),
        "deployment_valid_conditioning": bool(
            evaluation_mode != "hybrid"
            or args.semantic_conditioning_source == "rendered"
        ),
        "canopy_quality_valid": bool(
            branch_validity["canonical_canopy"]
        ),
        "view_count": len(rows),
        "aggregate": aggregate,
        "aggregate_semantics": {
            "scope": "unmasked_full_rgb",
            "ssim": "windowed_training_implementation",
            "historical_comparison_valid": False,
            "purpose": "optimization_diagnostic_only",
        },
        "protocol_aggregate": protocol_aggregate,
        "historical_uint8_protocol_aggregate": (
            historical_uint8_protocol_aggregate
        ),
        "high_frequency_aggregate": high_frequency_aggregate,
        "high_frequency_protocol": {
            "colour_space": "bt601_luminance",
            "responses": (
                "sobel_xy_div8_and_4_neighbour_laplacian_div4"
            ),
            "region_mask_boundary": "eroded_one_pixel",
            "tree_boundary_bands": {
                "inside": (
                    "tree pixels within radius of non-tree pixels"
                ),
                "outside": (
                    "non-tree pixels within radius of tree pixels"
                ),
                "radius": (
                    "max(2, round(min(image_height,image_width)*0.015))"
                ),
                "training_gate": False,
            },
            "aggregation": "per_view_then_mean",
            "historical_comparison_valid": False,
            "purpose": "smear_and_edge_fidelity_diagnostic",
        },
        "mixed_depth_order_protocol": {
            "tile_sort_key": "primitive_center_camera_depth",
            "surface_pixel_depth": (
                "perspective_correct_ray_surfel_intersection_after_sort"
            ),
            "volume_pixel_depth": "gaussian_center_camera_depth",
            "known_residual_approximation": (
                "a_large_oblique_surface_footprint_can_cross_volume_center_"
                "depth_within_one_tile"
            ),
            "diagnostic": "per_view.*.projected_footprint",
            "large_radius_pixels": 24.0,
            "claim_of_exact_per_pixel_cross_type_order": False,
        },
        "conditioned_trained_view_protocol_aggregate": (
            conditioned_trained_protocol_aggregate
        ),
        "metric_protocol": {
            "name": "cambridge_rgb_metrics_v3_tree_gpu_adaptermask",
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
            "primary_comparison_field": (
                "protocol_aggregate.canonical.non_tree_static"
                if localization_query
                else (
                    "historical_uint8_protocol_aggregate.canonical."
                    "static_valid"
                    if evaluation_mode in {"hybrid", "canonical"}
                    else (
                        "historical_uint8_protocol_aggregate.canonical."
                        "non_tree_static"
                    )
                )
            ),
            "diagnostic_only_field": "aggregate",
            "in_memory_float_protocol_field": "protocol_aggregate",
            "historical_png_protocol_field": (
                "historical_uint8_protocol_aggregate"
            ),
            "float_protocol_historical_comparison_valid": False,
            "historical_png_write_semantics": (
                "nan_to_num_clamp01_multiply255_uint8_floor"
            ),
            "per_view_then_mean": True,
            "static_valid_mask_indices": [0, 1, 2],
            "dynamic_valid_mask_indices": [0, 2],
            "tree_mask_index": 3,
            "tree_mask_sha256": evaluation_tree_mask_sha256,
            "tree_boundary_diagnostic": {
                "regions": [
                    "tree_boundary_inside_static",
                    "tree_boundary_outside_static",
                ],
                "radius": (
                    "max(2, round(min(image_height,image_width)*0.015))"
                ),
                "aggregation": "per_view_then_mean",
                "primary_comparison_field": False,
                "training_gate": False,
            },
            "ssim": "historical_global_scalar",
            "render_resolution": (
                [int(views[0].image_height), int(views[0].image_width)]
                if views
                else None
            ),
            "rgb_source_contract_match": rgb_source_contract_match,
            "rgb_source": evaluation_rgb_source,
            "camera_geometry_contract_match": (
                camera_geometry_contract_match
            ),
            "camera_geometry_sha256": (
                evaluation_camera_geometry_sha256
            ),
            "database_view_set_contract": database_view_set_contract,
            "historical_raster_contract_match": (
                historical_raster_contract_match
            ),
            "historical_18_756_comparable": (
                historical_18_756_comparable
            ),
            "historical_18_756_comparable_modes": (
                ["canonical"]
                if historical_18_756_comparable
                else []
            ),
            "historical_18_756_comparison_reason": (
                "exact_all1487_database_view_set_and_raster_contract"
                if historical_18_756_comparable
                else (
                    "subset_or_nonhistorical_database_view_set"
                    if (
                        historical_raster_contract_match
                        and not database_view_set_contract[
                            "exact_historical_database_view_set"
                        ]
                    )
                    else (
                        "noncanonical_or_oracle_conditioning_mode"
                        if historical_raster_contract_match
                        else "raster_camera_or_split_contract_mismatch"
                    )
                )
            ),
            "conditioned_historical_comparison_valid": False,
            "conditioned_historical_comparison_reason": (
                "known_database_view_sequence_time_code_has_no_standard_"
                "2dgs_equivalent"
            ),
            "evaluation_split": (
                "official_cambridge_query_gt_pose"
                if localization_query
                else "training_database_reconstruction_fit"
            ),
            "localization_query_evaluation": localization_query,
            "query_render_at_ground_truth_pose": localization_query,
            "pose_estimation_evaluated": False,
            "canonical_query_render_available": bool(
                localization_query and evaluation_mode == "canonical"
            ),
            "historical_non_tree_static_comparable": (
                historical_raster_contract_match
                and database_view_set_contract[
                    "exact_historical_database_view_set"
                ]
            ),
            "comparison_scope": (
                query_representation_contract["scope"]
                if localization_query
                else (
                    "full_historical_static_protocol"
                    if evaluation_mode == "hybrid"
                    else (
                        "canonical_database_fit"
                        if evaluation_mode == "canonical"
                        else "rigid_non_tree_static_only"
                    )
                )
            ),
        },
        "per_view": rows,
        "student_or_standard_renderer_used": False,
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    printed = (
        payload
        if args.print_full_json
        else {
            key: payload[key]
            for key in (
                "protocol",
                "teacher_state",
                "checkpoint_iteration",
                "render_implementation_validation",
                "evaluation_mode",
                "view_count",
                "aggregate",
                "aggregate_semantics",
                "protocol_aggregate",
                "metric_protocol",
            )
        }
    )
    print(json.dumps(printed, indent=2))


if __name__ == "__main__":
    main()
