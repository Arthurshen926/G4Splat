"""Explicit task-specific semantic policy for outdoor Cambridge scenes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


TASK_SEMANTIC_POLICY_VERSION = "outdoor_task_specific_v1"


def _sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_task_semantic_manifest(
    dataset: Path,
    mask_pickle: Path,
    output: Path,
    *,
    tree_mask_pickle: Path | None = None,
    tree_support_policy: str = "neutral",
    neutral_tree_support: float = 0.5,
) -> dict[str, Any]:
    """Persist semantics by *task*, never as an implicit mask-file side effect.

    The supplied Cambridge masks only expose a canonical-tree channel when
    ``tree_mask_pickle`` is present.  We record that limitation explicitly:
    trunk/canopy separation is unavailable rather than silently inventing a
    class label.  The training lookup uses this manifest to retain distinct
    RGB, geometry and plane treatment.
    """
    if tree_support_policy not in {"error", "neutral", "legacy_zero"}:
        raise ValueError("tree_support_policy must be error, neutral, or legacy_zero")
    if not 0.0 <= neutral_tree_support <= 1.0:
        raise ValueError("neutral_tree_support must be in [0, 1]")
    dataset = Path(dataset).resolve()
    mask_pickle = Path(mask_pickle).resolve()
    tree_mask = Path(tree_mask_pickle).resolve() if tree_mask_pickle is not None else None
    if not (dataset / "name_mapping.json").is_file():
        raise FileNotFoundError(f"Missing staged dataset mapping: {dataset / 'name_mapping.json'}")
    if not mask_pickle.is_file():
        raise FileNotFoundError(mask_pickle)
    if tree_mask is not None and not tree_mask.is_file():
        raise FileNotFoundError(tree_mask)

    payload: dict[str, Any] = {
        "schema_version": TASK_SEMANTIC_POLICY_VERSION,
        "dataset": str(dataset),
        "mask_pickle": str(mask_pickle),
        "tree_mask_pickle": str(tree_mask) if tree_mask is not None else None,
        "input_hashes": {
            "mask_pickle": _sha256(mask_pickle),
            "tree_mask_pickle": _sha256(tree_mask),
        },
        "class_availability": {
            "building": "base_static_mask_proxy",
            "ground": "base_static_mask_proxy",
            "rigid": "base_static_mask_proxy",
            "trunk": "unlabelled",
            "canopy": "tree_mask_index_3" if tree_mask is not None else "unlabelled",
            "sky": "base_mask_index_1",
            "transient": "base_mask_index_0",
            "boundary_uncertain": "distance_transform_feather",
        },
        "task_fields": {
            "chart_candidate_weight": {
                "base_keep_indices": [0, 1, 2],
                "canopy": "soft_penalty_only",
            },
            "matching_weight": {
                "base_keep_indices": [0, 1, 2],
                "canopy": "excluded_when_available",
                "boundary_feather_pixels": 6,
            },
            "alignment_weight": {
                "base_keep_indices": [0, 1, 2],
                "canopy": "excluded_when_available",
                "boundary_feather_pixels": 6,
            },
            "plane_seed_weight": {
                "base_keep_indices": [0, 1, 2],
                "canopy": "zero",
                "boundary_feather_pixels": 6,
            },
            "depth_anchor_weight": {
                "base_keep_indices": [0, 1, 2],
                "canopy": "low_confidence",
            },
            "rgb_weight": {
                "base_keep_indices": [0, 1, 2],
                "canopy": "soft_weight",
                "sky": "background_branch",
                "boundary_feather_pixels": 4,
            },
            "densification_weight": {
                "base_keep_indices": [0, 1, 2],
                "canopy": "restricted_real_view_only",
            },
        },
        "tree_support": {
            "policy": tree_support_policy,
            "neutral_prior": neutral_tree_support,
            "unknown_is_not_zero_support": True,
        },
        # This manifest describes task weights for the present single-2DGS
        # implementation.  Do not label it as a hybrid surface/volume/sky
        # representation until those independent fields actually exist.
        "representation_policy": "fixed_camera_structural_chart_single_2dgs_v2",
        "implementation_scope": {
            "independent_semantic_fields": False,
            "separate_surface_foliage_sky_models": False,
            "tree_handling": "mask_and_support_weighting_only",
        },
        "generation_policy": "none",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
