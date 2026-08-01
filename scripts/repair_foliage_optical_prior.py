#!/usr/bin/env python
"""Separate observed leaf optical existence from metric-depth authority.

This is a deterministic metadata-only initialization migration.  It keeps
every primitive centre, scale, colour, role, ray and camera contract intact;
only exact calibrated dense-ray dynamic alpha is lifted to the
observation-existence prior used by the current trainer. DAV2-only rows do
not receive this stronger owner-pixel optical authority.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from outdoor.foliage_geometry import (  # noqa: E402
    evidence_conditioned_leaf_optical_mass,
)
from outdoor.role_aware_initialization import (  # noqa: E402
    RIGID_CALIBRATED_INITIALIZATION_VERSION,
    SOURCE_DENSE_RAY,
)
from outdoor.scene_contract import sha256_file  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _quantiles(values: torch.Tensor) -> list[float]:
    if not int(values.numel()):
        return []
    return [
        float(value)
        for value in torch.quantile(
            values.float(),
            torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0]),
        ).tolist()
    ]


def _repair_opacity_payload(payload: dict) -> dict:
    required = {
        "opacities",
        "layer_role",
        "initialization_source",
        "occupancy_probability",
        "support_view_count",
        "unknown_view_count",
        "ray_depth_nll",
        "free_space_violation_count",
    }
    missing = required - set(payload)
    if missing:
        raise RuntimeError(
            f"Foliage seed lacks optical-prior fields: {sorted(missing)}"
        )
    opacities = torch.as_tensor(payload["opacities"]).clone()
    if opacities.ndim != 2 or opacities.shape[1] != 1:
        raise RuntimeError("Foliage opacities must have shape [N, 1]")
    layer_role = torch.as_tensor(payload["layer_role"]).reshape(-1)
    source = torch.as_tensor(payload["initialization_source"]).reshape(-1)
    exact_owner_dynamic = (layer_role == 2) & (
        source == SOURCE_DENSE_RAY
    )
    optical_prior, geometry_floor = evidence_conditioned_leaf_optical_mass(
        payload["occupancy_probability"],
        payload["support_view_count"],
        payload["unknown_view_count"],
        payload["ray_depth_nll"],
        payload["free_space_violation_count"],
    )
    optical_prior = optical_prior.to(opacities)
    before = opacities[:, 0].clone()
    opacities[exact_owner_dynamic, 0] = torch.maximum(
        before[exact_owner_dynamic], optical_prior[exact_owner_dynamic]
    )
    lifted = exact_owner_dynamic & (
        opacities[:, 0] > before + 1.0e-8
    )
    payload = dict(payload)
    payload["opacities"] = opacities
    audit = dict(payload.get("audit", {}))
    audit["optical_existence_prior_repair"] = {
        "contract": (
            "exact_owner_pixel_calibrated_optical_existence_separate_from_"
            "metric_depth_authority__dav2_only_unchanged__no_rgb_fit"
        ),
        "candidate_count": int(exact_owner_dynamic.sum()),
        "lifted_count": int(lifted.sum()),
        "before_quantiles": _quantiles(before[exact_owner_dynamic]),
        "after_quantiles": _quantiles(
            opacities[exact_owner_dynamic, 0]
        ),
        "geometry_floor_quantiles": _quantiles(
            geometry_floor[exact_owner_dynamic]
        ),
        "centres_or_rays_changed": False,
        "historical_rgb_fit_used": False,
    }
    audit["dense_dynamic_optical_mass_contract"] = (
        "exact_owner_pixel_optical_existence_prior_separate_from_spatial_"
        "depth_geometry_authority__dav2_only_rows_unchanged__"
        "no_hard_per_candidate_alpha_gate"
    )
    audit["protocol"] = RIGID_CALIBRATED_INITIALIZATION_VERSION
    payload["audit"] = audit
    return payload


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    args = _parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest_path = source / "initialization_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("historical_model_initialization", True)):
        raise RuntimeError("Optical-prior repair rejects historical models")
    foliage_path = Path(manifest["foliage_seed"]).expanduser().resolve()
    surface_path = Path(manifest["surface_seed"]).expanduser().resolve()
    for path in (
        foliage_path,
        foliage_path.with_suffix(".json"),
        surface_path,
        surface_path.with_suffix(".json"),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    payload = torch.load(foliage_path, map_location="cpu")
    repaired = _repair_opacity_payload(payload)
    output.mkdir(parents=True)
    target_surface = output / "surface_seed.npz"
    _link_or_copy(surface_path, target_surface)
    _link_or_copy(
        surface_path.with_suffix(".json"),
        target_surface.with_suffix(".json"),
    )
    target_foliage = output / "foliage_seed_gaussians.pth"
    temporary = output / ".foliage_seed_gaussians.pth.tmp"
    torch.save(repaired, temporary)
    os.replace(temporary, target_foliage)

    sidecar = json.loads(
        foliage_path.with_suffix(".json").read_text(encoding="utf-8")
    )
    sidecar.update(repaired["audit"])
    target_foliage.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
    )
    repair = repaired["audit"]["optical_existence_prior_repair"]
    initialization_contract = dict(
        manifest.get("initialization_contract", {})
    )
    initialization_contract["optical_existence_prior_repair"] = {
        **repair,
        "source_foliage_seed": str(foliage_path),
        "source_foliage_seed_sha256": sha256_file(foliage_path),
    }
    initialization_contract["exact_owner_optical_mass_calibration"] = {
        "minimum": 0.08,
        "evidence_gain": 0.32,
        "maximum": 0.40,
        "source_role": int(SOURCE_DENSE_RAY),
        "dav2_only_rows_changed": False,
        "historical_rgb_fit_used": False,
    }
    manifest.update(
        {
            "version": RIGID_CALIBRATED_INITIALIZATION_VERSION,
            "surface_seed": str(target_surface),
            "foliage_seed": str(target_foliage),
            "foliage": sidecar,
            "initialization_contract": initialization_contract,
            "historical_model_initialization": False,
        }
    )
    causal_reuse = dict(manifest.get("causal_reuse", {}))
    causal_reuse.update(
        {
            "surface_only": True,
            "foliage_geometry_reused_exactly": True,
            "foliage_optical_prior_repaired": True,
        }
    )
    manifest["causal_reuse"] = causal_reuse
    (output / "initialization_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "foliage_seed_sha256": sha256_file(target_foliage),
                **repair,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
