#!/usr/bin/env python3
"""Build source-separated MoGe3 alternatives for the MAtCha Chart base."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matcha.cambridge_masks import CambridgeMaskLookup  # noqa: E402
from outdoor.moge3_chart_base import (  # noqa: E402
    MOGE3_CHART_BASE_VERSION,
    chart_base_variants,
    robust_scene_scale,
)
from outdoor.moge3_evidence import (  # noqa: E402
    load_index as load_moge3_index,
    load_view as load_moge3_view,
    sha256_file,
)
from outdoor.moge3_depth_layers import reduce_depth_layers


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resize_scalar(
    value: np.ndarray,
    valid: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(value, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(value)
    source = torch.from_numpy(np.where(valid, value, 0.))[None, None]
    mask = torch.from_numpy(valid.astype(np.float32))[None, None]
    numerator = F.interpolate(source * mask, size=shape, mode="area")
    denominator = F.interpolate(mask, size=shape, mode="area")
    resized = numerator / denominator.clamp_min(1.0e-6)
    supported = denominator >= 0.50
    return resized[0, 0].numpy(), supported[0, 0].numpy()


def _resize_normal(
    value: np.ndarray,
    valid: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(value, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(value).all(-1)
    source = torch.from_numpy(np.where(valid[..., None], value, 0.)).permute(2, 0, 1)[None]
    mask = torch.from_numpy(valid.astype(np.float32))[None, None]
    numerator = F.interpolate(source * mask, size=shape, mode="area")
    denominator = F.interpolate(mask, size=shape, mode="area")
    resized = numerator / denominator.clamp_min(1.0e-6)
    norm = resized.norm(dim=1, keepdim=True)
    supported = (denominator >= 0.50) & torch.isfinite(norm) & (
        norm > 1.0e-5
    )
    resized = torch.where(
        supported,
        resized / norm.clamp_min(1.0e-6),
        torch.zeros_like(resized),
    )
    return resized[0].permute(1, 2, 0).numpy(), supported[0, 0].numpy()


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--charts-data", type=Path, required=True)
    parser.add_argument("--chart-cameras", type=Path, required=True)
    parser.add_argument("--moge3-index", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tree-mask-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--depth-evidence-policy', choices=('legacy', 'layered'), default='legacy',
                        help='Layered: preserve native depth proposals, reject mixed-cell base replacement, decouple normal validity')
    args = parser.parse_args()

    charts_path = args.charts_data.expanduser().resolve()
    cameras_path = args.chart_cameras.expanduser().resolve()
    index_path = args.moge3_index.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    tree_mask = args.tree_mask_pickle.expanduser().resolve()
    output = args.output.expanduser().resolve()
    chart_cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    names = [Path(value).name for value in chart_cameras["filepaths"]]
    stems = [Path(value).stem for value in names]
    index = load_moge3_index(index_path, verify_views=False)
    records = dict(index["records"])
    missing = sorted(set(stems) - set(records))
    if missing:
        raise RuntimeError(
            "MoGe3 index lacks MAtCha cameras: " + ", ".join(missing[:5])
        )
    masks = CambridgeMaskLookup(
        dataset, tree_mask, mask_indices=[0, 1, 2, 3]
    )

    with np.load(charts_path, allow_pickle=False) as charts:
        chart_depth = charts["depths"].astype(np.float32)
        reference_depth = charts.get(
            "reference_depths", chart_depth
        ).astype(np.float32)
        chart_confidence = charts["confs"].astype(np.float32)
        scale_factor = float(charts["scale_factor"])
        reference_mask = charts.get(
            "alignment_reference_mask",
            np.ones_like(chart_depth, dtype=bool),
        ).astype(bool)
        active = charts.get(
            "quality_selection_active",
            np.ones(len(chart_depth), dtype=bool),
        ).astype(bool)
        active &= charts.get(
            "alignment_gate_valid",
            np.ones(len(chart_depth), dtype=bool),
        ).astype(bool)
    if not np.isfinite(scale_factor) or scale_factor <= 0.0:
        raise RuntimeError("MAtCha chart scale_factor is invalid")
    if len(names) != len(chart_depth):
        raise RuntimeError("MAtCha camera and Chart counts differ")
    shape = tuple(int(value) for value in chart_depth.shape[1:])
    matcha_metric = chart_depth / scale_factor
    reference_metric = reference_depth / scale_factor

    raw_depths: list[np.ndarray] = []
    refinement_sigmas: list[np.ndarray] = []
    direct_normals: list[np.ndarray] = []
    depth_normals: list[np.ndarray] = []
    depth_normal_valids: list[np.ndarray] = []
    depth_valids: list[np.ndarray] = []
    layer_records: list[dict] = []
    rigid_masks: list[np.ndarray] = []
    calibration_samples: list[np.ndarray] = []
    for chart_index, (name, stem) in enumerate(zip(names, stems)):
        record = records[stem]
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"Indexed MoGe3 view changed: {path}")
        if sha256_file(path) != str(record["sha256"]):
            raise RuntimeError(f"Indexed MoGe3 hash changed: {path}")
        view = load_moge3_view(path)
        valid = (
            view["valid_mask"].astype(bool)
            & view["refinement_valid_mask"].astype(bool)
        )
        depth, depth_valid = _resize_scalar(view["depth_m"], valid, shape)
        refinement_std, refinement_valid = _resize_scalar(
            view["refinement_log_depth_std"], valid, shape
        )
        refinement_delta, delta_valid = _resize_scalar(
            view["refinement_final_delta_log_depth"], valid, shape
        )
        direct, direct_valid = _resize_normal(
            view["normal_direct_camera"], valid, shape
        )
        derived_source_valid = valid & view[
            "depth_normal_valid_mask"
        ].astype(bool)
        derived, derived_valid = _resize_normal(
            view["normal_depth_exact_k_camera"],
            derived_source_valid,
            shape,
        )
        combined_valid = (
            depth_valid
            & refinement_valid
            & delta_valid
            & direct_valid
            & derived_valid
        )
        independent_valid = depth_valid & refinement_valid & delta_valid
        if args.depth_evidence_policy == 'layered':
            layers = reduce_depth_layers(view['depth_m'], valid, shape)
            layer_records.append(layers)
            independent_valid &= layers['continuous_valid']
        rigid = masks.get_mask(
            name, shape, torch.device("cpu")
        ).numpy()
        rigid &= bool(active[chart_index])
        calibration_valid = (
            rigid
            & reference_mask[chart_index]
            & combined_valid
            & np.isfinite(reference_metric[chart_index])
            & (reference_metric[chart_index] > 0.05)
            & np.isfinite(depth)
            & (depth > 0.05)
            & np.isfinite(chart_confidence[chart_index])
            & (chart_confidence[chart_index] > 0.0)
        )
        if args.depth_evidence_policy == 'layered':
            calibration_valid &= independent_valid
        calibration_samples.append(
            np.log(
                reference_metric[chart_index][calibration_valid]
                / depth[calibration_valid]
            )
        )
        raw_depths.append(depth)
        refinement_sigmas.append(
            np.sqrt(
                np.maximum(refinement_std, 0.0) ** 2
                + np.maximum(refinement_delta, 0.0) ** 2
            ).astype(np.float32)
        )
        direct_normals.append(direct)
        depth_normals.append(derived)
        depth_normal_valids.append(combined_valid)
        depth_valids.append(independent_valid)
        rigid_masks.append(rigid)

    scale_audit = robust_scene_scale(calibration_samples)
    scale = float(scale_audit["metric_to_cambridge_scale"])
    global_sigma = float(scale_audit["global_log_scale_sigma"])
    outputs = {
        "depth_moge3": [],
        "depth_moge3_adaptive": [],
        "validity": [],
        "moge3_precision": [],
        "moge3_adaptive_weight": [],
        "moge3_depth_valid": [],
        "moge3_normal_valid": [],
    }
    for chart_index in range(len(names)):
        matcha_base = np.where(
            reference_mask[chart_index] & bool(active[chart_index]),
            matcha_metric[chart_index],
            0.0,
        ).astype(np.float32)
        variants = chart_base_variants(
            matcha_base,
            raw_depths[chart_index] * scale,
            rigid_valid=(
                rigid_masks[chart_index]
                & reference_mask[chart_index]
                & (depth_valids[chart_index] if args.depth_evidence_policy == 'layered'
                   else depth_normal_valids[chart_index])
            ),
            refinement_sigma=np.sqrt(
                refinement_sigmas[chart_index] ** 2 + global_sigma**2
            ),
            normal_direct_camera=direct_normals[chart_index],
            normal_depth_camera=depth_normals[chart_index],
            depth_normal_valid=depth_normal_valids[chart_index],
            depth_valid_independent_of_normal=args.depth_evidence_policy == 'layered',
        )
        for key in outputs:
            outputs[key].append(variants[key])
    arrays = {key: np.stack(value) for key, value in outputs.items()}
    if layer_records:
        for key in layer_records[0]:
            arrays['native_layer_' + key] = np.stack([r[key] for r in layer_records])
    metadata = {
        "schema_version": MOGE3_CHART_BASE_VERSION,
        "depth_units": "cambridge_metric_camera_z",
        "matcha_authority": "multiview_reference_and_fallback",
        "moge3_authority": "rigid_valid_exact_camera_base_and_confidence",
        "normal_role": "direct_target_depth_derived_agreement_confidence",
        "adaptive_fusion": "continuous_log_depth_no_binary_quality_gate",
        "scale_calibration": scale_audit,
        "input_hashes": {
            "charts_data": _sha256(charts_path),
            "chart_cameras": _sha256(cameras_path),
            "moge3_index": _sha256(index_path),
            "tree_mask_pickle": _sha256(tree_mask),
        },
        "camera_count": len(names),
        "shape": list(shape),
        "depth_evidence_policy": args.depth_evidence_policy,
        "native_layer_depth_units": "moge_raw_metric_camera_z",
        "native_layer_authority": "unverified_samples_with_original_integer_pixel_indices_not_cell_center_geometry",
    }
    payload = {
        "schema_version": np.asarray(MOGE3_CHART_BASE_VERSION),
        "image_names": np.asarray(names),
        "metadata_json": np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        ),
        **arrays,
    }
    _atomic_npz(output, payload)
    report = {
        **metadata,
        "output": str(output),
        "output_sha256": _sha256(output),
        "moge3_valid_pixels": int((arrays["moge3_precision"] > 0).sum()),
        "adaptive_weight_mean": float(
            arrays["moge3_adaptive_weight"][
                arrays["moge3_adaptive_weight"] > 0
            ].mean()
        )
        if bool((arrays["moge3_adaptive_weight"] > 0).any())
        else 0.0,
    }
    _atomic_json(output.with_suffix(".json"), report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
