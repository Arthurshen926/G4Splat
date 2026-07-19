#!/usr/bin/env python3
"""Export an auditable See3D artifact-repair bundle for each pseudo-view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image


def read_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def read_mask(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127


def copy_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def mask_rgb(mask: np.ndarray) -> np.ndarray:
    return np.repeat((np.uint8(mask) * 255)[..., None], 3, axis=2)


def unavailable_rgb(shape: tuple[int, int], text: str = "N/A") -> np.ndarray:
    image = np.full((shape[0], shape[1], 3), 190, dtype=np.uint8)
    cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (70, 70, 70), 2, cv2.LINE_AA)
    return image


def tile(image: np.ndarray, label: str, size: tuple[int, int] = (320, 180)) -> np.ndarray:
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    canvas = np.full((size[1] + 30, size[0], 3), 245, dtype=np.uint8)
    canvas[30:] = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    cv2.putText(canvas, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def mean_abs_difference(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    residual = np.abs(first.astype(np.float32) - second.astype(np.float32)) / 255.0
    return float(residual[mask].mean())


def optional_rgb(path: Path, fallback: np.ndarray) -> tuple[np.ndarray, bool]:
    return (read_rgb(path), True) if path.is_file() else (fallback.copy(), False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition-dir", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--composite-dir", type=Path, default=None)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    stage_root = args.stage_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and args.replace:
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = stage_root / "artifact_guided_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    else:
        source_root = stage_root / "select-gs"
        indices = sorted(
            int(path.stem.removeprefix("ori_warp_frame"))
            for path in source_root.glob("ori_warp_frame*.png")
        )
        reference_root_path = stage_root.parent / "ref-views"
        manifest = {
            "mode": "generic_see3d_stage",
            "reference_views": [
                {"image_name": path.stem, "path": str(path.resolve())}
                for path in sorted(reference_root_path.glob("*"))
                if path.is_file()
            ],
            "pseudo_views": [
                {"pseudo_view_index": index, "classification": "visibility_or_artifact_hole"}
                for index in indices
            ],
        }
    projective_report_path = stage_root / "projective_repair_report.json"
    projective_report = (
        json.loads(projective_report_path.read_text()) if projective_report_path.is_file() else {"views": []}
    )
    projective_by_index = {int(view["index"]): view for view in projective_report.get("views", [])}

    reference_root = output / "references"
    reference_records = []
    for index, reference in enumerate(manifest.get("reference_views", [])):
        source = Path(reference["path"])
        destination = reference_root / f"{index:02d}_{source.name}"
        copy_image(source, destination)
        reference_records.append({**reference, "bundle_path": str(destination.relative_to(output))})

    records = []
    for view in manifest.get("pseudo_views", []):
        index = int(view["pseudo_view_index"])
        stem = f"{index:06d}"
        view_root = output / f"view_{stem}"
        source_root = stage_root / "select-gs"
        condition_root = args.condition_dir or (
            stage_root / "select-gs-projective-condition"
            if (stage_root / "select-gs-projective-condition").is_dir()
            else stage_root / "select-gs-repair"
        )
        base = read_rgb(source_root / f"ori_warp_frame{stem}.png")
        original_known = read_mask(source_root / f"mask_frame{stem}.png")
        condition, has_condition = optional_rgb(
            condition_root / f"warp_frame{stem}.png",
            read_rgb(source_root / f"warp_frame{stem}.png"),
        )
        condition_known = (
            read_mask(condition_root / f"mask_frame{stem}.png")
            if (condition_root / f"mask_frame{stem}.png").is_file()
            else original_known
        )
        support_path = condition_root / f"projective_support_frame{stem}.png"
        has_projective_support = support_path.is_file()
        support_mask = read_mask(support_path) if has_projective_support else np.zeros_like(original_known)
        artifact_path = condition_root / f"artifact_edit_frame{stem}.png"
        artifact_mask = read_mask(artifact_path) if artifact_path.is_file() else np.zeros_like(original_known)
        raw, has_raw = optional_rgb(
            (args.raw_dir or stage_root / "select-gs-inpainted")
            / f"predict_warp_frame{stem}.png",
            base,
        )
        composite, has_composite = optional_rgb(
            (args.composite_dir or stage_root / "select-gs-inpainted-merged")
            / f"predict_warp_frame{stem}.png",
            base,
        )
        projective, has_projective = optional_rgb(
            stage_root / "select-gs-projective-merged" / f"predict_warp_frame{stem}.png", base
        )

        images = {
            "01_base_render.png": base,
            "02_original_known_mask.png": mask_rgb(original_known),
            "03_original_edit_mask.png": mask_rgb(~original_known),
            "04_projective_support.png": (
                mask_rgb(support_mask)
                if has_projective_support
                else unavailable_rgb(original_known.shape, "NO PROJECTIVE WARP")
            ),
            "05_condition.png": condition,
            "06_residual_edit_mask.png": mask_rgb(~condition_known),
            "07_artifact_edit_mask.png": mask_rgb(artifact_mask),
            "08_see3d_raw.png": raw,
            "09_see3d_composite.png": composite,
            "10_projective_repair.png": projective,
        }
        view_root.mkdir(parents=True, exist_ok=True)
        for name, image in images.items():
            Image.fromarray(image, mode="RGB").save(view_root / name)

        labels = [
            "base render",
            "original known",
            "original edit",
            "real-warp support" if has_projective_support else "real-warp support (N/A)",
            "See3D condition",
            "residual edit",
            "artifact edit (actual)",
            "See3D raw",
            "See3D composite",
            "projective repair",
        ]
        sheet = np.concatenate([tile(image, label) for image, label in zip(images.values(), labels)], axis=1)
        cv2.imwrite(str(view_root / "contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])

        original_edit = ~original_known
        residual_edit = ~condition_known
        record = {
            "index": index,
            "classification": view.get("classification"),
            "target_image_name": view.get("target_image_name"),
            "original_edit_fraction": float(original_edit.mean()),
            "projective_support_fraction": float(support_mask.mean()),
            "has_projective_support": has_projective_support,
            "artifact_edit_fraction": float(artifact_mask.mean()),
            "residual_edit_fraction": float(residual_edit.mean()),
            "has_projective_condition": has_condition,
            "has_see3d_raw": has_raw,
            "has_see3d_composite": has_composite,
            "has_projective_repair": has_projective,
            "see3d_raw_change_known_mae": mean_abs_difference(raw, base, original_known),
            "see3d_raw_change_edit_mae": mean_abs_difference(raw, base, original_edit),
            "composite_change_known_mae": mean_abs_difference(composite, base, original_known),
            "projective_change_known_mae": mean_abs_difference(projective, base, original_known),
            "projective_report": projective_by_index.get(index),
            "bundle_root": str(view_root.relative_to(output)),
        }
        records.append(record)

    report = {
        "version": 1,
        "mode": "see3d_artifact_repair_audit_bundle",
        "stage_root": str(stage_root),
        "references": reference_records,
        "views": records,
    }
    (output / "bundle_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
