#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import shutil
from PIL import Image


REQUIRED_PLY_PROPERTIES = {
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
}


def parse_namespace(path: Path) -> dict[str, object]:
    expression = ast.parse(path.read_text().strip(), mode="eval").body
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        raise ValueError(f"Expected Namespace(...) in {path}")
    if expression.func.id != "Namespace" or expression.args:
        raise ValueError(f"Unsupported cfg_args expression in {path}")
    return {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in expression.keywords
        if keyword.arg is not None
    }


def serialize_namespace(values: dict[str, object]) -> str:
    body = ", ".join(f"{key}={value!r}" for key, value in values.items())
    return f"Namespace({body})"


def read_ply_header(path: Path) -> list[str]:
    lines = []
    with path.open("rb") as file:
        for raw_line in file:
            line = raw_line.decode("ascii").rstrip("\r\n")
            lines.append(line)
            if line == "end_header":
                break
    if not lines or lines[-1] != "end_header":
        raise ValueError(f"Invalid or unterminated PLY header: {path}")
    return lines


def validate_ply(path: Path) -> dict[str, object]:
    header = read_ply_header(path)
    properties = {
        line.split(maxsplit=2)[2]
        for line in header
        if line.startswith("property ") and len(line.split(maxsplit=2)) == 3
    }
    missing = sorted(REQUIRED_PLY_PROPERTIES - properties)
    if missing:
        raise ValueError(f"Warm-start PLY is missing properties: {missing}")
    vertex_line = next(
        (line for line in header if line.startswith("element vertex ")),
        None,
    )
    if vertex_line is None:
        raise ValueError(f"Warm-start PLY has no vertex count: {path}")
    return {
        "vertices": int(vertex_line.rsplit(" ", 1)[1]),
        "properties": sorted(properties),
        "has_mip_filter": "mip_filter" in properties,
        "size_bytes": path.stat().st_size,
    }


def link_read_only(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        raise FileExistsError(f"Adapter destination already exists: {destination}")
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def prepare_adapter(
    *,
    matcha_scene: Path,
    matcha_model: Path,
    output: Path,
    iteration: int,
    legacy_resolution_cap: int = 1600,
) -> dict[str, object]:
    matcha_scene = matcha_scene.resolve()
    matcha_model = matcha_model.resolve()
    output = output.resolve()
    source_cfg = matcha_model / "cfg_args"
    source_ply = (
        matcha_model
        / "point_cloud"
        / f"iteration_{iteration}"
        / "point_cloud.ply"
    )
    missing = [path for path in (source_cfg, source_ply) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing MAtCha warm-start input(s): " + ", ".join(map(str, missing)))

    cfg = parse_namespace(source_cfg)
    cfg_source = Path(os.path.expanduser(str(cfg.get("source_path", "")))).resolve()
    if cfg_source != matcha_scene:
        raise ValueError(
            "MAtCha model/source coordinate mismatch: "
            f"cfg_args uses {cfg_source}, requested {matcha_scene}"
        )
    scene_required = [
        matcha_scene / "images",
        matcha_scene / "sparse",
        matcha_scene / "charts_data.npz",
    ]
    missing = [path for path in scene_required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing MAtCha scene input(s): " + ", ".join(map(str, missing)))
    ply_info = validate_ply(source_ply)

    scene_dir = output / "mast3r_sfm"
    baseline_model = output / "baseline_model"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Warm-start adapter output is not empty: {output}")
    scene_dir.mkdir(parents=True, exist_ok=True)
    baseline_iteration_dir = baseline_model / "point_cloud" / f"iteration_{iteration}"
    baseline_iteration_dir.mkdir(parents=True, exist_ok=True)

    for name in ("images", "sparse", "pointmaps", "mast3r_sfm"):
        source = matcha_scene / name
        if source.exists():
            link_read_only(source, scene_dir / name)
    for name in ("charts_data.npz", "cameras.json", "points.ply"):
        source = matcha_scene / name
        if source.exists():
            shutil.copy2(source, scene_dir / name)

    link_read_only(source_ply, baseline_iteration_dir / "point_cloud.ply")
    for name in ("cameras.json", "input.ply"):
        source = matcha_model / name
        if source.exists():
            link_read_only(source, baseline_model / name)

    adapter_cfg = dict(cfg)
    adapter_cfg["source_path"] = str(scene_dir)
    adapter_cfg["model_path"] = str(baseline_model)
    translated_resolution = None
    if adapter_cfg.get("resolution") == -1 and legacy_resolution_cap > 0:
        first_image = next(
            (
                path
                for path in sorted((matcha_scene / "images").rglob("*"))
                if path.is_file()
            ),
            None,
        )
        if first_image is not None:
            with Image.open(first_image) as image:
                if image.width > legacy_resolution_cap:
                    adapter_cfg["resolution"] = legacy_resolution_cap
                    translated_resolution = legacy_resolution_cap
    (baseline_model / "cfg_args").write_text(serialize_namespace(adapter_cfg))

    manifest = {
        "version": 1,
        "matcha_scene": str(matcha_scene),
        "matcha_model": str(matcha_model),
        "iteration": iteration,
        "scene_dir": str(scene_dir),
        "baseline_model": str(baseline_model),
        "baseline_ply": str(source_ply),
        "ply": ply_info,
        "coordinate_check": "cfg_source_path_match",
        "legacy_resolution_translation": translated_resolution,
    }
    (output / "warmstart_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare an isolated G4Splat scene around a trained MAtCha PLY."
    )
    parser.add_argument("--matcha_scene", type=Path, required=True)
    parser.add_argument("--matcha_model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--legacy_resolution_cap", type=int, default=1600)
    args = parser.parse_args()
    manifest = prepare_adapter(
        matcha_scene=args.matcha_scene,
        matcha_model=args.matcha_model,
        output=args.output,
        iteration=args.iteration,
        legacy_resolution_cap=args.legacy_resolution_cap,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
