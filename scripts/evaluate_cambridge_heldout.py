#!/usr/bin/env python3
"""Render a held-out Cambridge trajectory without fitting its appearance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from outdoor.scene_contract import validate_disjoint_contracts


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    print("[CMD] " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--mask-pickle", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--database-contract", type=Path)
    parser.add_argument("--heldout-contract", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.database_contract is None) != (args.heldout_contract is None):
        raise ValueError("Provide both camera contracts or neither")
    disjointness = None
    if args.database_contract is not None:
        disjointness = validate_disjoint_contracts(args.database_contract, args.heldout_contract)
    render_dir = args.output / f"ours_{args.iteration}"
    metric_path = render_dir / "rgb_metrics.json"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("PYTHONUNBUFFERED", "1")
    _run(
        [
            sys.executable,
            "2d-gaussian-splatting/render.py",
            "-s",
            str(args.dataset_path),
            "-m",
            str(args.model_path),
            "--iteration",
            str(args.iteration),
            "--data_device",
            "cpu",
            "--rgb_only",
            "--skip_test",
            "--skip_mesh",
            "--disable-color-correction",
            "--output_dir",
            str(render_dir),
        ],
        env=env,
        dry_run=args.dry_run,
    )
    _run(
        [
            sys.executable,
            "scripts/evaluate_render_dir.py",
            str(render_dir),
            "--dataset-path",
            str(args.dataset_path),
            "--mask-pickle",
            str(args.mask_pickle),
            "--output",
            str(metric_path),
            "--device",
            "cuda",
        ],
        env=env,
        dry_run=args.dry_run,
    )
    manifest = {
        "evaluation_kind": "trajectory_heldout_no_appearance_fit",
        "model_path": str(args.model_path.resolve()),
        "dataset_path": str(args.dataset_path.resolve()),
        "iteration": args.iteration,
        "color_correction": "disabled_for_heldout",
        "metrics": str(metric_path),
        "database_query_disjointness": disjointness,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "heldout_evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
