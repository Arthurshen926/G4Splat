#!/usr/bin/env python
"""Build fixed-view before/after panels for native foliage replacement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _read(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _find(root, index, suffix):
    matches = sorted(root.glob(f"{index:05d}_*_{suffix}.png"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {index:05d} *_{suffix}.png in {root}, got {matches}"
        )
    return matches[0]


def _metrics(prediction, target):
    error = prediction - target
    mse = float(np.mean(error**2))
    gray = np.mean(prediction, axis=2)
    target_gray = np.mean(target, axis=2)
    gradient = np.hypot(
        np.diff(gray, axis=1, append=gray[:, -1:]),
        np.diff(gray, axis=0, append=gray[-1:, :]),
    )
    target_gradient = np.hypot(
        np.diff(target_gray, axis=1, append=target_gray[:, -1:]),
        np.diff(target_gray, axis=0, append=target_gray[-1:, :]),
    )
    return {
        "psnr": float(-10.0 * np.log10(max(mse, 1e-12))),
        "mae": float(np.mean(np.abs(error))),
        "gradient_mae": float(np.mean(np.abs(gradient - target_gradient))),
        "high_frequency_energy_ratio": float(
            np.mean(gradient) / max(float(np.mean(target_gradient)), 1e-8)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--indices", default="408,409,410")
    args = parser.parse_args()
    run = args.run.resolve()
    before = run / "visualization" / "before"
    after = run / "visualization" / "after"
    output = run / "visualization" / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    indices = [int(value) for value in args.indices.split(",")]
    audit = {}

    fig, axes = plt.subplots(
        len(indices), 6, figsize=(24, 4.5 * len(indices)), constrained_layout=True
    )
    if len(indices) == 1:
        axes = axes[None]
    for row, index in enumerate(indices):
        target = _read(_find(after, index, "ground_truth"))
        parent = _read(_find(before, index, "canonical"))
        canonical = _read(_find(after, index, "canonical"))
        conditioned = _read(_find(after, index, "conditioned"))
        alpha = _read(_find(after, index, "volume_alpha"))
        parent_error = np.mean(np.abs(parent - target), axis=2)
        new_error = np.mean(np.abs(canonical - target), axis=2)
        scale = max(float(np.quantile(parent_error, 0.98)), 1e-6)
        improvement = np.clip(
            (parent_error - new_error) / scale * 0.5 + 0.5, 0, 1
        )
        panels = (
            (target, "Ground truth"),
            (parent, "68k parent"),
            (canonical, "Native mixed canonical"),
            (conditioned, "Conditioned appearance"),
            (alpha, "3D foliage alpha"),
            (
                plt.get_cmap("coolwarm")(improvement)[..., :3],
                "Error change (blue worse / red better)",
            ),
        )
        for axis, (image, title) in zip(axes[row], panels):
            axis.imshow(image)
            axis.set_title(title)
            axis.axis("off")
        before_metrics = _metrics(parent, target)
        after_metrics = _metrics(canonical, target)
        audit[str(index)] = {
            "before": before_metrics,
            "after_canonical": after_metrics,
            "after_conditioned": _metrics(conditioned, target),
            "delta": {
                key: after_metrics[key] - before_metrics[key]
                for key in before_metrics
            },
        }
        axes[row, 0].set_ylabel(f"{index:05d}", fontsize=14)
    figure_path = output / "00408_00410_native_mixed_comparison.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    (output / "00408_00410_visual_analysis.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"figure": str(figure_path), "metrics": audit}, indent=2))


if __name__ == "__main__":
    main()
