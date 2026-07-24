#!/usr/bin/env python
"""Visualize canonical/conditioned layered foliage and frequency changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scripts.visualize_native_mixed_replacement import _find, _metrics, _read


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
        len(indices), 7, figsize=(28, 4.5 * len(indices)),
        constrained_layout=True,
    )
    if len(indices) == 1:
        axes = axes[None]
    for row, index in enumerate(indices):
        target = _read(_find(after, index, "ground_truth"))
        parent = _read(_find(before, index, "canonical"))
        canonical = _read(_find(after, index, "canonical"))
        conditioned = _read(_find(after, index, "conditioned"))
        canonical_alpha = _read(
            _find(after, index, "canonical_volume_alpha")
        )
        conditioned_alpha = _read(
            _find(after, index, "conditioned_volume_alpha")
        )
        old_error = np.mean(np.abs(parent - target), axis=2)
        new_error = np.mean(np.abs(conditioned - target), axis=2)
        scale = max(float(np.quantile(old_error, 0.98)), 1e-6)
        improvement = np.clip(
            (old_error - new_error) / scale * 0.5 + 0.5, 0, 1
        )
        panels = (
            (target, "Ground truth"),
            (parent, "68k parent"),
            (canonical, "Localization canonical"),
            (conditioned, "Sequence/time conditioned"),
            (canonical_alpha, "Porous crown + skeleton α"),
            (conditioned_alpha, "Crown + dynamic leaf α"),
            (
                plt.get_cmap("coolwarm")(improvement)[..., :3],
                "Error change (blue worse / red better)",
            ),
        )
        for axis, (image, title) in zip(axes[row], panels):
            axis.imshow(image)
            axis.set_title(title)
            axis.axis("off")
        parent_metrics = _metrics(parent, target)
        canonical_metrics = _metrics(canonical, target)
        conditioned_metrics = _metrics(conditioned, target)
        audit[str(index)] = {
            "parent": parent_metrics,
            "canonical": canonical_metrics,
            "conditioned": conditioned_metrics,
            "canonical_delta": {
                key: canonical_metrics[key] - parent_metrics[key]
                for key in parent_metrics
            },
            "conditioned_delta": {
                key: conditioned_metrics[key] - parent_metrics[key]
                for key in parent_metrics
            },
        }
        axes[row, 0].set_ylabel(f"{index:05d}", fontsize=14)
    figure = output / "00408_00410_layered_v6_comparison.png"
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    metrics = output / "00408_00410_layered_v6_analysis.json"
    metrics.write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"figure": str(figure), "metrics": audit}, indent=2))


if __name__ == "__main__":
    main()
