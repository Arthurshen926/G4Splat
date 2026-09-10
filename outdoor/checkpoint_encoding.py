"""Lossless I/O encodings; these must not alter training state semantics."""
from __future__ import annotations

import torch


def compact_static_birth_vectors(state: dict) -> dict:
    """Avoid millions of individual torch storages for three-float CPU rows.

    StaticRayBirthAccumulator.restore already accepts ordinary vector lists
    through torch.as_tensor(...).float().cpu(). Float32 -> Python float ->
    float32 is exact. Camera IDs, witness budgets, cell order, weights and all
    other state remain unchanged. The input is never mutated.
    """
    def vector(value):
        if torch.is_tensor(value):
            if value.numel() != 3:
                raise ValueError("Static birth checkpoint vectors must have three entries")
            return value.detach().reshape(3).cpu().tolist()
        return value

    result = dict(state)
    for name in ("cells", "visual_hull_cells"):
        if name not in state:
            continue
        rows = []
        for source in state[name]:
            row = dict(source)
            row["weighted_center"] = vector(source["weighted_center"])
            row["weighted_color"] = vector(source["weighted_color"])
            if "depth_constraints" in source:
                row["depth_constraints"] = {
                    camera:(vector(constraint[0]),constraint[1],constraint[2])
                    for camera,constraint in source["depth_constraints"].items()
                }
            rows.append(row)
        result[name] = rows
    result["vector_encoding"] = "lossless_plain_three_float_lists_v1"
    return result
