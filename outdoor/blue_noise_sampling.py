"""Deterministic, camera-dephased sampling for foliage observations."""

from __future__ import annotations

import hashlib

import numpy as np


def stable_sampling_seed(*parts: object) -> int:
    """Return a process-independent 64-bit seed for one observation owner."""
    digest = hashlib.sha256(
        "\x1f".join(map(str, parts)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _coordinate_priority(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> np.ndarray:
    """SplitMix64 priorities keyed by coordinates, not input row order."""
    x = np.asarray(np.rint(x), dtype=np.uint64)
    y = np.asarray(np.rint(y), dtype=np.uint64)
    with np.errstate(over="ignore"):
        value = (
            x * np.uint64(0xD6E8FEB86659FD93)
            ^ y * np.uint64(0xA5A3564E27F8862D)
            ^ np.uint64(seed)
        )
        value ^= value >> np.uint64(30)
        value *= np.uint64(0xBF58476D1CE4E5B9)
        value ^= value >> np.uint64(27)
        value *= np.uint64(0x94D049BB133111EB)
        value ^= value >> np.uint64(31)
    # Fifty-three high bits are exactly representable as float64.
    return (value >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def deterministic_blue_noise_rows(
    indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    count: int,
    *,
    seed: int,
    score: np.ndarray | None = None,
) -> np.ndarray:
    """Select a reproducible Poisson-like subset without an axis-aligned grid.

    Candidate priority is a coordinate hash dephased by camera/instance.  A
    greedy inhibition radius supplies coverage; the radius is relaxed only
    when a sparse or thin mask cannot fill the requested quota.  The result is
    invariant to candidate array order and has no shared grid phase between
    cameras.
    """
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    count = min(max(int(count), 0), len(indices))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count == len(indices):
        return np.sort(indices)
    x_all = np.asarray(x, dtype=np.float64).reshape(-1)
    y_all = np.asarray(y, dtype=np.float64).reshape(-1)
    if max(indices, default=-1) >= len(x_all) or len(x_all) != len(y_all):
        raise ValueError("indices and coordinate arrays do not align")
    px = x_all[indices]
    py = y_all[indices]
    hashed = _coordinate_priority(px, py, int(seed))
    if score is None:
        priority = hashed
    else:
        values = np.asarray(score, dtype=np.float64).reshape(-1)[indices]
        finite = np.isfinite(values)
        if bool(finite.any()):
            low, high = np.quantile(values[finite], [0.05, 0.95])
            normalized = np.clip(
                (np.nan_to_num(values, nan=low) - low)
                / max(float(high - low), 1.0e-12),
                0.0,
                1.0,
            )
        else:
            normalized = np.zeros_like(values)
        # Image evidence dominates detail selection while the hash breaks
        # equal-gradient runs without introducing a common raster phase.
        priority = normalized + 0.15 * hashed
    order = np.lexsort((indices, -priority))

    # One candidate represents roughly one raster pixel.  The initial radius
    # is intentionally conservative; thin branches automatically relax it.
    radius = max(1.0, 0.72 * np.sqrt(len(indices) / float(count)))
    chosen: list[int] = []
    for _ in range(8):
        chosen = []
        buckets: dict[tuple[int, int], list[int]] = {}
        radius2 = radius * radius
        inverse = 1.0 / max(radius, 1.0e-12)
        for local in order:
            cell_x = int(np.floor(px[local] * inverse))
            cell_y = int(np.floor(py[local] * inverse))
            accepted = True
            for adjacent_y in range(cell_y - 1, cell_y + 2):
                for adjacent_x in range(cell_x - 1, cell_x + 2):
                    for other in buckets.get((adjacent_x, adjacent_y), ()):
                        dx = px[local] - px[other]
                        dy = py[local] - py[other]
                        if dx * dx + dy * dy < radius2:
                            accepted = False
                            break
                    if not accepted:
                        break
                if not accepted:
                    break
            if not accepted:
                continue
            chosen.append(int(local))
            buckets.setdefault((cell_x, cell_y), []).append(int(local))
            if len(chosen) == count:
                break
        if len(chosen) == count:
            break
        radius *= 0.72
    if len(chosen) < count:
        selected_local = set(chosen)
        chosen.extend(
            int(local)
            for local in order
            if int(local) not in selected_local
        )
    return np.sort(indices[np.asarray(chosen[:count], dtype=np.int64)])
