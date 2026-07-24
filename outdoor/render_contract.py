"""Zero-step renderer identity and implementation provenance utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


RENDER_FIELDS = (
    "render",
    "rend_alpha",
    "rend_depth",
    "rend_depth_median",
    "radii",
)


@torch.no_grad()
def capture_render_contract(render_fn, views, model, pipe, background, indices):
    records = []
    for index in indices:
        package = render_fn(views[index], model, pipe, background)
        records.append(
            {
                "index": int(index),
                "image_name": str(views[index].image_name),
                **{
                    key: package[key].detach().cpu().clone()
                    for key in RENDER_FIELDS
                },
            }
        )
    return records


def compare_render_contract(before, after, *, atol=0.0, rtol=0.0):
    if len(before) != len(after):
        raise RuntimeError("Render contract view counts differ")
    report = {"passed": True, "atol": float(atol), "rtol": float(rtol), "views": []}
    for expected, actual in zip(before, after):
        if (
            expected["index"] != actual["index"]
            or expected["image_name"] != actual["image_name"]
        ):
            raise RuntimeError("Render contract view identities differ")
        row = {
            "index": expected["index"],
            "image_name": expected["image_name"],
            "fields": {},
        }
        for key in RENDER_FIELDS:
            lhs, rhs = expected[key], actual[key]
            if lhs.shape != rhs.shape:
                equal = False
                maximum = float("inf")
                mean = float("inf")
            elif lhs.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                equal = bool(torch.equal(lhs, rhs))
                difference = (lhs.to(torch.int64) - rhs.to(torch.int64)).abs()
                maximum = float(difference.max().item()) if difference.numel() else 0.0
                mean = float(difference.float().mean().item()) if difference.numel() else 0.0
            else:
                difference = (lhs - rhs).abs()
                equal = bool(torch.allclose(lhs, rhs, atol=atol, rtol=rtol))
                maximum = float(difference.max().item()) if difference.numel() else 0.0
                mean = float(difference.mean().item()) if difference.numel() else 0.0
            row["fields"][key] = {
                "equal": equal,
                "max_abs": maximum,
                "mean_abs": mean,
            }
            report["passed"] = report["passed"] and equal
        report["views"].append(row)
    return report


def assert_render_contract(report):
    if report["passed"]:
        return
    failures = []
    for view in report["views"]:
        for field, stats in view["fields"].items():
            if not stats["equal"]:
                failures.append(
                    f"{view['image_name']}:{field} max_abs={stats['max_abs']}"
                )
    raise RuntimeError(
        "Zero-step renderer identity failed: " + ", ".join(failures[:8])
    )


def directory_sha256(path: str | Path, *, suffixes=None):
    root = Path(path)
    if suffixes is not None:
        suffixes = set(suffixes)
    digest = hashlib.sha256()
    files = [
        item
        for item in root.rglob("*")
        if item.is_file() and (suffixes is None or item.suffix in suffixes)
    ]
    for item in sorted(files):
        relative = item.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "root": str(root.resolve()),
        "file_count": len(files),
        "sha256": digest.hexdigest(),
    }


def declared_mip_filter(parent_contract, model):
    declared = parent_contract.get("input", {}).get("mip_filter")
    if declared == "disabled":
        model.set_mip_filter(False)
        return {"declared": declared, "restored": False, "source": "parent_contract"}
    if declared == "enabled":
        if not getattr(model, "use_mip_filter", False) or not hasattr(model, "mip_filter"):
            raise RuntimeError(
                "Parent declares enabled Mip filtering but its training state "
                "does not carry a restorable mip_filter tensor"
            )
        return {"declared": declared, "restored": True, "source": "model_state"}
    raise RuntimeError(f"Unsupported parent Mip filter contract: {declared!r}")
