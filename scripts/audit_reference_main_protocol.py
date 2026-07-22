#!/usr/bin/env python3
"""Audit the *released-main* Cambridge reconstruction contract of ULF-Loc/STDLoc.

The local ULF-Loc and STDLoc worktrees are intentionally allowed to contain
ongoing localization work.  Reading them directly therefore cannot establish
what their published Cambridge reconstruction recipe actually was.  This
utility reads every audited file through ``git show REVISION:path`` and emits
the executable, revision-pinned facts that matter for a fair G4Splat control:

* the Gaussian representation requested by the Cambridge launcher;
* its explicit topology/LR overrides;
* whether a nominal 2DGS entry point can construct its model; and
* the renderer and RGB/feature-loss path used by the release trainer.

It is deliberately descriptive rather than a score comparator: released
Cambridge launchers use 3DGS, so treating them as a one-variable 2DGS baseline
would conflate representation, renderer, and objective.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


_REQUIRED = (
    "train.py",
    "scene/__init__.py",
    "scene/gaussian_model.py",
    "gaussian_renderer/__init__.py",
    "arguments/__init__.py",
    "scripts/train_cambridge.sh",
)


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"git -C {root} {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout


def _revision_sources(root: Path, revision: str) -> tuple[str, dict[str, str]]:
    """Return a resolved immutable revision and its required source texts."""
    root = root.resolve()
    resolved = _git_output(root, "rev-parse", revision).strip()
    sources: dict[str, str] = {}
    for relative in _REQUIRED:
        sources[relative] = _git_output(root, "show", f"{resolved}:{relative}")
    return resolved, sources


def _tree(source: str, relative: str) -> ast.Module:
    try:
        return ast.parse(source, filename=relative)
    except SyntaxError as error:
        raise RuntimeError(f"Could not parse {relative}: {error}") from error


def _class_method_arity(tree: ast.AST, class_name: str, method_name: str) -> int | None:
    for node in tree.body if isinstance(tree, ast.Module) else ():
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for member in node.body:
            if isinstance(member, ast.FunctionDef) and member.name == method_name:
                positional = list(member.args.posonlyargs) + list(member.args.args)
                if positional and positional[0].arg == "self":
                    positional = positional[1:]
                if member.args.vararg is not None or member.args.kwarg is not None:
                    raise RuntimeError(
                        f"{class_name}.{method_name} uses variadic arguments; cannot audit arity"
                    )
                return len(positional) + len(member.args.kwonlyargs)
    return None


def _scene_initializer_arity(tree: ast.AST) -> int:
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_from_pcd"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "gaussians"
    ]
    if len(calls) != 1:
        raise RuntimeError(f"Expected one Scene.create_from_pcd call, found {len(calls)}")
    if calls[0].keywords:
        raise RuntimeError("Scene.create_from_pcd uses keywords; positional contract is ambiguous")
    return len(calls[0].args)


def _effective_cull(tree: ast.AST) -> float:
    values: list[float] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "densify_and_prune"
        ):
            continue
        if len(node.args) < 2:
            raise RuntimeError("densify_and_prune call does not pass an opacity cull threshold")
        try:
            values.append(float(ast.literal_eval(node.args[1])))
        except (TypeError, ValueError) as error:
            raise RuntimeError("Reference opacity cull threshold is not literal") from error
    if not values or len(set(values)) != 1:
        raise RuntimeError(f"Expected one literal effective opacity cull, found {values}")
    return values[0]


def _rgb_only_values(tree: ast.AST) -> list[bool | None]:
    values: list[bool | None] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "render_gsplat"
        ):
            continue
        rgb_only = next((item.value for item in node.keywords if item.arg == "rgb_only"), None)
        if rgb_only is None:
            values.append(None)
            continue
        try:
            value = ast.literal_eval(rgb_only)
        except (TypeError, ValueError) as error:
            raise RuntimeError("render_gsplat rgb_only is not a literal") from error
        if not isinstance(value, bool):
            raise RuntimeError(f"render_gsplat rgb_only must be bool, got {value!r}")
        values.append(value)
    return values


def _launcher_commands(source: str) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for line in source.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "train.py" not in line:
            continue
        fields = shlex.split(line)
        if "train.py" not in fields:
            continue
        def _after(flag: str) -> str | None:
            try:
                return fields[fields.index(flag) + 1]
            except (ValueError, IndexError):
                return None

        commands.append(
            {
                "gaussian_type": _after("-g") or _after("--gaussian_type"),
                "iterations": _after("--iterations"),
                "densify_grad_threshold": _after("--densify_grad_threshold"),
                "position_lr_init": _after("--position_lr_init"),
                "scaling_lr": _after("--scaling_lr"),
                "images": _after("--images"),
            }
        )
    if not commands:
        raise RuntimeError("No train.py commands found in scripts/train_cambridge.sh")
    return commands


def _declared_opacity_cull(tree: ast.AST) -> float:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "opacity_cull"
            for target in node.targets
        ):
            try:
                return float(ast.literal_eval(node.value))
            except (TypeError, ValueError) as error:
                raise RuntimeError("Declared opacity cull is not literal") from error
    raise RuntimeError("Could not find OptimizationParams.opacity_cull")


def _attribute_chain(node: ast.AST) -> tuple[str, ...] | None:
    """Return a simple dotted-name chain, or ``None`` for dynamic calls."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return tuple(reversed(parts))


def _training_checkpoint_order(tree: ast.AST) -> dict[str, Any]:
    """Audit observable checkpoint order in the reference training loop.

    A saved point cloud must represent the state used by the corresponding
    evaluation.  A terminal densification or opacity reset after saving is
    harmless only when it is not persisted.  Record the source ordering so
    the control runner can be checked against both released references rather
    than relying on a hand-read code fragment.
    """
    training = next(
        (
            node
            for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "training"
        ),
        None,
    )
    if training is None:
        raise RuntimeError("Could not find training() in reference train.py")

    targets = {
        "scene_save_lines": ("scene", "save"),
        "densify_and_prune_lines": ("gaussians", "densify_and_prune"),
        "reset_opacity_lines": ("gaussians", "reset_opacity"),
        "optimizer_step_lines": ("gaussians", "optimizer", "step"),
    }
    lines = {name: [] for name in targets}
    for node in ast.walk(training):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        for name, expected in targets.items():
            if chain == expected:
                lines[name].append(int(node.lineno))
    for value in lines.values():
        value.sort()

    def _before(first: str, second: str) -> bool | None:
        if not lines[first] or not lines[second]:
            return None
        return min(lines[first]) < min(lines[second])

    lines.update(
        {
            "save_precedes_densify": _before("scene_save_lines", "densify_and_prune_lines"),
            "save_precedes_opacity_reset": _before("scene_save_lines", "reset_opacity_lines"),
            "save_precedes_optimizer_step": _before("scene_save_lines", "optimizer_step_lines"),
        }
    )
    return lines


def audit_sources(*, root: Path, revision: str, sources: dict[str, str]) -> dict[str, Any]:
    """Build an immutable protocol audit from a revision's source texts."""
    train = _tree(sources["train.py"], "train.py")
    scene = _tree(sources["scene/__init__.py"], "scene/__init__.py")
    model = _tree(sources["scene/gaussian_model.py"], "scene/gaussian_model.py")
    renderer = _tree(sources["gaussian_renderer/__init__.py"], "gaussian_renderer/__init__.py")
    arguments = _tree(sources["arguments/__init__.py"], "arguments/__init__.py")
    scene_arity = _scene_initializer_arity(scene)
    initializers = {
        name: _class_method_arity(model, name, "create_from_pcd")
        for name in ("GaussianModel", "GaussianModel_2dgs")
    }
    backend_imports = sorted(
        alias.name
        for node in ast.walk(renderer)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    backend_from_imports = sorted(
        node.module or ""
        for node in ast.walk(renderer)
        if isinstance(node, ast.ImportFrom)
    )
    launcher = _launcher_commands(sources["scripts/train_cambridge.sh"])
    gaussian_types = sorted({row["gaussian_type"] for row in launcher})
    return {
        "root": str(root.resolve()),
        "revision": revision,
        "files": {
            relative: {
                "sha256": hashlib.sha256(source.encode()).hexdigest(),
                "bytes": len(source.encode()),
            }
            for relative, source in sources.items()
        },
        "cambridge_launcher": {
            "command_count": len(launcher),
            "gaussian_types": gaussian_types,
            "commands": launcher,
            "all_commands_use_3dgs": gaussian_types == ["3dgs"],
        },
        "two_dgs_constructor": {
            "scene_call_argument_count": scene_arity,
            "model_method_argument_count": initializers["GaussianModel_2dgs"],
            "callable": initializers["GaussianModel_2dgs"] == scene_arity,
        },
        "three_dgs_constructor": {
            "scene_call_argument_count": scene_arity,
            "model_method_argument_count": initializers["GaussianModel"],
            "callable": initializers["GaussianModel"] == scene_arity,
        },
        "trainer": {
            "effective_opacity_cull": _effective_cull(train),
            "declared_opacity_cull": _declared_opacity_cull(arguments),
            "render_rgb_only_values": _rgb_only_values(train),
            "checkpoint_order": _training_checkpoint_order(train),
            "contains_feature_loss": any(
                isinstance(node, ast.Name) and node.id == "Ll1_feature"
                for node in ast.walk(train)
            ),
        },
        "renderer": {
            "imports": backend_imports,
            "from_imports": backend_from_imports,
            "uses_gsplat": "gsplat" in backend_imports or "gsplat" in backend_from_imports,
            "uses_diff_surfel": (
                "diff_surfel_rasterization" in backend_imports
                or "diff_surfel_rasterization" in backend_from_imports
            ),
        },
    }


def build_main_protocol_audit(
    ulfloc_root: Path,
    ulfloc_revision: str,
    stdloc_root: Path,
    stdloc_revision: str,
) -> dict[str, Any]:
    """Compare immutable released-main protocol facts for both references."""
    ulf_sha, ulf_sources = _revision_sources(ulfloc_root, ulfloc_revision)
    std_sha, std_sources = _revision_sources(stdloc_root, stdloc_revision)
    ulfloc = audit_sources(root=ulfloc_root, revision=ulf_sha, sources=ulf_sources)
    stdloc = audit_sources(root=stdloc_root, revision=std_sha, sources=std_sources)
    return {
        "protocol": "reference_main_cambridge_contract_audit_v2",
        "references": {"ulfloc": ulfloc, "stdloc": stdloc},
        "conclusions": {
            "both_released_cambridge_launchers_use_3dgs": (
                ulfloc["cambridge_launcher"]["all_commands_use_3dgs"]
                and stdloc["cambridge_launcher"]["all_commands_use_3dgs"]
            ),
            "both_use_gsplat_renderer": (
                ulfloc["renderer"]["uses_gsplat"] and stdloc["renderer"]["uses_gsplat"]
            ),
            "ulfloc_2d_entrypoint_callable": ulfloc["two_dgs_constructor"]["callable"],
            "stdloc_2d_entrypoint_callable": stdloc["two_dgs_constructor"]["callable"],
            "effective_cull_matches": (
                ulfloc["trainer"]["effective_opacity_cull"]
                == stdloc["trainer"]["effective_opacity_cull"]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ulfloc-root", type=Path, required=True)
    parser.add_argument("--ulfloc-revision", default="origin/main")
    parser.add_argument("--stdloc-root", type=Path, required=True)
    parser.add_argument("--stdloc-revision", default="upstream/main")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite reference audit: {output}")
    report = build_main_protocol_audit(
        args.ulfloc_root,
        args.ulfloc_revision,
        args.stdloc_root,
        args.stdloc_revision,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
