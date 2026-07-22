#!/usr/bin/env python3
"""Statically audit the executable 2DGS contract of ULF-Loc and STDLoc.

The two reference repositories contain configuration defaults that do not
always describe the value actually supplied to their training loop.  This
tool deliberately reads their source without importing either project, so it
also exposes entry-point signature mismatches before an expensive experiment
is labelled as a reference baseline.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


_REQUIRED_FILES = (
    "train.py",
    "scene/__init__.py",
    "scene/gaussian_model.py",
    "arguments/__init__.py",
)


def _read_tree(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise RuntimeError(f"Could not parse {path}: {error}") from error


def _find_function(tree: ast.AST, class_name: str, function_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == function_name:
                    return member
    raise RuntimeError(f"Could not find {class_name}.{function_name}")


def _literal(node: ast.AST, *, description: str) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError) as error:
        raise RuntimeError(f"{description} is not a literal") from error


def _method_parameter_count(function: ast.FunctionDef) -> int:
    positional = list(function.args.posonlyargs) + list(function.args.args)
    if positional and positional[0].arg == "self":
        positional = positional[1:]
    if function.args.vararg is not None or function.args.kwarg is not None:
        raise RuntimeError(f"{function.name} has variadic parameters; cannot audit arity")
    return len(positional) + len(function.args.kwonlyargs)


def _scene_initializer_call_arity(tree: ast.AST) -> int:
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
        raise RuntimeError(f"Expected exactly one Scene Gaussian initializer call, found {len(calls)}")
    if calls[0].keywords:
        raise RuntimeError("Scene Gaussian initializer uses keywords; positional contract is ambiguous")
    return len(calls[0].args)


def _effective_densify_cull(tree: ast.AST) -> float:
    values = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "densify_and_prune"
        ):
            continue
        if len(node.args) < 2:
            raise RuntimeError("densify_and_prune call omits the opacity threshold")
        values.append(float(_literal(node.args[1], description="densify opacity threshold")))
    if not values:
        raise RuntimeError("Could not find a densify_and_prune call")
    if len(set(values)) != 1:
        raise RuntimeError(f"Reference uses multiple effective opacity thresholds: {values}")
    return values[0]


def _declared_opacity_cull(tree: ast.AST) -> float:
    function = _find_function(tree, "OptimizationParams", "__init__")
    values = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "opacity_cull"
            for target in node.targets
        ):
            values.append(float(_literal(node.value, description="declared opacity_cull")))
    if len(values) != 1:
        raise RuntimeError(f"Expected one declared opacity_cull default, found {values}")
    return values[0]


def _file_audit(root: Path) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for relative in _REQUIRED_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Reference file is missing: {path}")
        content = path.read_bytes()
        report[relative] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
    return report


def audit_reference(root: Path) -> dict[str, Any]:
    """Extract the reference's effective 2DGS initialization and prune contract."""
    root = root.resolve()
    files = _file_audit(root)
    train = _read_tree(root / "train.py")
    scene = _read_tree(root / "scene/__init__.py")
    model = _read_tree(root / "scene/gaussian_model.py")
    arguments = _read_tree(root / "arguments/__init__.py")

    initializer_parameters = _method_parameter_count(
        _find_function(model, "GaussianModel_2dgs", "create_from_pcd")
    )
    initializer_call_arguments = _scene_initializer_call_arity(scene)
    declared_cull = _declared_opacity_cull(arguments)
    effective_cull = _effective_densify_cull(train)
    return {
        "root": str(root),
        "files": files,
        "two_dgs_initializer": {
            "method_parameter_count_excluding_self": initializer_parameters,
            "scene_call_argument_count": initializer_call_arguments,
            "entrypoint_callable": initializer_parameters == initializer_call_arguments,
        },
        "opacity_cull": {
            "declared_argument_default": declared_cull,
            "effective_train_loop_value": effective_cull,
            "declared_matches_effective": declared_cull == effective_cull,
        },
    }


def build_reference_audit(ulfloc_root: Path, stdloc_root: Path) -> dict[str, Any]:
    """Compare source-level executable contracts from the two references."""
    ulfloc = audit_reference(ulfloc_root)
    stdloc = audit_reference(stdloc_root)
    return {
        "protocol": "reference_2dgs_training_contract_audit_v1",
        "references": {"ulfloc": ulfloc, "stdloc": stdloc},
        "comparisons": {
            "effective_opacity_cull_matches": (
                ulfloc["opacity_cull"]["effective_train_loop_value"]
                == stdloc["opacity_cull"]["effective_train_loop_value"]
            ),
            "ulfloc_2d_entrypoint_callable": ulfloc["two_dgs_initializer"]["entrypoint_callable"],
            "stdloc_2d_entrypoint_callable": stdloc["two_dgs_initializer"]["entrypoint_callable"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ulfloc-root", type=Path, required=True)
    parser.add_argument("--stdloc-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite reference audit: {output}")
    report = build_reference_audit(args.ulfloc_root, args.stdloc_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
