"""Reproducible code/import/CUDA provenance for long reconstruction runs."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess
from typing import Iterable


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _command(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        list(args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def collect_runtime_provenance(
    repo: Path,
    *,
    python_modules: Iterable[str] = (),
    extension_roots: Iterable[Path] = (),
) -> dict:
    repo = Path(repo).resolve()
    status = _command(repo, "git", "status", "--porcelain=v1")
    diff = _command(repo, "git", "diff", "--binary")
    imports = {}
    for name in python_modules:
        spec = importlib.util.find_spec(name)
        path = (
            None
            if spec is None or spec.origin in {None, "built-in"}
            else Path(spec.origin).resolve()
        )
        imports[name] = {
            "path": None if path is None else str(path),
            "sha256": (
                None
                if path is None or not path.is_file()
                else file_sha256(path)
            ),
        }
    extensions = []
    for root in extension_roots:
        root = Path(root).resolve()
        for path in sorted(root.rglob("*.so")):
            extensions.append(
                {
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
            )
    return {
        "git_commit": _command(repo, "git", "rev-parse", "HEAD").strip(),
        "git_dirty": bool(status.strip()),
        "git_status": status.splitlines(),
        "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "python_imports": imports,
        "extension_modules": extensions,
    }
