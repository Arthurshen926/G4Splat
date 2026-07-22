#!/usr/bin/env python3
"""Write the explicit task-specific semantic policy used by outdoor training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from outdoor.task_semantics import build_task_semantic_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mask-pickle", type=Path, required=True)
    parser.add_argument("--tree-mask-pickle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tree-support-policy",
        choices=["error", "neutral", "legacy_zero"],
        default="neutral",
    )
    parser.add_argument("--neutral-tree-support", type=float, default=0.5)
    args = parser.parse_args()
    payload = build_task_semantic_manifest(
        args.dataset,
        args.mask_pickle,
        args.output,
        tree_mask_pickle=args.tree_mask_pickle,
        tree_support_policy=args.tree_support_policy,
        neutral_tree_support=args.neutral_tree_support,
    )
    print(json.dumps({"output": str(args.output), "policy": payload["schema_version"]}, indent=2))


if __name__ == "__main__":
    main()
