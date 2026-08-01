#!/usr/bin/env python3
"""Build an immutable calibrated camera/data contract for one Cambridge split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from outdoor.scene_contract import build_scene_contract, validate_disjoint_contracts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-pickle", type=Path)
    parser.add_argument("--split", default="database_train")
    parser.add_argument(
        "--skip-image-hashes",
        action="store_true",
        help="Use only for a fast local dry run; production contracts hash every image.",
    )
    parser.add_argument(
        "--query-contract",
        type=Path,
        help="Optional existing query contract that must be source-image disjoint.",
    )
    args = parser.parse_args()
    payload = build_scene_contract(
        args.dataset,
        args.output,
        mask_pickle=args.mask_pickle,
        split=args.split,
        image_hashes=not args.skip_image_hashes,
    )
    result = {
        "output": str(args.output),
        "image_count": payload["image_count"],
        "camera_policy": payload["camera_policy"],
    }
    if args.query_contract is not None:
        result["disjointness"] = validate_disjoint_contracts(args.output, args.query_contract)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
