#!/usr/bin/env python3
"""Fuse plane, aligned-Chart and mono inverse depth with provenance sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from outdoor.inverse_depth import fuse_inverse_depth_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mast3r-scene", type=Path, required=True)
    parser.add_argument("--plane-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fuse_inverse_depth_directory(args.mast3r_scene, args.plane_root, args.output), indent=2))


if __name__ == "__main__":
    main()
