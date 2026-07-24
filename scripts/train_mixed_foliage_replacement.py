#!/usr/bin/env python
"""Canonical entry point for native 2DGS/3DGS replace-and-retire training.

The implementation remains import-compatible at its original development
location while experiments and manifests use this stable mainline name.
"""

from train_hybrid_foliage_3dgs import main


if __name__ == "__main__":
    main()
