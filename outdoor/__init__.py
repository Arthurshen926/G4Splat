"""Outdoor Cambridge reconstruction primitives.

The legacy G4Splat entrypoints are deliberately kept intact for controlled
ablations.  This package contains the explicit, provenance-carrying building
blocks used by the outdoor structural mainline.
"""

from .scene_contract import build_scene_contract

__all__ = ["build_scene_contract"]
