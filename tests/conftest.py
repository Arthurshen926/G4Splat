"""Bind native tests before collection can import an editable foreign build."""
from pathlib import Path
import sys

LOCAL_RASTER = (Path(__file__).resolve().parents[1]
                / "2d-gaussian-splatting/submodules/diff-surfel-rasterization")
sys.path.insert(0, str(LOCAL_RASTER))
