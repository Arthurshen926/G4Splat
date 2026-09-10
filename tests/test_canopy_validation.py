import ast
from pathlib import Path

import pytest
import torch

from scripts.evaluate_canopy_validation import (
    ADDITIONAL, DIAGNOSTIC, FIXED, masked_metrics, tensor_digest,
)


def test_prespecified_canopy_cohorts_are_disjoint():
    assert len(FIXED) == 16
    assert len(ADDITIONAL) == 32
    assert len(DIAGNOSTIC) == 3
    assert len(set(FIXED + ADDITIONAL + DIAGNOSTIC)) == 51


def test_masked_validation_metrics_do_not_count_excluded_pixels():
    target = torch.zeros(3, 2, 2)
    prediction = torch.ones_like(target)
    prediction[:,0,0] = .1
    mask = torch.tensor([[True,False],[False,False]])
    result = masked_metrics(prediction,target,mask)
    assert result["pixels"] == 1
    assert result["squared_error_sum"] == pytest.approx(.03)
    assert result["psnr"] == pytest.approx(20.)
    assert masked_metrics(prediction,target,~torch.ones_like(mask))["psnr"] is None


def test_rigid_fingerprint_detects_geometry_and_shape_changes():
    values = torch.arange(6).reshape(2,3).float()
    assert tensor_digest(values) == tensor_digest(values.clone())
    assert tensor_digest(values) != tensor_digest(values.reshape(3,2))
    changed = values.clone(); changed[0,0] = 1e-7
    assert tensor_digest(values) != tensor_digest(changed)


def test_validation_never_routes_render_by_reporting_masks():
    source = Path(__file__).resolve().parents[1]/"scripts/evaluate_canopy_validation.py"
    tree = ast.parse(source.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node,ast.Call)
             and isinstance(node.func,ast.Attribute) and node.func.attr == "render"]
    assert len(calls) == 1
    keywords = {kw.arg:kw.value for kw in calls[0].keywords}
    assert isinstance(keywords["task"],ast.Constant) and keywords["task"].value is None
    assert isinstance(keywords["conditioned"],ast.Constant) and keywords["conditioned"].value is False
