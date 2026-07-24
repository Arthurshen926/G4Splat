from types import SimpleNamespace

import pytest
import torch

from outdoor.render_contract import (
    assert_render_contract,
    capture_render_contract,
    compare_render_contract,
)


def _render(view, model, pipe, background):
    value = torch.full((1, 2, 2), float(model.value + view.offset))
    return {
        "render": value.expand(3, -1, -1),
        "rend_alpha": value,
        "rend_depth": value + 1,
        "rend_depth_median": value + 2,
        "radii": torch.tensor([1, 2], dtype=torch.int32),
    }


def test_zero_step_contract_covers_rgb_alpha_depth_and_radii():
    views = [
        SimpleNamespace(image_name="seq1/a.png", offset=0),
        SimpleNamespace(image_name="seq2/b.png", offset=1),
    ]
    model = SimpleNamespace(value=2)
    before = capture_render_contract(_render, views, model, None, None, [0, 1])
    after = capture_render_contract(_render, views, model, None, None, [0, 1])

    report = compare_render_contract(before, after)

    assert report["passed"]
    assert set(report["views"][0]["fields"]) == {
        "render",
        "rend_alpha",
        "rend_depth",
        "rend_depth_median",
        "radii",
    }


def test_zero_step_contract_reports_a_changed_field():
    view = SimpleNamespace(image_name="seq1/a.png", offset=0)
    before = capture_render_contract(
        _render, [view], SimpleNamespace(value=2), None, None, [0]
    )
    after = capture_render_contract(
        _render, [view], SimpleNamespace(value=3), None, None, [0]
    )
    report = compare_render_contract(before, after)

    assert not report["passed"]
    with pytest.raises(RuntimeError, match="Zero-step renderer identity failed"):
        assert_render_contract(report)
