import pytest
import torch
from scripts.canopy_optical_capacity import weighted_opacity_summary
from scripts.canopy_optical_capacity import weighted_size_summary


def test_capacity_is_weighted_by_actual_contribution_not_row_count():
    out = weighted_opacity_summary(torch.tensor([.99, .01]), torch.tensor([9., 1.]))
    assert out['contribution_from_peak_above_09'] == pytest.approx(.9)
    assert out['contribution_from_peak_below_01'] == pytest.approx(.1)
    assert out['weighted_peak_opacity'] == pytest.approx(.892)


def test_size_summary_tracks_render_contribution_not_inactive_row_count():
    out = weighted_size_summary(torch.tensor([2., .5]), torch.tensor([9., 1.]))
    assert out['weighted_size_ratio'] == pytest.approx(1.85)
    assert out['contribution_from_size_above_18'] == pytest.approx(.9)
    assert out['contribution_from_size_below_08'] == pytest.approx(.1)
    assert weighted_size_summary(torch.ones(2), torch.zeros(2))['weighted_size_ratio'] is None
