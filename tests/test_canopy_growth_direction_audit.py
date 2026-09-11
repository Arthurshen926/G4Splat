import pytest
import torch
from scripts.canopy_growth_direction_audit import summarize_growth


def test_growth_signs_exclude_zero_and_unexpanded():
    result = summarize_growth(torch.tensor([-1., 1., 0., -1., 1.]),
                              torch.tensor([1., 1., 1., 1., -1.]),
                              torch.tensor([1., 1., 1., -1., 1.]))
    assert result['expanded'] == 4
    assert result['expansion_conflict'] == 1
    assert result['joint_local_shrink'] == 1


def test_growth_rejects_invalid_evidence():
    with pytest.raises(ValueError):
        summarize_growth(torch.ones(2), torch.ones(3), torch.ones(2))
    with pytest.raises(ValueError):
        summarize_growth(torch.tensor([float('nan')]), torch.ones(1), torch.ones(1))
