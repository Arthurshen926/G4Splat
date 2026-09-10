import torch
import pytest
from scripts.audit_rigid_canopy_support import unsupported_canopy_surfels


def test_any_known_rigid_support_protects_a_surfel():
    result=unsupported_canopy_surfels(torch.tensor([0,1,2,0]),torch.tensor([3,50,50,1]))
    assert torch.equal(result,torch.tensor([True,False,False,False]))


def test_invalid_support_counts_are_rejected():
    with pytest.raises(ValueError):
        unsupported_canopy_surfels(torch.tensor([-1]),torch.tensor([5]))
