import pytest
import torch

from scripts.train_unified_outdoor_teacher import _final_surface_prune_mask


@pytest.mark.parametrize('policy', ['appearance_only', 'frozen'])
def test_final_cleanup_cannot_delete_frozen_handoff(policy):
    proposed = torch.tensor([True, False, True])
    original = proposed.clone()
    assert not _final_surface_prune_mask(proposed, policy).any()
    assert torch.equal(proposed, original)


def test_final_cleanup_joint_and_residual_authority():
    proposed = torch.tensor([True, False, True, True])
    assert torch.equal(_final_surface_prune_mask(proposed, 'joint'), proposed)
    assert torch.equal(
        _final_surface_prune_mask(proposed, 'atlas_residual', residual_start=3),
        torch.tensor([False, False, False, True]),
    )
    for boundary in (None, -1, 5):
        with pytest.raises(RuntimeError):
            _final_surface_prune_mask(proposed, 'atlas_residual', residual_start=boundary)
    with pytest.raises(ValueError):
        _final_surface_prune_mask(proposed, 'typo')
