import pytest
from scripts.canopy_candidate_gradient_conflict import training_gradient_conflict


def test_disabled_candidate_gradient_audit_cannot_silently_enable_candidates():
    with pytest.raises(ValueError, match='disabled candidates'):
        training_gradient_conflict(None, None, None, None, None, None,
            {'args': {'candidate_gain': 0}}, {}, None)
