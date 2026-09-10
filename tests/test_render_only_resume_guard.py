import pytest
from scripts.train_unified_outdoor_teacher import _assert_training_resume_payload


@pytest.mark.parametrize('payload',[{'diagnostic_only':True},{'checkpoint_kind':'render_state_only_nonresumable'}])
def test_explicit_nonresumable_artifacts_are_rejected_before_training_restore(payload):
    with pytest.raises(RuntimeError,match='not a resumable training checkpoint'):
        _assert_training_resume_payload(payload)


def test_resume_guard_does_not_change_legacy_unmarked_checkpoint_contract():
    _assert_training_resume_payload(None)
    _assert_training_resume_payload({'iteration':3000})
