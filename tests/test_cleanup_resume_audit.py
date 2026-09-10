import copy
import pytest
from scripts.train_unified_outdoor_teacher import _resume_training_contract_differences


def contracts():
    saved={"static_detail_global_cleanup":{"contract":"negative_only","weight":.25,"every":2,
        "camera_schedule":{"contract":"fixed_canonical_negative","canonical_sequence":"seq2",
            "camera_count":352,"schedule_sha256":"fixed","positive_exact_support_camera_count":233}}}
    current=copy.deepcopy(saved)
    current["static_detail_global_cleanup"]["camera_schedule"]["positive_exact_support_camera_count"]=240
    return saved,current


def test_positive_support_audit_cannot_block_unchanged_negative_schedule_resume():
    saved,current=contracts(); original=copy.deepcopy((saved,current))
    assert _resume_training_contract_differences(saved,current)==set()
    assert (saved,current)==original


@pytest.mark.parametrize("key,value",[("camera_count",351),("schedule_sha256","changed"),("canonical_sequence","seq1")])
def test_actual_negative_schedule_changes_remain_incompatible(key,value):
    saved,current=contracts()
    current["static_detail_global_cleanup"]["camera_schedule"][key]=value
    assert _resume_training_contract_differences(saved,current)=={"static_detail_global_cleanup"}


def test_cleanup_weight_change_still_fails():
    saved,current=contracts();current["static_detail_global_cleanup"]["weight"]=.5
    assert _resume_training_contract_differences(saved,current)=={"static_detail_global_cleanup"}
