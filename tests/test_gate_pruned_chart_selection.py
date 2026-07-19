import pytest

from scripts.materialize_gate_pruned_chart_selection import build_gate_pruned_selection


def test_gate_pruned_selection_removes_rejected_chart_and_opt_in_audits_subset():
    selection = {
        "image_names": ["seq1__frame00001.png", "seq1__frame00002.png", "seq1__frame00003.png"],
        "image_idx": [1, 2, 3],
    }
    gate = {
        "records": [
            {"image_name": "seq1__frame00001.png", "rejected": False},
            {"image_name": "seq1__frame00002.png", "rejected": True},
            {"image_name": "seq1__frame00003.png", "rejected": False},
        ]
    }

    result = build_gate_pruned_selection(selection, gate, minimum_active=2)

    assert result["image_names"] == ["seq1__frame00001.png", "seq1__frame00003.png"]
    assert result["image_idx"] == [1, 3]
    assert result["audit_active_from_selection"]
    assert result["gate_pruned_selection"]["excluded_chart_names"] == ["seq1__frame00002.png"]


def test_gate_pruned_selection_refuses_to_drop_below_minimum():
    selection = {"image_names": ["seq1__frame00001.png"]}
    gate = {"records": [{"image_name": "seq1__frame00001.png", "rejected": True}]}

    with pytest.raises(RuntimeError, match="gate-valid"):
        build_gate_pruned_selection(selection, gate, minimum_active=1)
