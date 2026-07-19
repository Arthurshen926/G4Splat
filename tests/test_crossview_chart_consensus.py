from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.build_crossview_chart_consensus import (
    _active_indices,
    apply_rejected_supported_weight,
    consistency_weights_from_relative_error,
    consensus_candidate_depth,
    gate_consensus_by_relative_error,
    select_overlap_neighbours,
)


def test_consensus_requires_independent_support_and_preserves_fallback() -> None:
    source = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    valid = np.array([[True, True], [True, False]])
    neighbours = np.array(
        [
            [[2.1, np.nan], [4.2, 9.0]],
            [[1.9, 2.9], [3.8, np.nan]],
            [[2.0, 3.1], [np.nan, np.nan]],
        ],
        dtype=np.float32,
    )
    consensus, support, error = consensus_candidate_depth(source, valid, neighbours, 2)

    assert np.array_equal(support, np.array([[3, 2], [2, 1]], dtype=np.uint8))
    assert np.allclose(consensus[:2, :], np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32))
    assert np.isfinite(error[0, 0])
    assert np.isfinite(error[0, 1])
    assert np.isfinite(error[1, 0])
    assert np.isnan(error[1, 1])


def test_consensus_falls_back_when_only_one_neighbour_observes_a_pixel() -> None:
    source = np.array([[2.0, 3.0]], dtype=np.float32)
    valid = np.array([[True, True]])
    neighbours = np.array([[[1.0, np.nan]], [[np.nan, np.nan]]], dtype=np.float32)
    consensus, support, error = consensus_candidate_depth(source, valid, neighbours, 2)

    assert np.array_equal(consensus, source)
    assert np.array_equal(support, np.array([[1, 0]], dtype=np.uint8))
    assert np.isnan(error).all()


def test_consistency_weights_leave_unsupported_pixels_exactly_neutral() -> None:
    errors = np.array([[0.0, 0.25], [1.0, np.nan]], dtype=np.float32)
    support = np.array([[2, 2], [1, 3]], dtype=np.uint8)
    valid = np.array([[True, True], [True, False]])
    weights = consistency_weights_from_relative_error(
        errors,
        support,
        valid,
        min_consensus_views=2,
        floor=0.5,
        sigma=0.25,
    )

    assert weights[0, 0] == 1.0
    assert 0.5 < weights[0, 1] < 1.0
    assert weights[1, 0] == 1.0
    assert weights[1, 1] == 1.0


def test_relative_error_gate_preserves_source_at_occlusion_like_outliers() -> None:
    source = np.array([[2.0, 3.0, 4.0]], dtype=np.float32)
    consensus = np.array([[2.1, 4.0, 2.0]], dtype=np.float32)
    errors = np.array([[0.05, 0.25, np.nan]], dtype=np.float32)

    gated, accepted = gate_consensus_by_relative_error(
        source, consensus, errors, max_relative_error=0.15
    )

    assert np.array_equal(accepted, np.array([[True, False, False]]))
    assert np.allclose(gated, np.array([[2.1, 3.0, 4.0]], dtype=np.float32))


def test_rejected_supported_pixels_can_be_attenuated_without_touching_unobserved_pixels() -> None:
    source = np.array([[2.0, 3.0, 4.0]], dtype=np.float32)
    consensus = np.array([[2.1, 4.0, 4.0]], dtype=np.float32)
    errors = np.array([[0.05, 0.25, np.nan]], dtype=np.float32)
    support = np.array([[2, 2, 0]], dtype=np.uint8)
    valid = np.array([[True, True, True]])
    _, accepted = gate_consensus_by_relative_error(
        source, consensus, errors, max_relative_error=0.15
    )
    weights = consistency_weights_from_relative_error(
        errors,
        support,
        valid,
        min_consensus_views=2,
        floor=0.0,
        sigma=0.06,
    )
    weights, rejected_supported = apply_rejected_supported_weight(
        weights,
        accepted,
        support,
        valid,
        min_consensus_views=2,
        rejected_supported_weight=0.25,
    )

    assert accepted.tolist() == [[True, False, False]]
    assert weights[0, 0] > 0.25
    assert weights[0, 1] == pytest.approx(0.25)
    assert weights[0, 2] == pytest.approx(1.0)
    assert rejected_supported.tolist() == [[False, True, False]]


def test_rejected_supported_default_preserves_legacy_neutral_weights() -> None:
    weights = np.array([[0.8, 0.3, 1.0]], dtype=np.float32)
    accepted = np.array([[True, False, False]])
    support = np.array([[2, 2, 0]], dtype=np.uint8)
    valid = np.array([[True, True, True]])

    adjusted, rejected_supported = apply_rejected_supported_weight(
        weights,
        accepted,
        support,
        valid,
        min_consensus_views=2,
        rejected_supported_weight=1.0,
    )

    assert np.allclose(adjusted, np.array([[0.8, 1.0, 1.0]], dtype=np.float32))
    assert rejected_supported.tolist() == [[False, True, False]]


def test_overlap_selection_uses_measured_coverage_and_never_backfills_below_gate() -> None:
    peers = np.array([7, 3, 9, 4], dtype=np.int64)
    overlap = np.array([0.10, 0.55, 0.55, 0.03], dtype=np.float32)

    selected = select_overlap_neighbours(peers, overlap, count=3, min_overlap=0.05)

    # Equal coverage is deterministically resolved by chart id, and the 3%
    # candidate remains excluded rather than being used just to fill a slot.
    assert np.array_equal(selected, np.array([3, 9, 7], dtype=np.int64))


def test_selection_excludes_a_chart_from_consensus_neighbours(tmp_path: Path) -> None:
    names = ["seq1__frame00001.png", "seq1__frame00002.png", "seq1__frame00003.png"]
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "records": [
                    {"image_name": name, "rejected": False}
                    for name in names
                ]
            }
        )
    )
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"image_names": names[:2]}))

    active = _active_indices(gate, names, selection)

    assert np.array_equal(active, np.array([0, 1], dtype=np.int64))


def test_selection_cannot_reenable_a_hard_gate_rejection(tmp_path: Path) -> None:
    names = ["seq1__frame00001.png", "seq1__frame00002.png", "seq1__frame00003.png"]
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "records": [
                    {"image_name": names[0], "rejected": False},
                    {"image_name": names[1], "rejected": False},
                    {"image_name": names[2], "rejected": True},
                ]
            }
        )
    )
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"image_names": names}))

    with pytest.raises(RuntimeError, match="hard-gate rejected"):
        _active_indices(gate, names, selection)
