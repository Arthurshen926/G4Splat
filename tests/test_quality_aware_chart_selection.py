import numpy as np
import pytest
from typing import Optional

from scripts.select_quality_aware_charts import JointSelector


def _selector(
    support_matrix: np.ndarray,
    *,
    sequence_matrix: Optional[np.ndarray] = None,
    sequence_names: Optional[list[str]] = None,
    min_sequence_support: int = 0,
) -> JointSelector:
    count = support_matrix.shape[1]
    return JointSelector(
        reliabilities=np.linspace(0.1, 0.9, count),
        cells=[np.array([index], dtype=np.int64) for index in range(count)],
        cell_weights=np.ones(count, dtype=np.float64),
        target_distances=np.zeros((count, count), dtype=np.float64),
        centers=np.stack(
            [np.array([float(index), 0.0, 0.0]) for index in range(count)]
        ),
        support_matrix=support_matrix,
        pair_scores=np.zeros((count, count), dtype=np.float64),
        sequence_matrix=sequence_matrix,
        sequence_names=sequence_names,
        min_support=2,
        min_baseline=0.5,
        min_sequence_support=min_sequence_support,
    )


def test_selector_seeds_baseline_pairs_before_quality_greedy_pass():
    selector = _selector(
        np.array(
            [
                [True, True, False, False],
                [False, False, True, True],
            ]
        )
    )

    selected, diagnostics = selector.select(4)

    assert selected == [0, 1, 2, 3]
    assert all(cluster["support_count"] == 2 for cluster in diagnostics["constraints"]["clusters"])
    assert all(cluster["baseline_satisfied"] for cluster in diagnostics["constraints"]["clusters"])


def test_selector_reports_an_infeasible_hard_gated_pose_cell_without_relaxing_it():
    selector = _selector(np.array([[True, False, False]]))

    with pytest.raises(RuntimeError, match="no baseline-separated support pair"):
        selector.select(2)


def test_selector_preserves_a_baseline_pair_for_each_temporal_sequence():
    selector = _selector(
        np.array([[True, True, True, True, False, False]]),
        sequence_matrix=np.array(
            [
                [True, True, False, False, False, False],
                [False, False, True, True, False, False],
            ]
        ),
        sequence_names=["seq_a", "seq_b"],
        min_sequence_support=2,
    )

    selected, diagnostics = selector.select(4)

    assert selected == [0, 1, 2, 3]
    sequences = diagnostics["constraints"]["sequences"]
    assert [record["sequence"] for record in sequences] == ["seq_a", "seq_b"]
    assert all(record["support_count"] == 2 for record in sequences)
    assert all(record["baseline_satisfied"] and record["passed"] for record in sequences)


def test_selector_fails_when_an_active_sequence_lacks_a_baseline_pair():
    selector = _selector(
        np.array([[True, True, True]]),
        sequence_matrix=np.array([[True, False, False]]),
        sequence_names=["seq_sparse"],
        min_sequence_support=2,
    )

    with pytest.raises(RuntimeError, match="same-sequence trajectory seq_sparse"):
        selector.select(2)
