from outdoor.foliage_view_graph import (
    greedy_diverse_views,
    instance_balanced_diverse_views,
)


def _record(image_id, sequence, x, canopy=0.5):
    return {
        "image_id": image_id,
        "image_name": f"{sequence}__frame{image_id:05d}.png",
        "sequence_id": sequence,
        "camera_center": [float(x), 0.0, 0.0],
        "canopy_fraction": float(canopy),
    }


def test_instance_balanced_views_reserve_small_tree_evidence():
    records = [
        _record(1, "seq1", 0.0, 0.9),
        _record(2, "seq2", 2.0, 0.9),
        _record(3, "seq3", 4.0, 0.8),
        _record(4, "seq1", 6.0, 0.2),
        _record(5, "seq2", 8.0, 0.2),
        _record(6, "seq3", 10.0, 0.2),
    ]
    # Instance 0 is the dominant foreground crown. Instance 1 has weaker,
    # disjoint cameras that a global canopy ranking would skip.
    support = {
        0: {1: 20, 2: 18, 3: 16},
        1: {4: 2, 5: 2, 6: 1},
    }

    selected = instance_balanced_diverse_views(
        records,
        support_by_instance=support,
        limit=4,
        minimum_center_distance=0.1,
    )
    selected_ids = {int(row["image_id"]) for row in selected}

    assert len(selected) == 4
    assert selected_ids & {1, 2, 3}
    assert selected_ids & {4, 5, 6}


def test_instance_balanced_views_are_unique_and_bounded():
    records = [_record(index, f"seq{index % 2}", index) for index in range(8)]
    selected = instance_balanced_diverse_views(
        records,
        support_by_instance={
            0: {0: 3, 1: 2, 2: 1},
            1: {2: 4, 3: 3, 4: 2},
            2: {5: 2, 6: 2, 7: 1},
        },
        limit=5,
    )

    ids = [int(row["image_id"]) for row in selected]
    assert len(ids) == 5
    assert len(set(ids)) == len(ids)


def test_greedy_views_do_not_collapse_to_consecutive_high_canopy_frames():
    records = [
        _record(index, "seq1", float(index), canopy=0.95)
        for index in range(5)
    ]
    records.extend(
        [
            _record(50, "seq1", 50.0, canopy=0.75),
            _record(100, "seq1", 100.0, canopy=0.70),
        ]
    )

    selected = greedy_diverse_views(
        records,
        limit=3,
        minimum_center_distance=20.0,
    )
    frames = {int(row["image_id"]) for row in selected}

    assert frames & set(range(5))
    assert 100 in frames
    assert len(frames & {50, 100}) >= 1


def test_later_instance_reservations_respect_global_temporal_context():
    records = [
        _record(index, "seq1", float(index), canopy=0.9)
        for index in (0, 1, 2, 50, 99, 100)
    ]
    selected = instance_balanced_diverse_views(
        records,
        support_by_instance={
            0: {0: 8, 1: 7, 2: 6},
            1: {1: 8, 2: 7, 50: 6, 99: 5, 100: 4},
        },
        limit=4,
        minimum_center_distance=20.0,
        maximum_reserved_per_instance=2,
    )
    frames = {int(row["image_id"]) for row in selected}

    assert len(frames) == 4
    assert max(frames) - min(frames) >= 50
