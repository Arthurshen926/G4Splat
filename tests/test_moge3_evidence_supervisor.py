from pathlib import Path

import pytest

from scripts.supervise_moge3_evidence import _shard_ranges, _worker_command


def test_moge3_shards_are_disjoint_and_complete() -> None:
    ranges = _shard_ranges(1487, 3)
    assert ranges == ((0, 496), (496, 496), (992, 495))
    covered = [
        value
        for start, limit in ranges
        for value in range(start, start + limit)
    ]
    assert covered == list(range(1487))


def test_moge3_shards_reject_empty_or_oversubscribed_partition() -> None:
    with pytest.raises(ValueError):
        _shard_ranges(0, 1)
    with pytest.raises(ValueError):
        _shard_ranges(2, 3)


def test_moge3_worker_command_freezes_precision_and_exact_range() -> None:
    command = _worker_command(
        python=Path("/env/python"),
        dataset=Path("/data"),
        scene_contract=Path("/contract.json"),
        output=Path("/evidence"),
        model="model",
        model_revision="immutable-revision",
        refine_steps=3,
        resolution_level=9,
        start=496,
        limit=496,
        use_fp16=True,
    )
    assert command[0] == "/env/python"
    assert command[command.index("--model-revision") + 1] == "immutable-revision"
    assert command[command.index("--start") + 1] == "496"
    assert command[command.index("--limit") + 1] == "496"
    assert "--use-fp16" in command
    assert "--replace-views" not in command

