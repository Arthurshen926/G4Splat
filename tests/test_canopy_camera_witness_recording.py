import pytest
import torch
from outdoor.canopy_support_expansion import record_distinct_camera_witnesses_


def test_independent_witnesses_not_repeated_visits_or_full_table_overwrites():
    table = torch.tensor([[-1, -1], [7, -1], [4, 5]], dtype=torch.int32)
    witness = torch.tensor([True, True, True])
    assert record_distinct_camera_witnesses_(table, witness, 7) == 1
    assert table.tolist() == [[7, -1], [7, -1], [4, 5]]
    assert record_distinct_camera_witnesses_(table, witness, 7) == 0
    assert record_distinct_camera_witnesses_(table, torch.tensor([True, False, True]), 8) == 1
    assert table.tolist() == [[7, 8], [7, -1], [4, 5]]


def test_no_authority_without_a_real_witness_or_slot():
    table = torch.full((3, 2), -1, dtype=torch.int32)
    assert record_distinct_camera_witnesses_(table, torch.zeros(3, dtype=torch.bool), 7) == 0
    assert (table == -1).all()
    assert record_distinct_camera_witnesses_(table[:, :0], torch.ones(3, dtype=torch.bool), 7) == 0


@pytest.mark.parametrize('camera', [-1, 32768])
def test_camera_identity_cannot_overflow_or_be_a_sentinel(camera):
    with pytest.raises(ValueError):
        record_distinct_camera_witnesses_(torch.full((1, 2), -1, dtype=torch.int16), torch.tensor([True]), camera)
