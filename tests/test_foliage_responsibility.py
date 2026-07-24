import torch

from outdoor.foliage_responsibility import rigid_safe_replacement_mask


def test_rigid_protection_cannot_be_diluted_across_views():
    aggregate = torch.tensor([0.04, 0.04, 0.40])
    maximum_view = torch.tensor([0.10, 0.80, 0.10])

    safe = rigid_safe_replacement_mask(aggregate, maximum_view, 0.25)

    assert safe.tolist() == [True, False, False]
