import pytest
import torch
from scripts.canopy_seed_ray_sampling import select_seed_rays


def test_uniform_and_zero_score_paths_keep_exact_historical_permutation():
    expected = torch.randperm(100, generator=torch.Generator().manual_seed(1701))[:10]
    assert torch.equal(select_seed_rays(100, 10, 1701), expected)
    assert torch.equal(select_seed_rays(100, 10, 1701, torch.zeros(100)), expected)


def test_priority_reallocates_without_replacement_or_budget_growth():
    scores = torch.zeros(1000); scores[:10] = 1.
    selected = select_seed_rays(1000, 20, 1701, scores)
    assert len(selected) == len(selected.unique()) == 20
    assert (selected < 10).sum() >= 8
    assert torch.equal(selected, select_seed_rays(1000, 20, 1701, scores))
    assert (selected >= 10).any()


def test_full_population_falls_back_and_global_rng_is_untouched():
    state = torch.random.get_rng_state()
    expected = torch.randperm(3, generator=torch.Generator().manual_seed(3))
    assert torch.equal(select_seed_rays(3, 20, 3, torch.tensor([0., 1., 2.])), expected)
    assert torch.equal(state, torch.random.get_rng_state())


@pytest.mark.parametrize('scores', [torch.tensor([-1., 0.]), torch.tensor([float('nan'), 0.]), torch.ones(3)])
def test_invalid_score_cannot_silently_select_geometry(scores):
    with pytest.raises(ValueError): select_seed_rays(2, 1, 0, scores)
