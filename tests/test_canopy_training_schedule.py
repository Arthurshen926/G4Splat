import torch
import pytest
from scripts.canopy_training_schedule import training_schedule


def test_reshuffle_covers_every_view_once_per_epoch():
    views=list(range(20));schedule=training_schedule(views,60,reshuffle=True)
    assert all(sorted(schedule[i:i+20])==views for i in (0,20,40))
    assert schedule[:20]!=schedule[20:40]


def test_schedule_preserves_legacy_and_rng_and_resume_suffix():
    torch.manual_seed(99);before=torch.random.get_rng_state().clone()
    views=list(range(20))
    legacy=training_schedule(views,65)
    assert legacy[:20]==legacy[20:40]
    active=training_schedule(views,65,reshuffle=True)
    assert active[:20]==legacy[:20]
    assert active[37:]==training_schedule(views,65,reshuffle=True)[37:]
    assert torch.equal(before,torch.random.get_rng_state())


def test_duplicate_training_views_rejected():
    with pytest.raises(ValueError):training_schedule([1,1],4)
