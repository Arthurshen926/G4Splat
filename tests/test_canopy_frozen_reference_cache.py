import torch
from scripts.canopy_frozen_reference_cache import FrozenReferenceCache


def test_cache_preserves_float_bits_and_does_not_alias_renderer_or_consumer():
    source = torch.randn(3, 4, 5, dtype=torch.float64, requires_grad=True)
    expected = source.detach().clone()
    cache = FrozenReferenceCache(2)
    first = cache.get((1, 4, 5), lambda: source, 'cpu')
    assert not first.requires_grad and torch.equal(first, expected)
    first.zero_()
    with torch.no_grad(): source.fill_(2.)
    second = cache.get((1, 4, 5), lambda: (_ for _ in ()).throw(AssertionError()), 'cpu')
    assert torch.equal(second, expected)
    assert cache.stats()['hits'] == 1 and cache.stats()['misses'] == 1


def test_lru_capacity_and_disabled_mode():
    cache = FrozenReferenceCache(1)
    render = lambda: torch.ones(3, 2, 2)
    for key in ('a', 'b', 'a'): cache.get(key, render, 'cpu')
    assert cache.stats() == dict(capacity=1, entries=1, hits=0, misses=3, bytes=48)
    disabled = FrozenReferenceCache(0)
    for _ in range(2): disabled.get('a', render, 'cpu')
    assert disabled.stats()['entries'] == 0 and disabled.stats()['misses'] == 2
