import pytest
import torch
from scripts.canopy_reference_bundle_cache import FrozenReferenceBundleCache


def test_bundle_reuses_one_render_and_does_not_expose_stored_tensors():
    cache=FrozenReferenceBundleCache(2);calls=[]
    rgb=torch.rand(3,2,4,requires_grad=True);alpha=torch.rand(1,2,4,requires_grad=True)
    def render():calls.append(1);return rgb,alpha
    first,second=cache.get('a',render,'cpu')
    assert not first.requires_grad and not second.requires_grad
    torch.testing.assert_close(first,rgb,rtol=0,atol=0)
    torch.testing.assert_close(second,alpha,rtol=0,atol=0)
    first.zero_();second.zero_()
    first,second=cache.get('a',render,'cpu')
    assert len(calls)==1
    torch.testing.assert_close(first,rgb,rtol=0,atol=0)
    torch.testing.assert_close(second,alpha,rtol=0,atol=0)
    assert cache.stats()['bytes']==4*2*4*4


def test_capacity_and_disabled_cache():
    for capacity,expected in [(0,3),(1,3),(2,2)]:
        cache=FrozenReferenceBundleCache(capacity);calls=[]
        def render():calls.append(1);return torch.zeros(3,1,1),torch.ones(1,1,1)
        for key in ('a','b','a'):cache.get(key,render,'cpu')
        assert len(calls)==expected and len(cache.values)<=capacity


def test_bundle_rejects_misaligned_contribution():
    cache=FrozenReferenceBundleCache(1)
    with pytest.raises(ValueError,match='Aligned'):
        cache.get('a',lambda:(torch.zeros(3,2,2),torch.zeros(1,1,1)),'cpu')
    assert not cache.values
