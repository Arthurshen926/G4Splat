import copy
import pytest
import torch
from scripts.canopy_row_active_adam import RowActiveAdam


def test_all_active_rows_match_standard_adam():
    a = torch.nn.Parameter(torch.ones(4, 3, dtype=torch.float64)); b = torch.nn.Parameter(a.detach().clone())
    oa = RowActiveAdam([a], lr=.02); ob = torch.optim.Adam([b], lr=.02, eps=1e-15)
    for step in range(12):
        grad = torch.arange(1, 13, dtype=a.dtype).reshape_as(a)*(step+1)/10
        a.grad = grad; b.grad = grad.clone(); oa.step(); ob.step()
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


def test_zero_gradient_rows_keep_parameters_moments_and_clock():
    p = torch.nn.Parameter(torch.zeros(2, 3)); o = RowActiveAdam([p])
    p.grad = torch.ones_like(p); o.step()
    before = p.detach().clone(); state = copy.deepcopy(o.state[p])
    p.grad = torch.tensor([[0., 0., 0.], [1., 1., 1.]])
    for _ in range(8): o.step()
    assert torch.equal(p[0], before[0])
    for key in ('step', 'exp_avg', 'exp_avg_sq'): assert torch.equal(o.state[p][key][0], state[key][0])
    assert o.state[p]['step'].tolist() == [1, 9]


def test_saved_row_clock_and_future_update_replay():
    p = torch.nn.Parameter(torch.zeros(2)); o = RowActiveAdam([p], lr=.03)
    p.grad = torch.tensor([1., 0.]); o.step()
    q = torch.nn.Parameter(p.detach().clone()); restored = RowActiveAdam([q])
    restored.load_state_dict(copy.deepcopy(o.state_dict()))
    for grad in (torch.tensor([0., 2.]), torch.tensor([3., 1.])):
        p.grad = grad; q.grad = grad.clone(); o.step(); restored.step()
        assert torch.equal(p, q)
    assert restored.state[q]['step'].dtype == torch.long


def test_nonfinite_gradient_cannot_partially_update_earlier_group():
    p = torch.nn.Parameter(torch.zeros(2)); q = torch.nn.Parameter(torch.zeros(2))
    o = RowActiveAdam([p, q]); p.grad = torch.ones_like(p); q.grad = torch.full_like(q, float('nan'))
    with pytest.raises(ValueError): o.step()
    assert torch.equal(p, torch.zeros_like(p)) and not o.state
