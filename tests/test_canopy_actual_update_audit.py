import torch
from scripts.canopy_actual_update_audit import ActualUpdateAudit


def test_postprocess_can_reverse_adam_descent():
    p=torch.nn.Parameter(torch.tensor([1.,2.])); p.grad=torch.tensor([1.,1.])
    audit=ActualUpdateAudit([('candidate.logits',p)])
    with torch.no_grad():p.sub_(.1)
    assert audit.measure()['gradient_dot_delta']<0
    with torch.no_grad():p.add_(.2)
    assert audit.measure()['gradient_dot_delta']>0


def test_zero_gradient_momentum_movement_is_measured():
    p=torch.nn.Parameter(torch.tensor([1.,2.])); opt=torch.optim.Adam([p],lr=.1)
    p.grad=torch.ones_like(p);opt.step();p.grad=torch.zeros_like(p)
    audit=ActualUpdateAudit([('candidate.logits',p)]);opt.step()
    record=audit.measure()['parameters']['candidate.logits']
    assert record['zero_gradient_changed_rows']==2 and record['tau_decreased_rows']==2
