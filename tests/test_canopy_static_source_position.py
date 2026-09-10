import pytest
import torch
from scripts.canopy_static_source_position import StaticSourcePosition, StaticSourcePositionView


def make():
    xyz = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    scales = torch.ones_like(xyz)*.2
    eligible = torch.tensor([False, True, False, True])
    return xyz, eligible, StaticSourcePosition(xyz, scales, eligible, 2.)


def test_identity_bounded_motion_and_ineligible_rows_are_exact():
    xyz, eligible, p = make()
    assert torch.equal(p.apply(xyz), xyz)
    p.apply(xyz).sum().backward()
    assert p.code.grad.abs().sum() > 0
    with torch.no_grad(): p.code.fill_(100.)
    updated = p.apply(xyz)
    assert torch.equal(updated[~eligible], xyz[~eligible])
    assert ((updated-xyz).norm(dim=1) <= .400001).all()
    assert p.audit()['moved_rows'] == 2


def test_state_restore_cannot_change_eligible_rows_or_initial_bounds():
    xyz, eligible, p = make()
    with torch.no_grad(): p.code.fill_(.1)
    state = {k: v.clone() for k, v in p.state_dict().items()}
    _, _, q = make(); q.load_verified_state(state)
    assert torch.equal(p.apply(xyz), q.apply(xyz))
    for key in ('indices', 'extent', 'code'):
        corrupt = {k: v.clone() for k, v in state.items()}
        if key == 'code': corrupt[key][0, 0] = float('nan')
        else: corrupt[key][0] = 0
        with pytest.raises(ValueError, match='authority'): q.load_verified_state(corrupt)
    corrupt = {k: v.clone() for k, v in state.items()}
    corrupt['indices'] = corrupt['indices'].float()+.5
    with pytest.raises(ValueError, match='authority'): q.load_verified_state(corrupt)


def test_wrapper_preserves_nonposition_state_and_rejects_temporal_render():
    xyz, eligible, p = make()
    class Base:
        def __len__(self): return 4
        def conditioned_state(self, temporal_code, **kwargs): return self.xyz, self.features, self.opacity
    b = Base(); b.xyz = xyz; b.features = torch.ones(4, 1, 3); b.opacity = torch.ones(4)
    b.scales = torch.ones(4, 3); b.static_leaf_mask = eligible
    b.dynamic_leaf_mask = b.persistent_envelope_mask = torch.zeros(4, dtype=torch.bool)
    v = StaticSourcePositionView(b, p)
    out = v.conditioned_state(None, include_dynamic=False)
    assert torch.equal(out[0], xyz) and out[1] is b.features and out[2] is b.opacity
    assert v.scales is b.scales
    with pytest.raises(ValueError, match='static'): v.conditioned_state(torch.zeros(1), include_dynamic=False)
    with pytest.raises(ValueError, match='static'): v.conditioned_state(None, include_dynamic=True)
    with pytest.raises(ValueError, match='authority'):
        v.conditioned_state(None, include_dynamic=False, base_geometry_gradient_gate=torch.zeros(4))
