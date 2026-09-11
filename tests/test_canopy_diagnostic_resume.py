import copy
import pytest
import torch
from scripts.canopy_diagnostic_resume import validate_resume, validate_module_state


def fixture():
    manifest = dict(args=dict(steps=8, gradient_accumulation=2, output='new', resume='saved', stop_after=None), script_sha256='same')
    saved = copy.deepcopy(manifest); saved['args'].update(output='old', resume=None, stop_after=4)
    payload = dict(diagnostic_only=True, diagnostic_optimizer_state_version=2, step=4,
        completed_training_images=4, completed_optimizer_updates=2, optimizer_schedule_horizon=8,
        gradient_accumulation_pending=False, manifest=saved, optimizer_parameter_layout=[['x']],
        training_order=[3, 2], cuda_rng=[])
    return manifest, payload


def test_resume_horizon_identity_and_fresh_output(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 0)
    manifest, payload = fixture()
    assert validate_resume(payload, manifest, [['x']], [3, 2]) == 4
    for key, value in [('optimizer_schedule_horizon', 12), ('gradient_accumulation_pending', True),
                       ('diagnostic_optimizer_state_version', 1), ('completed_optimizer_updates', 3)]:
        bad = copy.deepcopy(payload); bad[key] = value
        with pytest.raises(ValueError): validate_resume(bad, manifest, [['x']], [3, 2])
    with pytest.raises(ValueError): validate_resume(payload, manifest, [['y']], [3, 2])
    with pytest.raises(ValueError): validate_resume(payload, manifest, [['x']], [2, 3])
    manifest['script_sha256'] = 'different'
    with pytest.raises(ValueError): validate_resume(payload, manifest, [['x']], [3, 2])


def test_fixed_buffers_not_overwritten_by_resume():
    module = torch.nn.Linear(2, 1)
    module.register_buffer('bounds', torch.ones(2))
    state = copy.deepcopy(module.state_dict()); state['weight'].add_(1)
    validate_module_state(module, state)
    state['bounds'][0] = 2
    with pytest.raises(ValueError): validate_module_state(module, state)


def test_adam_saved_state_continuation_exact():
    a = torch.nn.Linear(2, 1).double(); opt = torch.optim.Adam(a.parameters(), lr=.01)
    def step(model, optimizer, k):
        optimizer.zero_grad(); model(torch.tensor([[k, 1.]], dtype=torch.double)).square().sum().backward(); optimizer.step()
    for k in range(4): step(a, opt, k)
    b = copy.deepcopy(a); resumed = torch.optim.Adam(b.parameters(), lr=.01)
    resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
    for k in range(4, 8): step(a, opt, k); step(b, resumed, k)
    for x, y in zip(a.parameters(), b.parameters()): assert torch.equal(x, y)
def test_causal_history_survives_a_zero_update_tail_and_rejects_missing_flags():
    import pytest
    from scripts.canopy_diagnostic_resume import CAUSAL_HISTORY_KEYS, validate_causal_history
    previous = dict.fromkeys(CAUSAL_HISTORY_KEYS, True)
    restored = validate_causal_history(previous)
    assert all(old or False for old in restored)
    with pytest.raises(ValueError, match='causal history'):
        validate_causal_history(None)
    previous['saw_update'] = 1
    with pytest.raises(ValueError, match='causal history'):
        validate_causal_history(previous)
