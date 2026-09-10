from copy import deepcopy
import torch
from outdoor.hybrid_gaussian_renderer import VolumetricFoliageModel
from outdoor.mass_consistent_foliage import MassConsistentFoliage, IdentityPreservingMassFoliage
from scripts.train_unified_outdoor_teacher import (
    _resume_training_contract_differences, _trainer_repair_hash_change_is_allowed,
)


def _model(cls):
    model = cls(1, dynamic_rank=0, device='cpu')
    model.initialize_from_volume_state(dict(
        version='independent_sfm_semantic_canopy_volume_v1',
        centers=torch.tensor([[0., 0., 2.], [.2, 0., 2.]]),
        scales=torch.tensor([[.2, .16, .14]]*2), colors=torch.full((2, 3), .4),
        opacities=torch.full((2, 1), .3),
        quaternions=torch.tensor([[1., 0., 0., 0.]]*2),
        layer_role=torch.zeros(2, dtype=torch.int8),
        tree_instance_id=torch.zeros(2, dtype=torch.int32),
        replacement_group=torch.arange(2, dtype=torch.int64)))
    return model


def test_training_wrapper_is_forward_identical_and_respects_geometry_authority():
    base, fixed = _model(VolumetricFoliageModel), _model(MassConsistentFoliage)
    gate = torch.tensor([0., 1.])
    args = dict(include_dynamic=False, base_geometry_gradient_gate=gate)
    before = base.conditioned_state(None, **args)
    after = fixed.conditioned_state(None, **args)
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    gl = torch.autograd.grad(before[2].sum(), base.opacity_logits)[0]
    fl, fs = torch.autograd.grad(after[2].sum(), (fixed.opacity_logits, fixed.log_scales))
    assert torch.equal(gl, fl)
    assert torch.count_nonzero(fs[0]) == 0 and torch.count_nonzero(fs[1]) > 0
    assert base.capture().keys() == fixed.capture().keys()


def test_opacity_only_backward_cannot_acquire_scale_authority():
    model = _model(MassConsistentFoliage)
    alpha = model.conditioned_state(None, include_dynamic=False)[2]
    torch.autograd.backward(alpha.sum(), inputs=(model.opacity_logits,))
    assert model.log_scales.grad is None
    assert model.opacity_logits.grad is not None


def test_integrated_mass_regularizer_has_no_scale_gradient_in_mass_coordinates():
    model = _model(MassConsistentFoliage)
    grad = torch.autograd.grad(model.integrated_optical_mass().sum(), model.log_scales, allow_unused=True)[0]
    assert grad is None


def test_direct_opacity_priors_do_not_gain_ungated_geometry_gradients():
    model = _model(MassConsistentFoliage)
    model.opacities.square().sum().backward()
    assert model.log_scales.grad is None
    assert model.opacity_logits.grad is not None


def test_noop_mass_restore_preserves_logits_bitwise_and_keeps_row_protection():
    model = _model(IdentityPreservingMassFoliage)
    with torch.no_grad(): model.opacity_logits.copy_(torch.tensor([[-2.1234567], [.7654321]]))
    before = model.opacity_logits.detach().clone()
    mass = model.integrated_optical_mass().detach().clone()
    report = model.restore_integrated_optical_mass(mass)
    assert torch.equal(before, model.opacity_logits)
    assert report['mean_relative_alpha_compensation'] == 0
    with torch.no_grad(): model.log_scales.add_(.1)
    model.restore_integrated_optical_mass(mass, rows=torch.tensor([True, False]))
    torch.testing.assert_close(model.integrated_optical_mass()[0], mass[0], rtol=2e-6, atol=1e-8)
    assert torch.equal(model.opacity_logits[1], before[1])


def test_mass_gradient_resume_requires_explicit_change_without_loosening_other_contracts():
    old = dict(seed=1701, integrated_optical_mass_compensation=True)
    original = deepcopy(old)
    legacy = dict(old, mass_consistent_scale_gradients=False)
    new = dict(old, mass_consistent_scale_gradients=True)
    assert not _resume_training_contract_differences(old, legacy)
    assert 'mass_consistent_scale_gradients' in _resume_training_contract_differences(old, new)
    assert not _resume_training_contract_differences(old, new, allow_optical_mass_gradient_repair=True)
    new['seed'] = 1702
    assert 'seed' in _resume_training_contract_differences(old, new, allow_optical_mass_gradient_repair=True)
    assert old == original
    assert _trainer_repair_hash_change_is_allowed(
        {'trainer', 'optical_mass_gradient', 'mass_consistent_foliage'}, enabled=True)
    assert not _trainer_repair_hash_change_is_allowed({'hybrid_renderer'}, enabled=True)
