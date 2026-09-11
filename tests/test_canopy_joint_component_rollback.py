from types import SimpleNamespace
import pytest
import torch
from scripts.canopy_joint_component_rollback import MODES, component_rollback


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('fail', [False, True])
def test_rollback_restores_every_component_even_on_failure(mode, fail):
    original=SimpleNamespace(features=torch.zeros(2,4,3),opacity_logits=torch.zeros(2,1))
    refined=SimpleNamespace(features=torch.ones(2,4,3),opacity_logits=torch.ones(2,1))
    position=SimpleNamespace(code=torch.ones(2,3))
    candidate=SimpleNamespace(position_code=torch.ones(3,3),scale_code=torch.ones(3))
    tensors=[refined.features,refined.opacity_logits,position.code,candidate.position_code,candidate.scale_code]
    snapshots=[x.clone() for x in tensors]
    try:
        with component_rollback(mode,original,refined,position,candidate) as gain:
            assert gain == (0. if mode=='candidates_disabled' else 1.)
            if mode=='source_dc_original':
                assert not refined.features[:,:1].any() and refined.features[:,1:].eq(1).all()
            elif mode=='source_sh_original':
                assert not refined.features[:,1:].any() and refined.features[:,:1].eq(1).all()
            elif mode=='source_opacity_original':assert not refined.opacity_logits.any()
            elif mode=='source_position_original':assert not position.code.any()
            elif mode=='candidate_position_initial':assert not candidate.position_code.any()
            elif mode=='candidate_size_initial':assert not candidate.scale_code.any()
            if fail: raise RuntimeError('simulated renderer error')
    except RuntimeError as exc:
        assert fail and str(exc)=='simulated renderer error'
    for tensor,snapshot in zip(tensors,snapshots):assert torch.equal(tensor,snapshot)


@pytest.mark.parametrize('mode', ['candidate_size_initial',
    'candidate_isotropic_code_initial', 'candidate_shape_initial'])
@pytest.mark.parametrize('fail', [False, True])
def test_shape_decomposition_restores_actual_scales(mode, fail):
    from scripts.canopy_selective_shape import SelectiveSpatialFootprintCandidates
    from scripts.canopy_joint_component_rollback import component_rollback
    # Exercise the actual nonlinear scale property, not just dummy tensors.
    candidate = SimpleNamespace(
        cloud=SimpleNamespace(scales=torch.tensor([[1., 2., 3.]])),
        scale_code=torch.tensor([.4]), shape_code=torch.tensor([[.5, -.2, -.3]]),
        shape_eligible=torch.tensor([True]), log_radius=.693,
        position_code=torch.zeros(1, 3))
    scales = lambda: SelectiveSpatialFootprintCandidates.scales.fget(candidate)
    before = scales().clone()
    original = SimpleNamespace(features=torch.zeros(1, 2, 3), opacity_logits=torch.zeros(1))
    refined = SimpleNamespace(features=torch.ones(1, 2, 3), opacity_logits=torch.ones(1))
    position = SimpleNamespace(code=torch.zeros(1, 3))
    try:
        with component_rollback(mode, original, refined, position, candidate):
            if mode == 'candidate_size_initial':
                assert torch.equal(scales(), candidate.cloud.scales)
            elif mode == 'candidate_isotropic_code_initial':
                assert not candidate.scale_code.any() and candidate.shape_code.any()
            else:
                assert not candidate.shape_code.any() and candidate.scale_code.any()
            if fail:
                raise RuntimeError('render failed')
    except RuntimeError as exc:
        assert fail and str(exc) == 'render failed'
    assert torch.equal(scales(), before)
