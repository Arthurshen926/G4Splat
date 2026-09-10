from types import SimpleNamespace
import pytest
from scripts.calibrate_canopy_optical_ablation import _validate_joint_checkpoint_refinement
from scripts.calibrate_canopy_optical_ablation import _restore_parameter_update_authority_
import torch


def arguments(**changes):
    values=dict(refine_checkpoint_leaves=True, depth_profile_initialization='matched-seed',
        replacement_foliage=None, evaluate_opacity=None, position_mode='free',
        native_rigid_alpha_weight=3., production_geometry_objective='none',
        production_color_objective='none')
    values.update(changes)
    return SimpleNamespace(**values)


def test_explicit_checkpoint_joint_refinement_allowed():
    _validate_joint_checkpoint_refinement(arguments())
    _validate_joint_checkpoint_refinement(arguments(refine_checkpoint_leaves=False,
        depth_profile_initialization=None,native_rigid_alpha_weight=0.))


@pytest.mark.parametrize('changes', [
    dict(depth_profile_initialization=None), dict(replacement_foliage='other'),
    dict(evaluate_opacity='replay'), dict(position_mode='source-ray'),
    dict(native_rigid_alpha_weight=-1.), dict(native_rigid_alpha_weight=float('nan')),
    dict(native_rigid_alpha_weight=float('inf')), dict(production_geometry_objective='native'),
    dict(production_color_objective='native'),
])
def test_checkpoint_joint_refinement_rejects_ambiguous_sources(changes):
    with pytest.raises(ValueError):
        _validate_joint_checkpoint_refinement(arguments(**changes))


def test_single_camera_rows_cannot_drift_on_high_order_adam_momentum():
    parameter=torch.nn.Parameter(torch.zeros(3,2,3))
    optimizer=torch.optim.Adam([parameter],lr=.1)
    previous=parameter.detach().clone()
    parameter.grad=torch.ones_like(parameter);optimizer.step()
    owner=torch.tensor([True,True,False]);persistent=torch.tensor([True,False,False])
    _restore_parameter_update_authority_(parameter,previous,owner,optimizer.state[parameter],high_order_owner=persistent)
    assert parameter[0].ne(0).all() and parameter[1,0].ne(0).all()
    assert not parameter[1,1:].any() and not parameter[2].any()
    for key in ('exp_avg','exp_avg_sq'):
        assert not optimizer.state[parameter][key][1,1:].any()
        assert not optimizer.state[parameter][key][2].any()
