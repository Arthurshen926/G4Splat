import torch
import pytest
from scripts.canopy_candidate_gradient_conflict import summarize_gradient_balance
from scripts.canopy_candidate_gradient_conflict import candidate_adam_first_moment


def test_training_gradient_signs_keep_opposing_views_separate():
    components = {'canopy_positive': torch.tensor([1., 0.]), 'canopy_negative': torch.tensor([3., 2.]),
                  'background_positive': torch.tensor([4., 0.]), 'background_negative': torch.tensor([0., 0.])}
    out = summarize_gradient_balance(components, torch.tensor([3., 1.]))
    assert out['opacity_decrease_net_weight_fraction'] == .75
    assert out['opacity_increase_net_weight_fraction'] == .25
    assert out['canopy_positive'] == .75 and out['canopy_negative'] == 2.75


@pytest.mark.parametrize('key', ['rigid_boundary_preservation_weight', 'surface_rgb_feasibility_weight'])
def test_gradient_audit_cannot_silently_omit_new_training_losses(key):
    from scripts.canopy_candidate_gradient_conflict import training_gradient_conflict
    with pytest.raises(ValueError, match='auxiliary losses'):
        training_gradient_conflict(None, None, None, None, None, None, {'args': {key: 1.}}, None, None)


def test_saved_moment_is_resolved_by_parameter_name_not_assumed_group_index():
    moment = torch.tensor([-.2, .1])
    payload = dict(optimizer_parameter_layout=[['source.features'], ['candidate.cloud.logits']],
        optimizer_state=dict(param_groups=[dict(params=[8]), dict(params=[3])], state={3:dict(exp_avg=moment)}))
    assert torch.equal(candidate_adam_first_moment(payload), moment)
    assert candidate_adam_first_moment({}) is None
    components = {k: torch.zeros(2) for k in ('canopy_positive', 'canopy_negative', 'background_positive', 'background_negative')}
    components['canopy_positive'][0] = 1.; components['canopy_negative'][1] = 1.
    result = summarize_gradient_balance(components, torch.tensor([3., 1.]), moment)
    assert result['last_adam_moment_increase_weight_fraction'] == .75
    assert result['net_gradient_opposes_last_adam_moment_weight_fraction'] == 1.
