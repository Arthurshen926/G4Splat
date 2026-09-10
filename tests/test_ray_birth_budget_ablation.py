from copy import deepcopy
import pytest
from scripts.train_unified_outdoor_teacher import (
    _resume_training_contract_differences,STATIC_RAY_BIRTH_VISUAL_HULL_CONTRACT)


def contracts():
    old={'static_ray_birth':dict(contract=STATIC_RAY_BIRTH_VISUAL_HULL_CONTRACT,
         maximum_births_per_topology_event=2048,minimum_cameras=2,initial_opacity=.04),
         'maximum_volume_gaussians':2000000}
    new=deepcopy(old);new['static_ray_birth']['maximum_births_per_topology_event']=8192
    return old,new


def test_explicit_allocator_ablation_preserves_inputs_and_all_other_contracts():
    old,new=contracts();original=deepcopy(old)
    assert 'static_ray_birth' in _resume_training_contract_differences(old,new)
    assert not _resume_training_contract_differences(old,new,allow_static_ray_birth_budget_ablation=True)
    assert old==original
    new['maximum_volume_gaussians']+=1
    assert 'maximum_volume_gaussians' in _resume_training_contract_differences(old,new,allow_static_ray_birth_budget_ablation=True)


@pytest.mark.parametrize('key,value',[('minimum_cameras',1),('initial_opacity',.8),
    ('maximum_births_per_topology_event',0),('maximum_births_per_topology_event',8192.)])
def test_allocator_ablation_never_relaxes_evidence_or_optical_constraints(key,value):
    old,new=contracts();new['static_ray_birth'][key]=value
    with pytest.raises(RuntimeError,match='only its positive integer event cap'):
        _resume_training_contract_differences(old,new,allow_static_ray_birth_budget_ablation=True)
