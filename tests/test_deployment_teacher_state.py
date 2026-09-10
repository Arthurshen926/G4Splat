from types import SimpleNamespace
import pytest
import torch
from scripts.deployment_teacher_state import deployment_surface_capture, deployment_birth_audit
from scripts.train_unified_outdoor_teacher import _assert_training_resume_payload


def test_deployment_capture_preserves_native_values_without_mutating_checkpoint():
    state=[torch.arange(3) for _ in range(13)]
    state[0]=3;state[10]={'state':{'exp_avg':torch.ones(4)}}
    state[12]={'source_type':torch.tensor([1,2,3])}
    capture=tuple(state)
    result=deployment_surface_capture(capture)
    assert result[10]=={} and capture[10]['state']
    assert all(result[i] is capture[i] for i in range(13) if i!=10)
    with pytest.raises(RuntimeError):
        _assert_training_resume_payload({'checkpoint_kind':'render_state_only_nonresumable','surface':result})


def test_deployment_birth_audit_never_captures_or_iterates_candidates():
    class CountOnly:
        def __len__(self):return 3000000
        def __iter__(self):raise AssertionError('Deployment walked a training pool')
    def forbidden():raise AssertionError('Deployment captured training tensors')
    acc=SimpleNamespace(total_proposals=10,total_births=2,cells=CountOnly(),
        visual_hull_cells=CountOnly(),consumed_first_hit_witnesses=CountOnly(),capture=forbidden)
    audit=deployment_birth_audit(acc)
    assert not audit['training_candidate_pool_exported']
    assert audit['pending_visual_hull_cells']==3000000


def test_unrecognized_surface_schema_is_rejected():
    with pytest.raises(ValueError):deployment_surface_capture((1,2,3))


def test_only_deployment_export_omits_pool_and_optimizer():
    import ast
    from pathlib import Path
    tree=ast.parse((Path(__file__).resolve().parents[1]/'scripts/train_unified_outdoor_teacher.py').read_text())
    final=[];checkpoints=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or len(node.args)<2:
            continue
        if not isinstance(node.args[1],ast.Dict):continue
        values={k.value:v for k,v in zip(node.args[1].keys,node.args[1].values) if isinstance(k,ast.Constant)}
        if node.func.id=='_atomic_torch_save' and isinstance(node.args[0],ast.Name) and node.args[0].id=='teacher_state':
            final.append(values)
        if node.func.id=='_save_checkpoint' and 'static_ray_birth_state' in values:
            checkpoints.append(values)
    assert len(final)==1 and checkpoints
    assert 'static_ray_birth_state' not in final[0]
    assert final[0]['checkpoint_kind'].value=='render_state_only_nonresumable'
    assert final[0]['surface'].func.id=='deployment_surface_capture'
    for payload in checkpoints:
        assert isinstance(payload['surface'].func,ast.Attribute)
        assert payload['surface'].func.attr=='capture'
        assert 'volume_optimizer' in payload
