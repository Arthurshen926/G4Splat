from types import SimpleNamespace
import pytest
import torch
from outdoor.hybrid_gaussian_renderer import VolumetricFoliageModel, static_detail_forward_visibility_gate
from scripts.train_unified_outdoor_teacher import _adapt_volume, _volume_stats, _finalize_synchronous_measured_splits


@pytest.mark.parametrize('mode', ['all', 'none', 'partial', 'mixed'])
def test_split_publication_preserves_verified_parent_until_whole_family_passes(mode):
    leaf=VolumetricFoliageModel(1,device='cpu')
    leaf.initialize_from_volume_state({
        'version':'independent_sfm_semantic_canopy_volume_v1',
        'centers':torch.tensor([[0.,0.,3.],[1.,0.,3.]]),
        'scales':torch.full((2,3),.05),'colors':torch.full((2,3),.4),
        'opacities':torch.full((2,1),.2),'quaternions':torch.tensor([[1.,0.,0.,0.]]*2),
        'layer_role':torch.zeros(2,dtype=torch.int8),'static_detail':torch.ones(2,dtype=torch.bool),
        'support_camera_ids':torch.tensor([[1,2],[1,2]],dtype=torch.int32),
        'verified_camera_ids':torch.tensor([[1,2],[1,2]],dtype=torch.int32),
        'verified_camera_count':torch.full((2,),2,dtype=torch.int16),
        'verified_sequence_count':torch.ones(2,dtype=torch.int16),
        'verification_state':torch.ones(2,dtype=torch.int8),
        'support_view_count':torch.full((2,),2,dtype=torch.int16),
        'support_sequence_count':torch.ones(2,dtype=torch.int16),
        'occupancy_probability':torch.full((2,),.9),
    })
    before={k:getattr(leaf,k).detach().clone() for k in ['xyz','log_scales','quaternions','features','opacity_logits']}
    args=SimpleNamespace(volume_split_radius=3.,maximum_volume_gaussians=20,maximum_volume_splits=2)
    stats=_volume_stats(leaf)
    for key in ('contribution','residual','gradient','gradient_count'):stats[key].fill_(1.)
    stats['radius'].fill_(6.)
    event=_adapt_volume(args,leaf,stats,volume_budget=20,phase='static_foliage',current_iteration=400)
    assert event['split_parents']>0
    assert not static_detail_forward_visibility_gate(leaf,99,include_pending_exact=False).any()

    def verify(rows):
        accepted=rows if mode=='all' else rows[:1] if mode=='partial' else rows[:0]
        if mode=='mixed':
            accepted=rows[leaf.split_proposal_family_id[rows]==leaf.split_proposal_family_id[rows[0]]]
        leaf.verification_state[accepted]=1
        leaf.verified_camera_count[accepted]=2
        leaf.verified_sequence_count[accepted]=1
        leaf.verified_camera_ids[accepted]=leaf.support_camera_ids[accepted]
        return {'test_witness_rows':len(accepted)}

    _finalize_synchronous_measured_splits(args,leaf,event,current_iteration=400,verify_children=verify)
    assert static_detail_forward_visibility_gate(leaf,99,include_pending_exact=False).all()
    assert not leaf.split_parent_snapshot_family_id.numel()
    assert not leaf.proposal_kind.any()
    assert event['new_count']==len(leaf)
    assert len(event['_new_to_old'])==len(leaf)
    restored=VolumetricFoliageModel(1,device='cpu');restored.restore(leaf.capture())
    torch.testing.assert_close(restored.xyz,leaf.xyz)
    if mode in ('none','partial'):
        assert len(leaf)==2 and event['net_growth']==0 and event['split_parents']==0
        assert not event['_synchronously_committed_split_rows'].numel()
        for key,value in before.items():
            torch.testing.assert_close(getattr(leaf,key),value[event['_new_to_old']],atol=0,rtol=0)
    elif mode=='all':
        assert len(leaf)>2 and event['split_parents']>0
        assert len(event['_synchronously_committed_split_rows'])==len(leaf)
    else:
        committed=event['_synchronously_committed_split_rows']
        assert event['split_parents']==1 and len(leaf)==3 and len(committed)==2
        parent=torch.ones(len(leaf),dtype=torch.bool);parent[committed]=False
        for key,value in before.items():
            torch.testing.assert_close(getattr(leaf,key)[parent],value[event['_new_to_old'][parent]],atol=0,rtol=0)
