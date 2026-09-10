from types import SimpleNamespace
import copy
import torch
from outdoor.hybrid_gaussian_renderer import VolumetricFoliageModel,LAYER_CANONICAL_CROWN,VERIFICATION_VERIFIED
from outdoor.canopy_support_expansion import temporary_camera_support
from scripts.train_unified_outdoor_teacher import _adapt_volume,_volume_stats


def test_retained_native_witness_unlocks_actual_topology_not_only_audit_count():
    payload={'version':'independent_sfm_semantic_canopy_volume_v1',
        'centers':torch.tensor([[0.,0.,3.],[1.,0.,3.]]),
        'scales':torch.full((2,3),.05),'colors':torch.full((2,3),.4),
        'opacities':torch.full((2,1),.2),'quaternions':torch.tensor([[1.,0.,0.,0.]]*2),
        'layer_role':torch.full((2,),LAYER_CANONICAL_CROWN,dtype=torch.int8),
        'static_detail':torch.ones(2,dtype=torch.bool),
        'support_camera_ids':torch.tensor([[1,-1],[1,-1]],dtype=torch.int32),
        'verified_camera_ids':torch.tensor([[1,-1],[1,-1]],dtype=torch.int32),
        'support_view_count':torch.ones(2,dtype=torch.int16),
        'support_sequence_count':torch.ones(2,dtype=torch.int16),
        'occupancy_probability':torch.full((2,),.9)}
    results=[]
    for corrected in [False,True]:
        leaf=VolumetricFoliageModel(1,device='cpu');leaf.initialize_from_volume_state(copy.deepcopy(payload))
        with temporary_camera_support(leaf,torch.tensor([0,1]),2):
            leaf.verified_camera_ids[:,1]=2
            leaf.verified_camera_count.fill_(2);leaf.verified_sequence_count.fill_(1)
            leaf.verification_state.fill_(VERIFICATION_VERIFIED)
        if not corrected:leaf.support_view_count.fill_(1)
        stats=_volume_stats(leaf)
        for key in ('contribution','residual','gradient','gradient_count'):stats[key].fill_(1.)
        stats['radius'].fill_(6.)
        event=_adapt_volume(SimpleNamespace(volume_split_radius=3.,maximum_volume_gaussians=20,
            maximum_volume_splits=2),leaf,stats,volume_budget=20,phase='static_foliage',current_iteration=400)
        results.append(event['split_parents'])
        if corrected:
            assert (leaf.verification_state!=VERIFICATION_VERIFIED).all()
            assert (leaf.verified_camera_count==0).all()
    assert results[0]==0 and results[1]>0
