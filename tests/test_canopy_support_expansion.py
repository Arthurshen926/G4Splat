from types import SimpleNamespace
import torch
from outdoor.canopy_support_expansion import independent_canopy_camera_candidates
from outdoor.canopy_support_expansion import temporary_camera_support
from outdoor.canopy_support_expansion import measured_single_canonical_eligibility
from outdoor.canopy_support_expansion import front_of_known_rigid_mask
import pytest


def test_new_camera_candidate_requires_known_canopy_and_current_depth():
    view=SimpleNamespace(world_view_transform=torch.eye(4),focal_x=10.,focal_y=10.,cx=2.,cy=2.)
    xyz=torch.tensor([[0.,0.,5.],[0.,0.,6.],[0.,0.,-5.],[10.,0.,5.],[-.5,0.,5.],[0.,0.,5.]])
    bounds=torch.stack((torch.full((5,5),4.8),torch.full((5,5),5.2)))
    mask=torch.ones(5,5,dtype=torch.bool);mask[2,1]=False
    eligible=torch.tensor([True,True,True,True,True,False])
    original=xyz.clone()
    rigid=dict(rigid_depth=torch.full((5,5),10.),rigid_alpha=torch.ones(5,5))
    result=independent_canopy_camera_candidates(xyz,view,bounds,mask,eligible,**rigid)
    assert result.tolist()==[0]
    assert torch.equal(xyz,original)
    bounds[:,2,2]=float('nan')
    assert not len(independent_canopy_camera_candidates(xyz,view,bounds,mask,eligible,**rigid))


def test_candidate_uses_exact_extrinsic_translation():
    transform=torch.eye(4);transform[3,0]=-1.
    view=SimpleNamespace(world_view_transform=transform,focal_x=10.,focal_y=10.,cx=2.,cy=2.)
    xyz=torch.tensor([[1.,0.,5.],[0.,0.,5.]])
    bounds=torch.stack((torch.full((5,5),4.8),torch.full((5,5),5.2)))
    mask=torch.zeros(5,5,dtype=torch.bool);mask[2,2]=True
    assert independent_canopy_camera_candidates(xyz,view,bounds,mask,torch.ones(2,dtype=torch.bool),
        rigid_depth=torch.full((5,5),10.),rigid_alpha=torch.ones(5,5)).tolist()==[0]


def test_tiny_native_tail_behind_nearly_opaque_surface_cannot_fund_new_support():
    view=SimpleNamespace(world_view_transform=torch.eye(4),focal_x=10.,focal_y=10.,cx=2.,cy=2.)
    xyz=torch.tensor([[0.,0.,5.]])
    bounds=torch.stack((torch.full((5,5),4.8),torch.full((5,5),5.2)))
    arguments=(xyz,view,bounds,torch.ones(5,5,dtype=torch.bool),torch.ones(1,dtype=torch.bool))
    assert not len(independent_canopy_camera_candidates(*arguments,
        rigid_depth=torch.full((5,5),4.9),rigid_alpha=torch.full((5,5),.999)))
    assert not len(independent_canopy_camera_candidates(*arguments,
        rigid_depth=torch.full((5,5),5.02),rigid_alpha=torch.ones(5,5)))
    assert independent_canopy_camera_candidates(*arguments,
        rigid_depth=torch.full((5,5),6.),rigid_alpha=torch.ones(5,5)).tolist()==[0]
    assert front_of_known_rigid_mask(xyz,view,torch.full((1,5,5),4.9),torch.ones(1,5,5)).tolist()==[False]


def test_real_contribution_without_front_geometry_cannot_add_camera_witness():
    from outdoor.hybrid_gaussian_renderer import VolumetricFoliageModel
    from scripts.train_unified_outdoor_teacher import _update_static_child_verification_from_render
    foliage=VolumetricFoliageModel(1,device="cpu")
    foliage.initialize_from_volume_state({"version":"independent_sfm_semantic_canopy_volume_v1",
        "centers":torch.tensor([[0.,0.,2.]]),"scales":torch.full((1,3),.1),
        "colors":torch.full((1,3),.5),"opacities":torch.full((1,1),.2),
        "quaternions":torch.tensor([[1.,0.,0.,0.]]),"layer_role":torch.tensor([0],dtype=torch.int8),
        "support_camera_ids":torch.tensor([[7,8]],dtype=torch.int32),
        "support_sequence_count":torch.tensor([1],dtype=torch.int16),
        "evidence_primitive_id":torch.tensor([11])})
    foliage.split(torch.tensor([0]),birth_iteration=20)
    responsibility=torch.zeros(len(foliage.xyz),7)
    responsibility[:,0]=.001;responsibility[:,1]=.001;responsibility[:,4:7]=.0005
    package=SimpleNamespace(responsibility=responsibility,structural_count=0)
    lookup=torch.zeros(9,dtype=torch.int16)
    features=foliage.features.detach().clone()
    audit=_update_static_child_verification_from_render(foliage,package,camera_id=7,
        camera_sequence_lookup=lookup,required_sequence_count=1,
        candidate_geometry_gate=torch.zeros(len(foliage.xyz),dtype=torch.bool))
    assert audit['new_camera_witnesses']==0
    assert torch.equal(features,foliage.features)
    assert not (foliage.verified_camera_ids==7).any()


def test_temporary_candidates_require_actual_new_witness_and_rollback_errors():
    foliage=SimpleNamespace(support_camera_ids=torch.tensor([[1,-1],[2,-1]]),
                            verified_camera_ids=torch.tensor([[1,-1],[2,-1]]),
                            support_view_count=torch.ones(2,dtype=torch.int16))
    with temporary_camera_support(foliage,torch.tensor([0,1]),3) as audit:
        assert foliage.support_camera_ids[:,1].tolist()==[3,3]
        foliage.verified_camera_ids[0,1]=3
    assert foliage.support_camera_ids.tolist()==[[1,3],[2,-1]]
    assert audit['new_real_camera_witnesses']==1
    assert foliage.support_view_count.tolist()==[2,1]
    with pytest.raises(RuntimeError):
        with temporary_camera_support(foliage,torch.tensor([1]),4):
            raise RuntimeError('render failed')
    assert foliage.support_camera_ids[1].tolist()==[2,-1]
    assert foliage.support_view_count.tolist()==[2,1]
    with pytest.raises(ValueError):
        with temporary_camera_support(foliage,torch.tensor([0]),3): pass


def test_support_expansion_does_not_borrow_noncanonical_or_proposal_authority():
    foliage=SimpleNamespace(xyz=torch.zeros(6,3),static_leaf_mask=torch.ones(6,dtype=torch.bool),
        dynamic_leaf_mask=torch.tensor([False,False,False,False,False,True]),
        verification_state=torch.tensor([3,3,0,3,3,3]),proposal_kind=torch.tensor([0,0,0,2,0,0]),
        verified_camera_count=torch.ones(6,dtype=torch.int16),
        verified_camera_ids=torch.tensor([[1,-1],[3,-1],[1,-1],[1,-1],[2,-1],[1,-1]]),
        support_camera_ids=torch.tensor([[1,-1],[3,-1],[1,-1],[1,-1],[2,-1],[1,-1]]))
    lookup=torch.tensor([-1,2,2,1])
    actual=measured_single_canonical_eligibility(foliage,2,lookup,measured_state=3,proposal_none=0)
    assert actual.tolist()==[True,False,False,False,False,False]
    assert not measured_single_canonical_eligibility(foliage,99,lookup,measured_state=3,proposal_none=0).any()
