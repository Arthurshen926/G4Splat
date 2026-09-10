import copy
import numpy as np
import pytest
import torch
from outdoor.canonical_canopy_initialization import CONTRACT,validate_fresh_canonical_seed,exclude_validation_cameras
from outdoor.hybrid_gaussian_renderer import LAYER_CANONICAL_CROWN,VERIFICATION_MEASURED_SINGLE,VERIFICATION_VERIFIED


def fixture():
    cameras=[{'image_id':i,'image_name':f'seq2__frame{i:05d}','sequence_id':'seq2'} for i in (1,2,3)]
    payload={'centers':torch.tensor([[0.,0.,3.],[1.,0.,3.]]),
        'scales':torch.full((2,3),.02),'colors':torch.full((2,3),.4),
        'quaternions':torch.tensor([[1.,0.,0.,0.]]*2),'opacities':torch.full((2,1),.04),
        'static_detail':torch.ones(2,dtype=torch.bool),'layer_role':torch.full((2,),LAYER_CANONICAL_CROWN),
        'verification_state':torch.tensor([VERIFICATION_VERIFIED,VERIFICATION_MEASURED_SINGLE]),
        'support_camera_ids':torch.tensor([[1,2],[2,-1]]),
        'support_view_count':torch.tensor([2,1]),
        'observation_camera_ids':torch.tensor([[1,-1],[2,-1]]),
        'verified_camera_ids':torch.tensor([[1,2],[2,-1]]),
        'observation_uv':torch.full((2,2,2),.5),'observation_depth':torch.full((2,2),3.),
        'audit':{'fresh_canonical_leaves':{'contract':CONTRACT,'optimizer_steps':0,'proposal_pool':False,
            'canonical_sequence':'seq2','excluded_camera_ids':[3],
            'depth_profiles_by_image':{'seq2__frame00001':1.2,'seq2__frame00002':.8}}}}
    return payload,cameras


def test_fresh_seed_preserves_authority_and_parameters_without_fusion():
    payload,cameras=fixture();before=copy.deepcopy(payload)
    audit=validate_fresh_canonical_seed(payload,cameras)
    assert audit['canonical_scene_sequence']=='seq2' and audit['output_static_rows']==2
    for key,value in before.items():
        if torch.is_tensor(value): assert torch.equal(value,payload[key])


@pytest.mark.parametrize('problem',['trained','proposal','excluded','duplicate_witness','missing_profile','noncanonical','opacity','unmeasured','stale_support_count'])
def test_bad_seed_authority_is_rejected(problem):
    payload,cameras=fixture();audit=payload['audit']['fresh_canonical_leaves']
    if problem=='trained':audit['optimizer_steps']=1
    if problem=='proposal':audit['proposal_pool']=True
    if problem=='excluded':payload['verified_camera_ids'][0,1]=3
    if problem=='duplicate_witness':payload['verified_camera_ids'][0,1]=1
    if problem=='missing_profile':audit['depth_profiles_by_image'].pop('seq2__frame00002')
    if problem=='noncanonical':cameras[1]['sequence_id']='seq1'
    if problem=='opacity':payload['opacities'][0]=.2
    if problem=='unmeasured':payload['observation_depth'][0,0]=float('nan')
    if problem=='stale_support_count':payload['support_view_count'][0]=1
    with pytest.raises(ValueError):validate_fresh_canonical_seed(payload,cameras)


def test_exclusions_apply_to_every_stream_without_renumbering_cameras():
    schedules={'rgb':np.array([1,2,3,4,1,2]),'canopy':np.array([3,2,1,3])}
    out=exclude_validation_cameras(schedules,{2,3})
    assert out['rgb'].tolist()==[1,4,1,1,4,1]
    assert out['canopy'].tolist()==[1,1,1,1]
    assert schedules['rgb'].tolist()==[1,2,3,4,1,2]
    assert exclude_validation_cameras(schedules,set()) is schedules
    with pytest.raises(ValueError):exclude_validation_cameras(schedules,{1,2,3,4})


@pytest.mark.parametrize('problem',['valid','asymmetric','negative','unknown_contract'])
def test_versioned_measured_covariance_is_validated(problem):
    from outdoor.canopy_position_uncertainty import CONTRACT as POSITION_CONTRACT
    payload,cameras=fixture()
    payload['audit']['fresh_canonical_leaves']['position_uncertainty_contract']=POSITION_CONTRACT
    payload['position_covariance']=torch.eye(3)[None].repeat(2,1,1)
    if problem=='asymmetric':payload['position_covariance'][0,0,1]=.4
    if problem=='negative':payload['position_covariance'][0,0,0]=-1
    if problem=='unknown_contract':payload['audit']['fresh_canonical_leaves']['position_uncertainty_contract']='unknown'
    if problem=='valid':validate_fresh_canonical_seed(payload,cameras)
    else:
        with pytest.raises(ValueError):validate_fresh_canonical_seed(payload,cameras)


def test_runtime_canopy_profiles_leave_scene_gauge_unchanged():
    from outdoor.training_evidence import OutdoorGeometryEvidence
    geometry=OutdoorGeometryEvidence.__new__(OutdoorGeometryEvidence)
    geometry.moge3_records={'a':{},'b':{}}
    geometry.configure_moge3_metric_scale(.83)
    profiles={'a':1.8,'b':.7}
    geometry.configure_moge3_canopy_depth_scales(profiles);profiles['a']=4.
    assert geometry.moge3_metric_to_cambridge_scale==.83
    assert geometry.moge3_canopy_depth_scales['a']==1.8
    with pytest.raises(ValueError):geometry.configure_moge3_canopy_depth_scales({'missing':1.})
    with pytest.raises(ValueError):geometry.configure_moge3_canopy_depth_scales({'a':float('nan')})


def test_measured_lifecycle_has_no_empty_envelope_bootstrap_and_preserves_polish():
    from scripts.train_unified_outdoor_teacher import _phase,_resolved_phase_schedule,_static_detail_stage_visible
    name='static_measured_handoff_quality'
    assert _phase(0,30000,name)=='static_foliage'
    assert _static_detail_stage_visible(_phase(0,30000,name))
    assert _phase(25499,30000,name)=='ownership_cleanup'
    assert _phase(25500,30000,name)=='canonical_polish'
    assert _resolved_phase_schedule(name,30000)==(
        ('static_foliage',13500),('dynamic_appearance',19500),
        ('ownership_cleanup',25500),('canonical_polish',30000))
    # The legacy schedule remains a distinct, unchanged control.
    assert _phase(0,30000,'static_handoff_quality')=='canonical_bootstrap'
