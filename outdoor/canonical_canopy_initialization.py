"""Typed handoff of fresh canonical observations, never a trained checkpoint."""
import math
import numpy as np
import torch

CONTRACT = 'fresh_exact_k_canonical_leaf_observations_rigid_anchored_v1'


def exclude_validation_cameras(schedules, excluded_indices):
    """Keep camera indices stable and preserve each training stream's length."""
    if not excluded_indices:
        return schedules
    result={}
    for name,schedule in schedules.items():
        schedule=np.asarray(schedule)
        allowed=schedule[~np.isin(schedule,list(excluded_indices))]
        if len(schedule) and not len(allowed):
            raise ValueError(f'Validation exclusion empties training stream {name}')
        result[name]=np.resize(allowed,len(schedule)).astype(schedule.dtype,copy=False)
    return result


def validate_fresh_canonical_seed(payload, fixed_cameras):
    """Preserve measured/native-witness authority without re-fusing leaves.

    The legacy fusion entry point consumes dynamic observations associated
    with envelopes. These fresh leaves already have one fixed canonical
    acquisition and must not enter its no-dynamic-observation fallback.
    """
    audit = payload.get('audit', {}).get('fresh_canonical_leaves', {})
    if audit.get('contract') != CONTRACT:
        raise ValueError('Unsupported fresh canonical seed contract')
    if audit.get('optimizer_steps') != 0 or audit.get('proposal_pool') is not False:
        raise ValueError('Only unoptimized measured leaves can initialize training')
    canonical = audit.get('canonical_sequence')
    cameras = {int(v['image_id']): v for v in fixed_cameras}
    if not canonical or not cameras:
        raise ValueError('Fixed canonical cameras are required')
    count = len(payload['centers'])
    if not count:
        raise ValueError('Fresh canopy initialization is empty')
    uncertainty = audit.get('position_uncertainty_contract')
    if uncertainty is not None:
        from outdoor.canopy_position_uncertainty import CONTRACT as POSITION_CONTRACT
        if uncertainty != POSITION_CONTRACT:
            raise ValueError('Unsupported position uncertainty contract')
        covariance=payload['position_covariance']
        if (covariance.shape!=(count,3,3) or not torch.isfinite(covariance).all()
                or not torch.allclose(covariance,covariance.transpose(1,2),atol=1e-7,rtol=1e-6)
                or (torch.linalg.cholesky_ex(covariance)[1]!=0).any()):
            raise ValueError('Invalid measured position covariance')
    for key, width in [('centers',3),('scales',3),('quaternions',4),('colors',3),('opacities',1)]:
        value=payload[key]
        if value.shape != (count,width) or not torch.isfinite(value).all():
            raise ValueError(f'Invalid fresh leaf field: {key}')
    if not (payload['scales']>0).all() or not ((payload['colors']>=0)&(payload['colors']<=1)).all():
        raise ValueError('Invalid fresh leaf scale/color')
    if not torch.allclose(payload['opacities'],torch.full_like(payload['opacities'],.04),atol=1.e-7,rtol=0):
        raise ValueError('Fresh leaves must retain their unoptimized opacity')
    # No envelope, skeleton, dynamic layer or hidden extra SH can piggyback.
    from outdoor.hybrid_gaussian_renderer import LAYER_CANONICAL_CROWN, VERIFICATION_MEASURED_SINGLE, VERIFICATION_VERIFIED
    if payload['static_detail'].shape != (count,) or not payload['static_detail'].all():
        raise ValueError('Only local static detail leaves are permitted')
    if payload['layer_role'].shape!=(count,) or not (payload['layer_role']==LAYER_CANONICAL_CROWN).all():
        raise ValueError('Unexpected non-leaf layer in canonical initialization')
    states=payload['verification_state']
    if states.shape!=(count,) or not ((states==VERIFICATION_MEASURED_SINGLE)|(states==VERIFICATION_VERIFIED)).all():
        raise ValueError('Unverified proposals cannot become measured seeds')
    support=payload['support_camera_ids']; observed=payload['observation_camera_ids']
    verified=payload['verified_camera_ids']
    if support.ndim!=2 or support.shape[0]!=count or observed.shape!=support.shape or verified.shape[0]!=count:
        raise ValueError('Invalid camera lineage table')
    ordered=support.sort(dim=1).values
    unique=ordered>=0
    unique[:,1:] &= ordered[:,1:]!=ordered[:,:-1]
    if (payload['support_view_count'].shape!=(count,)
            or not torch.equal(payload['support_view_count'].long(),unique.sum(1))):
        raise ValueError('Fresh support count must match actual distinct camera IDs')
    excluded=set(int(v) for v in audit['excluded_camera_ids'])
    for table in (support,observed,verified):
        for camera in torch.unique(table[table>=0]).tolist():
            record=cameras.get(int(camera))
            if record is None or record['sequence_id']!=canonical or camera in excluded:
                raise ValueError('Noncanonical, missing or excluded camera authority')
    if not (observed[:,0]>=0).all() or not torch.equal(observed[:,0],support[:,0]):
        raise ValueError('Every fresh leaf requires its actual source camera')
    uv=payload['observation_uv'][:,0]; depth=payload['observation_depth'][:,0]
    if not torch.isfinite(uv).all() or not ((uv>=0)&(uv<=1)).all() or not torch.isfinite(depth).all() or not (depth>.2).all():
        raise ValueError('Measured source bearings/depths must be valid')
    sorted_ids=verified.sort(dim=1).values
    first=torch.ones_like(sorted_ids,dtype=torch.bool)
    first[:,1:]=sorted_ids[:,1:]!=sorted_ids[:,:-1]
    distinct=(first&(sorted_ids>=0)).sum(1)
    required=torch.where(states==VERIFICATION_VERIFIED,2,1)
    if not (distinct>=required).all():
        raise ValueError('Native verification lacks independent camera witnesses')
    profiles=audit['depth_profiles_by_image']
    expected={str(v['image_name']).rsplit('.',1)[0] for v in fixed_cameras if v['sequence_id']==canonical and int(v['image_id']) not in excluded}
    if set(profiles)!=expected:
        raise ValueError('Incomplete canonical per-view depth calibration')
    for scale in profiles.values():
        if not math.isfinite(float(scale)) or float(scale)<=0:
            raise ValueError('Invalid per-view depth calibration')
    return {
        'contract':CONTRACT, 'canonical_sequence_policy':'scene',
        'canonical_scene_sequence':canonical, 'input_dynamic_rows':0,
        'associated_dynamic_rows':0, 'output_static_rows':count,
        'fused_static_leaf_clusters':count,
        'minimum_supporting_views':2, 'minimum_supporting_sequences':1,
        'camera_sequence_metadata_source':'fixed_runtime_cameras',
        'camera_sequence_metadata_count':len(cameras),
        'fresh_leaf_authority_preserved':True,
    }
