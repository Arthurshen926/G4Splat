import torch
import pytest
from outdoor.canopy_depth_calibration import fit_rigid_depth_scale
from outdoor.canopy_depth_calibration import calibrated_canopy_evidence


def test_visible_rigid_calibration_ignores_unobserved_and_opaque_outliers():
    depth=torch.ones(40,40)*5;wall=depth*1.2
    wall[:4]=100
    known=torch.ones_like(depth,dtype=torch.bool);known[:4]=False
    audit=fit_rigid_depth_scale(depth,wall,torch.ones_like(depth),known)
    assert audit['applied'] and abs(audit['scale']-1.2)<1e-6


def test_missing_rigid_evidence_does_not_invent_scale():
    depth=torch.ones(40,40)
    audit=fit_rigid_depth_scale(depth,depth*2,torch.zeros_like(depth),torch.ones_like(depth,dtype=torch.bool))
    assert not audit['applied'] and audit['scale']==1.


def test_consistent_metric_ratio_is_not_rejected_by_arbitrary_global_guess_bound():
    depth=torch.ones(40,40)*5
    audit=fit_rigid_depth_scale(depth,depth*2.7,torch.ones_like(depth),torch.ones_like(depth,dtype=torch.bool))
    assert audit['applied'] and abs(audit['scale']-2.7)<1e-5


@pytest.mark.parametrize('scale', [0., -1., float('nan'), float('inf')])
def test_invalid_canopy_scale_is_rejected(scale):
    with pytest.raises(ValueError):
        calibrated_canopy_evidence({'moge3_depth_m': torch.ones(1,2,2)}, scale)


@pytest.mark.parametrize('scale', [.4, 1., 2.7])
def test_canopy_calibration_keeps_rigid_target_and_metric_thickness_cap(scale):
    from scripts.train_unified_outdoor_teacher import _moge3_depth_query_bounds
    depth=torch.full((1,4,4), 100.)
    evidence={'moge3_depth_m':depth,
              'moge3_valid_mask':torch.ones_like(depth),
              'moge3_refinement_log_depth_std':torch.ones_like(depth)*.1,
              'moge3_refinement_final_delta_log_depth':torch.zeros_like(depth)}
    calibrated=calibrated_canopy_evidence(evidence,scale)
    assert calibrated['moge3_depth_m'] is depth
    assert 'moge3_canopy_depth_m' not in evidence
    _,bounds,valid,_,audit=_moge3_depth_query_bounds(calibrated,global_log_scale_sigma=.1)
    assert valid.all()
    assert (bounds[1]-bounds[0]).max() <= .8001
    assert torch.allclose(bounds.mean(0),depth[0]*scale,atol=1e-4)
    assert audit['mean_hit_metric_half_width'] <= .40001
    with pytest.raises(ValueError,match='twice'):
        calibrated_canopy_evidence(calibrated,scale)


def test_identity_canopy_calibration_has_exact_legacy_bounds():
    from scripts.train_unified_outdoor_teacher import _moge3_depth_query_bounds
    depth=torch.arange(1.,17.).reshape(1,4,4)
    evidence={'moge3_depth_m':depth,
              'moge3_valid_mask':torch.ones_like(depth),
              'moge3_refinement_log_depth_std':torch.ones_like(depth)*.04,
              'moge3_refinement_final_delta_log_depth':torch.zeros_like(depth)}
    a=_moge3_depth_query_bounds(evidence,global_log_scale_sigma=.1)
    b=_moge3_depth_query_bounds(calibrated_canopy_evidence(evidence,1.),global_log_scale_sigma=.1)
    for x,y in zip(a[:4],b[:4]): assert torch.equal(x,y)
