import numpy as np
from outdoor.native_observation_matching import match_patch,triangulate_rays
from outdoor.native_observation_matching import triangulate_consensus
from outdoor.native_observation_matching import epipolar_pixel_line
from outdoor.native_observation_matching import rebind_angular_sigma
import pytest


def test_subpixel_matching_recovers_fractional_translation():
    from scipy.ndimage import gaussian_filter,shift
    source=gaussian_filter(np.random.default_rng(9).random((64,64,3)),(.8,.8,0))
    target=shift(source,(.5,2.5,0),order=1)
    integer=match_patch(source,[30,30],target,[32,31])
    fractional=match_patch(source,[30,30],target,[32,31],subpixel=True)
    assert integer is not None and fractional is not None
    np.testing.assert_allclose(fractional[0],[32.5,30.5],atol=.25)
    assert fractional[1]<integer[1]


def test_retriangulation_preserves_source_angular_footprint():
    sigma=np.array([.01,.03]);old=np.array([2.,5.]);new=np.array([4.,1.])
    rebound=rebind_angular_sigma(sigma,old,new)
    np.testing.assert_allclose(rebound/new,sigma/old)
    with pytest.raises(ValueError):rebind_angular_sigma(sigma,old,np.array([-1.,2.]))


def test_epipolar_line_contains_projections_with_rotated_camera():
    a=.3;rotation=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    source=np.array([.1,.2,0]);target=np.array([1.,.1,.2]);direction=np.array([.2,.3,1.])
    line=epipolar_pixel_line(source,direction,target,rotation,700,710,959.5,539.5)
    for depth in [3.,5.,8.]:
        q=(source+depth*direction-target)@rotation
        pixel=np.array([q[0]/q[2]*700+959.5,q[1]/q[2]*710+539.5,1.])
        assert abs(line@pixel)<1e-10


def test_consensus_rejects_outlier_without_exporting_stale_support():
    centers=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[-1.,0.,0.]])
    point=np.array([.2,.3,5.]);directions=point-centers
    directions[-1]=np.array([2.,2.,3.])
    observed=directions[:,:2]/directions[:,2:]*1000
    def errors(p):
        q=p-centers
        return np.linalg.norm(q[:,:2]/q[:,2:]*1000-observed,axis=1)
    fit=triangulate_consensus(centers,directions,errors)
    assert fit is not None
    np.testing.assert_allclose(fit[0],point,atol=1e-10)
    np.testing.assert_array_equal(fit[1],[True,True,True,False])
    assert errors(triangulate_rays(centers,directions)).max()>1.5
    assert triangulate_consensus(centers,directions,lambda p:np.full(4,np.inf)) is None


def test_triangulation_uses_observed_rays_not_a_depth_prior():
    centers=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    point=np.array([.2,.3,5.])
    np.testing.assert_allclose(triangulate_rays(centers,point-centers),point,atol=1e-10)
    assert triangulate_rays(centers,np.tile([0.,0.,1.],(3,1))) is None


def test_rgb_matching_recovers_shift_and_rejects_flat_patches():
    source=np.random.default_rng(14).random((40,40,3)).astype(np.float32)
    target=np.zeros_like(source);target[2:,3:]=source[:-2,:-3]
    found=match_patch(source,[20,20],target,[22,21])
    assert found is not None
    np.testing.assert_array_equal(found[0],[23,22])
    assert match_patch(np.zeros_like(source),[20,20],target,[22,21]) is None
    assert match_patch(source,[20,20],target,[22,21],epipolar_line=[0.,1.,-22.]) is not None
    assert match_patch(source,[20,20],target,[22,21],epipolar_line=[0.,0.,0.]) is None
    found=match_patch(source,[20,20],target,[22,21],epipolar_line=[0.,1.,-18.])
    assert found is None or abs(found[0][1]-18)<=1.5
