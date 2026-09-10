import numpy as np
from scripts.audit_nvm_canopy_tracks import triangulate_training_observations


def test_training_pixels_determine_point_without_original_nvm_position():
    xyz=np.array([.4,-.2,5.])
    cameras=[dict(rotation=np.eye(3).tolist(),position=[x,0,0],fx=500.,fy=490.,cx=319.5,cy=179.5)
             for x in (-1.,0.,1.)]
    pixels=[]
    for c in cameras:
        q=xyz-np.array(c['position'])
        pixels.append([q[0]/q[2]*c['fx']+c['cx'],q[1]/q[2]*c['fy']+c['cy']])
    np.testing.assert_allclose(triangulate_training_observations(cameras,pixels),xyz,atol=1.e-8)


def test_parallel_training_rays_do_not_supply_depth():
    cameras=[dict(rotation=np.eye(3).tolist(),position=[x,0,0],fx=500.,fy=500.,cx=320.,cy=180.)
             for x in (-1.,0.,1.)]
    assert triangulate_training_observations(cameras,[[320.,180.]]*3) is None


def test_retriangulation_uses_exact_rotated_camera_convention():
    xyz=np.array([.4,.3,6.]);cameras=[];pixels=[]
    for x,angle in [(-1.,-.12),(0.,.03),(1.,.14)]:
        co,si=np.cos(angle),np.sin(angle)
        rotation=np.array([[co,0,si],[0,1,0],[-si,0,co]])
        c=dict(rotation=rotation.tolist(),position=[x,.1,0],fx=470.,fy=490.,cx=310.2,cy=175.7)
        q=(xyz-np.array(c['position']))@rotation
        cameras.append(c);pixels.append([q[0]/q[2]*c['fx']+c['cx'],q[1]/q[2]*c['fy']+c['cy']])
    np.testing.assert_allclose(triangulate_training_observations(cameras,pixels),xyz,atol=1.e-8)
