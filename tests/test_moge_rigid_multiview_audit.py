import numpy as np
from scripts.audit_moge_rigid_multiview import camera_points_to_world,world_points_to_camera,nearest_other_cameras,baseline_angle


def test_nonidentity_camera_roundtrip():
    r=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    p=np.array([[1.,2.,3.]]);center=np.array([3.,4.,5.])
    world=camera_points_to_world(p,r,center)
    assert np.allclose(world,[[1.,5.,8.]])
    assert np.allclose(world_points_to_camera(world,r,center),p)


def test_coincident_cameras_never_count_self_as_support():
    centers=np.zeros((4,3))
    neighbors=nearest_other_cameras(centers)
    assert neighbors.shape==(4,3)
    for index,row in enumerate(neighbors):
        assert index not in row
        assert len(set(row))==3
    assert nearest_other_cameras(np.zeros((1,3))).shape==(1,0)


def test_baseline_angle_rejects_coincident_views():
    p=np.array([[0.,0.,1.]])
    assert np.allclose(baseline_angle(p,[0,0,0],[0,0,0]),0)
    assert np.allclose(baseline_angle(p,[0,0,0],[1,0,0]),45)
