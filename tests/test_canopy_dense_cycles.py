import numpy as np
from scripts.audit_canopy_dense_cycles import pair_matches, project


def test_reverse_depth_check_rejects_wrong_surface_at_same_pixel():
    camera=dict(position=[0,0,0],rotation=np.eye(3),fx=100,fy=100,cx=50,cy=50)
    source=dict(uv=np.array([[50,50],[60,50]]),xyz=np.array([[0,0,2],[.2,0,2]]))
    target=dict(uv=np.array([[50,50],[60,50]]),depth=np.array([3.,2.]))
    assert pair_matches(source,target,camera).tolist()==[-1,1]


def test_rotated_camera_projection_roundtrip():
    rotation=np.array([[0,0,1],[0,1,0],[-1,0,0.]])
    camera=dict(position=[2,3,4],rotation=rotation,fx=100,fy=80,cx=50,cy=40)
    local=np.array([[.2,.1,2.]])
    world=local@rotation.T+camera['position']
    uv,z=project(world,camera)
    np.testing.assert_allclose(uv,[[60,44]])
    np.testing.assert_allclose(z,[2])
