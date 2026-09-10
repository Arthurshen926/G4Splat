import numpy as np

from scripts.audit_canopy_descriptor_geometry import calibrated_projection, triangulate_matches


def test_descriptor_audit_triangulates_exact_calibrated_pixels():
    angle = .17
    rotation = [[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]]
    a = dict(fx=550.,fy=570.,cx=307.,cy=173.,rotation=np.eye(3).tolist(),position=[0.,0.,0.])
    b = dict(a,position=[.7,.1,.02],rotation=rotation)
    points = np.array([[.1,.2,4.],[-.7,.4,6.],[.8,-.3,8.]])
    def project(c):
        p=calibrated_projection(c)
        x=points@p[:,:3].T+p[:,3]
        return x[:,:2]/x[:,2:]
    recovered,valid,error,angles = triangulate_matches(project(a),project(b),a,b)
    np.testing.assert_allclose(recovered,points,atol=1.e-11)
    assert valid.all() and (error < 1.e-10).all() and (angles > .5).all()


def test_descriptor_cycles_use_pixels_not_epipolar_acceptance():
    from scripts.audit_canopy_descriptor_cycles import close_cycles
    ab={'left':np.array([[1,2],[10,20]]),'right':np.array([[3,4],[30,40]])}
    bc={'left':np.array([[3,4],[30,40]]),'right':np.array([[5,6],[50,60]])}
    ac={'left':np.array([[1,2],[10,20]]),'right':np.array([[5,6],[80,90]])}
    i,j,k=close_cycles(ab,bc,ac)
    assert i.tolist()==j.tolist()==k.tolist()==[0]


def test_empty_descriptor_geometry_is_unknown_not_failure():
    points,valid,error,angles=triangulate_matches(np.empty((0,2)),np.empty((0,2)),{}, {})
    assert points.shape==(0,3) and not len(valid) and not len(error) and not len(angles)
