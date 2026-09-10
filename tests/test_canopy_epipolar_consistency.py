import numpy as np
from scripts.audit_canopy_epipolar_consistency import fundamental_from_cameras,sampson_pixel_distance


def test_static_depth_changes_disparity_not_epipolar_error():
    c={'fx':100.,'fy':90.,'cx':30.,'cy':20.,'rotation':np.eye(3),'position':[0.,0.,0.]}
    target={**c,'position':[1.,0.,0.]}
    matrix=fundamental_from_cameras(c,target)
    source=np.array([[40.,25.],[50.,30.]])
    matched=source-np.array([[20.,0.],[10.,0.]])
    assert np.max(sampson_pixel_distance(source,matched,matrix))<1.e-10
    matched[:,1]+=3
    assert (sampson_pixel_distance(source,matched,matrix)>2).all()
