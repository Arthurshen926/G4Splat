import numpy as np
import pytest
from scripts.audit_rigid_track_depth import independent_track_mask
from scripts.audit_rigid_track_depth import cycle_archive
import torch
from types import SimpleNamespace


def test_distinct_cameras_and_whole_track_validation_exclusion():
    offsets=np.arange(0,13,3)
    cameras=np.array([0,1,2,0,0,1,0,1,9,0,1,2])
    keep=independent_track_mask(offsets,cameras,[0,1,2],np.array([.2]*4),
                                np.array([2.]*4),np.array([1,1,1,0]))
    assert keep.tolist()==[True,False,False,False]


def test_invalid_offsets_fail_closed():
    with pytest.raises(ValueError):
        independent_track_mask([0,4],[0,1,2],[0,1,2],[.1],[2],[1])


def test_pairs_are_not_silently_promoted_to_cycle_evidence():
    args=([0,2],[0,1],[0,1],[.2],[2.],np.array([0]))
    assert not independent_track_mask(*args).any()
    assert independent_track_mask(*args,minimum_cameras=2,require_cycle=False).tolist()==[True]


def test_cycle_native_reprojection_and_positive_information(tmp_path):
    views=[];pixels=[]
    points=np.array([[0.,0.,10.],[.3,.2,11.]])
    for i,center in enumerate((-1.,0.,1.)):
        transform=torch.eye(4);transform[3,0]=-center
        views.append(SimpleNamespace(world_view_transform=transform,
            camera_center=torch.tensor([center,0.,0.]),focal_x=500.,focal_y=500.,cx=320.,cy=180.,
            image_width=640,image_height=360,image_name=f'seq2__frame{i:05d}'))
        pixels.append(np.stack((500*(points[:,0]-center)/points[:,2]+320,
                                500*points[:,1]/points[:,2]+180),1))
    # The archive's old geometric flag cannot hide a new native mismatch.
    pixels[2][1,0]+=4
    np.savez(tmp_path/'tracks_0_1_2.npz',points=points,left=pixels[0],middle=pixels[1],right=pixels[2],
             geometric=np.ones(2,dtype=bool),third_error=np.zeros(2),labels=np.ones(2,dtype=int))
    data,sources=cycle_archive(tmp_path,views)
    assert data['xyz'].shape==(1,3)
    assert np.all(data['position_covariance_diag']>0)
    assert data['observation_offsets'].tolist()==[0,3]
    assert len(sources)==1
def test_original_observations_do_not_invent_descriptor_cycles(tmp_path):
    import json
    from types import SimpleNamespace
    from scripts.audit_rigid_track_depth import nvm_archive, FIXED, ADDITIONAL
    views=[SimpleNamespace(image_name=f'seq2__frame{i:05}',image_width=640,image_height=360) for i in range(3)]
    record=dict(xyz=[0,0,5],angle_p10=2.,covariance=np.eye(3).tolist(),
                observations=[dict(index=i,image_name=v.image_name,uv=[320,180],reprojection=.1) for i,v in enumerate(views)])
    (tmp_path/'audit.json').write_text(json.dumps(dict(
        scope='original_observed_nvm_canopy_tracks__read_only_not_coverage_prior',
        excluded_views=FIXED+ADDITIONAL,records=[record])))
    archive=nvm_archive(tmp_path,views)
    assert not archive['descriptor_cycle_rank'].any()
    args=(archive['observation_offsets'],archive['observation_camera_indices'],[0,1,2],
          archive['reprojection_error'],archive['triangulation_angle_p10'],archive['descriptor_cycle_rank'])
    assert not independent_track_mask(*args).any()
    assert independent_track_mask(*args,require_cycle=False).all()
