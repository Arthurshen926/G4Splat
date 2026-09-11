import numpy as np
import torch
import pytest
from types import SimpleNamespace
from scripts.canopy_moge_position_supervision import NativeDepthViews,project_depth_pixels


def test_native_projection_preserves_subpixel_ray():
    v=SimpleNamespace(focal_x=10.,focal_y=10.,cx=4.,cy=4.,image_width=8,image_height=8)
    camera=torch.tensor([[.04,0.,1.]])
    x,y=project_depth_pixels(camera,v,(24,24))
    assert x.item()==13 and y.item()==12
    low,_=project_depth_pixels(camera,v,(8,8))
    assert x.item()!=low.item()*3


def test_native_depth_does_not_mix_layers(tmp_path):
    depth=np.array([[1.,10.,1.],[10.,1.,10.]],dtype=np.float32)
    valid=np.ones_like(depth,dtype=bool);valid[0,0]=False
    np.savez(tmp_path/'view.npz',depth_m=depth,valid_mask=valid)
    result=NativeDepthViews(tmp_path,['view'])[0]
    assert result.shape==(2,3) and np.isnan(result[0,0])
    assert set(result[np.isfinite(result)])=={1.,10.}


def test_native_depth_rejects_changed_source_on_refresh(tmp_path):
    path=tmp_path/'view.npz'
    np.savez(path,depth_m=np.ones((2,2),dtype=np.float32),valid_mask=np.ones((2,2),dtype=bool))
    source=NativeDepthViews(tmp_path,['view']);source[0]
    np.savez(path,depth_m=np.full((2,2),2.,dtype=np.float32),valid_mask=np.ones((2,2),dtype=bool))
    with pytest.raises(ValueError,match='changed'):source[0]
