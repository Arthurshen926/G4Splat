from types import SimpleNamespace
import pytest
import torch
from scripts.canopy_candidate_diagnostic import ray_candidates
from scripts.canopy_native_pixel_camera import NativePixelCamera


def test_candidate_seed_roundtrip_matches_native_viewport_not_corner_coordinates():
    camera=SimpleNamespace(cx=34.5,cy=20.,focal_x=53.,focal_y=41.,
        world_view_transform=torch.eye(4),camera_center=torch.zeros(3))
    uv=torch.tensor([[12.,14.],[37.,30.]])
    def projected_pixels(view):
        xyz,_,_=ray_candidates(view,uv,torch.tensor([2.,3.]),torch.zeros(2,3),[1.])
        # Unchanged exact-K projection followed by native ndc2Pix.
        return torch.stack((xyz[:,0]/xyz[:,2]*camera.focal_x+camera.cx-.5,
                            xyz[:,1]/xyz[:,2]*camera.focal_y+camera.cy-.5),1)
    torch.testing.assert_close(projected_pixels(camera),uv-.5,atol=3e-6,rtol=0)
    adjusted=NativePixelCamera(camera)
    torch.testing.assert_close(projected_pixels(adjusted),uv,atol=3e-6,rtol=0)
    assert camera.cx==34.5 and camera.cy==20.
    with pytest.raises(ValueError):NativePixelCamera(adjusted)
