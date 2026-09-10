from types import SimpleNamespace
import torch
from outdoor.canopy_patch_stereo import candidate_patch_scores,normalized_patch_correlation,unambiguous_depth_matches


def test_patch_correlation_rejects_flat_and_accepts_affine_brightness():
    torch.manual_seed(17)
    texture=torch.rand(5,25,3)
    score=normalized_patch_correlation(texture,texture*.6+.1)
    torch.testing.assert_close(score,torch.ones_like(score))
    assert (normalized_patch_correlation(torch.ones_like(texture),texture)==-1).all()


def test_patch_depth_requires_a_unique_peak_not_uniform_dark_matches():
    scores=torch.tensor([[.8,.81,.8,.82,.8,.8,.8],[-1.,.1,.5,.9,.89,.3,.2]])
    valid=unambiguous_depth_matches(scores)
    assert not valid[0].any()
    assert valid[1].tolist()==[False,False,False,True,True,False,False]


def test_physical_peak_support_does_not_shrink_with_finer_depth_grid():
    def support(step):
        x=torch.arange(-.3,.30001,step)
        scores=(.95-20*x.square())[None]
        radius=round(.075/step)
        accepted=unambiguous_depth_matches(scores,separation_bins=radius,peak_support_bins=radius)[0]
        return x[accepted].abs().max()
    assert abs(float(support(.005)-support(.01)))<=.0051


def test_extended_peak_support_still_rejects_ambiguity_and_bad_scores():
    flat=torch.ones(2,61)*.9
    assert not unambiguous_depth_matches(flat,separation_bins=15,peak_support_bins=15).any()
    low=flat*.5
    low[:,30]=.64
    assert not unambiguous_depth_matches(low,separation_bins=15,peak_support_bins=15).any()


def test_exact_identity_warp_preserves_patch_but_has_no_depth_authority():
    torch.manual_seed(23)
    view=SimpleNamespace(original_image=torch.rand(3,32,40),image_width=40,image_height=32,
        focal_x=30.,focal_y=29.,cx=19.1,cy=15.3,world_view_transform=torch.eye(4),camera_center=torch.zeros(3))
    scores=candidate_patch_scores(view,view,torch.tensor([[12.,13.],[25.,18.]]),torch.tensor([3.,5.]),torch.linspace(-.3,.3,17))
    torch.testing.assert_close(scores,torch.ones_like(scores),atol=1e-5,rtol=1e-5)
    assert not unambiguous_depth_matches(scores).any()


def test_translated_camera_recovers_true_plane_depth_without_moge_target():
    torch.manual_seed(29)
    source=SimpleNamespace(original_image=torch.rand(3,40,64),image_width=64,image_height=40,
        focal_x=24.,focal_y=24.,cx=31.2,cy=19.4,world_view_transform=torch.eye(4),camera_center=torch.zeros(3))
    transform=torch.eye(4);transform[3,0]=-1.
    target=SimpleNamespace(**{**vars(source),'original_image':source.original_image.roll(-8,dims=2),
        'world_view_transform':transform,'camera_center':torch.tensor([1.,0.,0.])})
    scores=candidate_patch_scores(source,target,torch.tensor([[28.,19.],[40.,25.]]),torch.tensor([3.,3.]),torch.linspace(-.3,.3,17))
    assert scores.argmax(dim=1).tolist()==[8,8]
    assert unambiguous_depth_matches(scores)[:,8].all()
