import torch
from scripts.canopy_candidate_diagnostic import CandidateCloud
from scripts.canopy_candidate_ray_refinement import RayDepthCandidates, rigid_rgb_preservation, project_candidate_dc_


def test_ray_refinement_preserves_source_projection_and_angular_scale():
    xyz = torch.tensor([[1., 2., 10.], [3., -1., 20.]])
    cloud = CandidateCloud(xyz, torch.ones_like(xyz)*.1, torch.ones_like(xyz)*.3)
    candidate = RayDepthCandidates(cloud, torch.zeros_like(xyz))
    torch.testing.assert_close(candidate.xyz, xyz, rtol=0, atol=0)
    with torch.no_grad(): candidate.depth_code.copy_(torch.tensor([2., -2.]))
    torch.testing.assert_close(candidate.xyz[:, :2]/candidate.xyz[:, 2:], xyz[:, :2]/xyz[:, 2:])
    torch.testing.assert_close(candidate.scales/candidate.xyz[:, 2:], cloud.scales/xyz[:, 2:])
    assert (candidate.ratio.log().abs() <= .15).all()
    candidate.xyz.sum().backward()
    assert candidate.depth_code.grad.abs().sum() > 0
    assert torch.equal(cloud.xyz, xyz)
    assert cloud.logits.grad is None


def test_rigid_rgb_reference_has_no_gradient_or_canopy_authority():
    rgb = torch.ones(3, 2, 2, requires_grad=True)
    reference = torch.zeros_like(rgb, requires_grad=True)
    rigid = torch.tensor([[True, False], [False, False]])
    loss = rigid_rgb_preservation(rgb, reference, rigid); loss.backward()
    assert reference.grad is None
    assert torch.equal(rgb.grad[:, ~rigid], torch.zeros(3, 3))
    assert rgb.grad[:, rigid].sum() == 1


def test_zero_ray_code_is_exact_even_with_large_camera_translation():
    xyz = torch.tensor([[.123456, 2.123456, 10.123456]])
    cloud = CandidateCloud(xyz, torch.ones_like(xyz), torch.ones_like(xyz)*.5)
    candidate = RayDepthCandidates(cloud, torch.tensor([[1000., 50., -150.]]))
    assert torch.equal(candidate.xyz, xyz)


def test_projected_dc_recovers_from_native_dead_color():
    dc = torch.tensor([[-2., 3., 0.]], requires_grad=True)
    c0 = .28209479177387814
    (dc*c0+.5).clamp_min(0).sum().backward()
    assert dc.grad[0, 0] == 0
    dc.grad = None; project_candidate_dc_(dc)
    rgb = (dc*c0+.5).clamp_min(0)
    torch.testing.assert_close(rgb, torch.tensor([[0., 1., .5]]))
    (rgb-torch.tensor([[.2, .8, .5]])).square().sum().backward()
    assert dc.grad[0, 0] < 0  # A true recovery gradient now exists.
    assert dc.grad[0, 1] > 0
