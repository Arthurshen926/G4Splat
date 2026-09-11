import torch
import pytest
import copy
from scripts.canopy_candidate_optical_coordinate import CandidateOpticalCoordinate


def test_tau_gradient_matches_direct_nonnegative_radiance_derivative():
    logits=torch.nn.Parameter(torch.tensor([-6.,-2.,1.],dtype=torch.float64))
    coord=CandidateOpticalCoordinate(logits)
    kernel=torch.tensor([.1,.5,.9],dtype=torch.float64)
    loss=torch.exp(-torch.nn.functional.softplus(logits)*kernel).sum()
    loss.backward();coord.transfer_gradient()
    expected=-kernel*torch.exp(-coord.optical_depth.detach()*kernel)
    torch.testing.assert_close(coord.optical_depth.grad,expected)


def test_projected_tau_can_recover_after_reaching_its_shared_lower_bound():
    logits=torch.nn.Parameter(torch.tensor([-6.]))
    coord=CandidateOpticalCoordinate(logits)
    with torch.no_grad():coord.optical_depth.fill_(-.1)
    coord.project_and_sync()
    assert logits.sigmoid().item()==pytest.approx(1e-5,rel=2e-5)
    coord.clear_gradient()
    (1-logits.sigmoid()).sum().backward();coord.transfer_gradient()
    with torch.no_grad():coord.optical_depth.add_(coord.optical_depth.grad,alpha=-.005)
    coord.project_and_sync()
    assert logits.sigmoid().item()>.004


def test_coordinate_state_requires_same_parameter_and_bounds():
    logits=torch.nn.Parameter(torch.tensor([-6.,-2.]))
    coord=CandidateOpticalCoordinate(logits);saved=coord.state_dict()
    coord.load_state_dict(saved)
    saved['optical_depth'][0]=.1
    with pytest.raises(AssertionError):coord.load_state_dict(saved)
    assert torch.isfinite(logits).all()


def test_native_peak_kernel_chain_rule():
    logits=torch.nn.Parameter(torch.tensor([-7.,-2.,1.],dtype=torch.float64))
    coord=CandidateOpticalCoordinate(logits)
    g=torch.tensor([.1,.5,.9],dtype=torch.float64)
    (1-logits.sigmoid()*g).sum().backward();coord.transfer_gradient()
    torch.testing.assert_close(coord.optical_depth.grad,-g*torch.exp(-coord.optical_depth.detach()))


def test_tau_adam_state_roundtrip_and_gradient_reset():
    logits=torch.nn.Parameter(torch.tensor([-6.,-2.]))
    coord=CandidateOpticalCoordinate(logits)
    opt=torch.optim.Adam([coord.optical_depth],lr=.01)
    (1-logits.sigmoid()).sum().backward();coord.transfer_gradient();opt.step();coord.project_and_sync()
    restored_logits=torch.nn.Parameter(logits.detach().clone())
    restored=CandidateOpticalCoordinate(restored_logits);restored.load_state_dict(coord.state_dict())
    restored_opt=torch.optim.Adam([restored.optical_depth],lr=.01);restored_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    for c,o in ((coord,opt),(restored,restored_opt)):
        o.zero_grad(set_to_none=True);c.clear_gradient();assert c.logits.grad is None
        (1-c.logits.sigmoid()).sum().backward();c.transfer_gradient();o.step();c.project_and_sync()
    torch.testing.assert_close(logits,restored_logits)
