import torch
from scripts.audit_canopy_foreground_matting import darkening_layer_bound


def test_darkening_bound_does_not_fill_a_real_background_gap():
    background=torch.full((3,2,2),.8)
    target=background.clone();target[:,0,0]=.16
    bound=darkening_layer_bound(target,background,slack=0)
    torch.testing.assert_close(bound,torch.tensor([[.8,0.],[0.,0.]]))


def test_bright_foreground_does_not_receive_an_unjustified_darkening_bound():
    assert not darkening_layer_bound(torch.ones(3,2,2),torch.full((3,2,2),.1)).any()
