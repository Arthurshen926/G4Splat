import torch
from outdoor.canopy_detail_loss import masked_canopy_ssim_loss,canopy_owned_color_gradient


def test_canopy_structural_factor_has_no_rigid_or_unknown_pixel_gradient():
    pred=torch.rand(3,24,40,requires_grad=True)
    target=torch.rand_like(pred)
    mask=torch.zeros(24,40,dtype=torch.bool);mask[:,:20]=True
    loss=masked_canopy_ssim_loss(pred,target,mask)
    loss.backward()
    assert pred.grad[:,mask].abs().sum()>0
    assert torch.count_nonzero(pred.grad[:,~mask])==0


def test_identical_texture_has_zero_structural_loss():
    image=torch.rand(3,24,24)
    assert abs(float(masked_canopy_ssim_loss(image,image,torch.ones(24,24,dtype=torch.bool))))<1.e-6


def test_thin_or_empty_region_does_not_borrow_a_background_window():
    pred=torch.rand(3,24,24,requires_grad=True);target=torch.rand_like(pred)
    mask=torch.zeros(24,24,dtype=torch.bool);mask[:,8:12]=True
    loss=masked_canopy_ssim_loss(pred,target,mask);loss.backward()
    assert float(loss)==0
    assert torch.count_nonzero(pred.grad)==0


def test_rigid_color_camouflage_is_blocked_but_optical_negative_gradient_remains():
    color=torch.tensor([.4,.3],requires_grad=True)
    opacity=torch.tensor([.7,.5],requires_grad=True)
    prediction=color*opacity
    canopy=(prediction[0]-.2).square()
    rigid=10*(prediction[1]-.8).square()
    expected_optical=torch.autograd.grad(canopy+rigid,opacity,retain_graph=True)[0]
    owned=canopy_owned_color_gradient(canopy,color)
    (canopy+rigid).backward();color.grad=owned
    assert color.grad[0]!=0 and color.grad[1]==0
    torch.testing.assert_close(opacity.grad,expected_optical,rtol=0,atol=0)
    assert opacity.grad[1]!=0
