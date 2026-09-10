import torch
from scripts.train_unified_outdoor_teacher import (_static_volume_native_color_inputs,_static_volume_intrinsic_color_inputs,
    _masked_ssim_loss,_oriented_multiscale_high_frequency_loss)


def test_correct_translucent_composite_must_not_be_recolored_toward_background():
    leaf=torch.full((3,1,1),.2,requires_grad=True)
    alpha=torch.full((1,1,1),.5);background=torch.full((3,1,1),.8)
    target=torch.full((3,1,1),.5)
    prediction=leaf*alpha+background*(1-alpha)
    native,weight=_static_volume_native_color_inputs(prediction,alpha,torch.ones(1,1))
    ((native-target).square()*weight).sum().backward()
    assert torch.equal(leaf.grad,torch.zeros_like(leaf))
    # The removed production coupling creates a false brightening request
    # despite physically exact native composition.
    leaf.grad=None
    intrinsic,_=_static_volume_intrinsic_color_inputs(leaf*alpha+(1-alpha),alpha,torch.ones(3),torch.ones(1,1))
    (intrinsic-target).square().sum().backward()
    assert (leaf.grad<0).all()


def test_hidden_leaf_has_no_color_permission_and_support_cannot_create_opacity_gradient():
    prediction=torch.full((3,1,2),.3,requires_grad=True)
    alpha=torch.tensor([[[0.,.005]]],requires_grad=True)
    native,weight=_static_volume_native_color_inputs(prediction,alpha,torch.ones(1,2))
    (native*weight).sum().backward()
    assert not prediction.grad[:,:,0].any()
    assert (prediction.grad[:,:,1]==1).all() and alpha.grad is None


def test_exact_rgb_match_has_exact_zero_ssim_gradient_even_with_small_adam_epsilon():
    torch.manual_seed(51)
    image=torch.nn.Parameter(torch.rand(3,32,40));original=image.detach().clone()
    loss=_masked_ssim_loss(image,original,torch.ones(32,40))
    loss.backward()
    assert loss.item()==0 and torch.count_nonzero(image.grad)==0
    optimizer=torch.optim.Adam([image],lr=.0025,eps=1.e-15);optimizer.step()
    assert torch.equal(image,original)


def test_multiscale_sobel_cannot_leak_leaf_color_gradients_into_rigid_pixels():
    torch.manual_seed(52)
    image=torch.rand(3,32,40,requires_grad=True);target=torch.rand_like(image)
    weight=torch.zeros(32,40);weight[4:26,8:29]=1
    loss=_oriented_multiscale_high_frequency_loss(image,target,weight);loss.backward()
    assert image.grad[:,weight.bool()].abs().sum()>0
    assert torch.count_nonzero(image.grad[:,~weight.bool()])==0


def test_stable_ssim_preserves_the_mathematical_objective_away_from_equality():
    import torch.nn.functional as F
    torch.manual_seed(53)
    x=torch.rand(3,24,28,dtype=torch.float64);y=torch.rand_like(x)
    blur=lambda z:F.avg_pool2d(z[None],11,stride=1,padding=5)[0]
    mx,my=blur(x),blur(y);vx,vy=blur(x*x)-mx*mx,blur(y*y)-my*my
    covariance=blur(x*y)-mx*my
    similarity=((2*mx*my+.01**2)*(2*covariance+.03**2)/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))).mean(0)
    expected=(1-similarity[5:-5,5:-5]).mean()
    actual=_masked_ssim_loss(x,y,torch.ones(24,28,dtype=torch.float64))
    torch.testing.assert_close(actual,expected,atol=1.e-12,rtol=1.e-12)
