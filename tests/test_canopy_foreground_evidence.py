import torch
from outdoor.canopy_foreground_evidence import foreground_contrast_evidence,foreground_darkening_bound,foreground_darkening_loss
from outdoor.canopy_foreground_evidence import mixed_order_darkening_bound
import pytest


def test_gap_is_unknown_not_positive_occluder_and_rigid_masks_unchanged():
    rgb=torch.ones(3,32,64)*.8;background=rgb.clone()
    canopy=torch.zeros(32,64,dtype=torch.bool);canopy[:,32:]=True
    rigid=~canopy;original=rigid.clone()
    rgb[:,10:20,40:50]=.2
    positive,profile=foreground_contrast_evidence(rgb,background,canopy,rigid,torch.ones(1,32,64))
    assert profile['applied'] and abs(profile['threshold']-.02)<1.e-6
    assert positive.sum()==100
    assert torch.equal(rigid,original)
    assert not positive[0,40]


def test_insufficient_background_anchor_preserves_uncertain_canopy():
    image=torch.ones(3,8,8)
    canopy=torch.ones(8,8,dtype=torch.bool)
    result,profile=foreground_contrast_evidence(image,image,canopy,~canopy,torch.ones(1,8,8))
    assert not profile['applied'] and torch.equal(result,canopy)


def test_persisted_calibration_is_not_refit_to_new_view_or_model():
    image=torch.ones(3,32,32);mask=torch.ones(32,32,dtype=torch.bool)
    profile={'applied':True,'threshold':.2}
    result,returned=foreground_contrast_evidence(image,image-.1,mask,~mask,torch.zeros(1,32,32),profile)
    assert not result.any() and returned is profile


def test_darkening_only_grows_admissible_deficits_not_gaps_or_satisfied_alpha():
    alpha=torch.tensor([[.1,.1,.1,.9]],requires_grad=True)
    required=torch.tensor([[.8,0.,.8,.8]],requires_grad=True)
    loss=foreground_darkening_loss(alpha,required,torch.tensor([[True,True,False,True]]))
    loss.backward()
    assert alpha.grad[0,0]<0 and torch.equal(alpha.grad[0,1:],torch.zeros(3))
    assert required.grad is None


def test_background_error_slack_weakens_bound_and_cannot_train_background_colors():
    b=torch.ones(3,2,2,requires_grad=True);c=torch.ones(3,2,2)*.4
    low=foreground_darkening_bound(c,b,.02);high=foreground_darkening_bound(c,b,.2)
    assert (high<low).all() and not high.requires_grad


def test_mixed_depth_order_requires_contributor_ceiling_not_composite_denominator():
    # Near black surface alpha .5; far white background; opaque black leaf
    # inserted between them. Native leaf alpha is .5, not one.
    b=torch.full((3,1,1),.5);c=torch.zeros_like(b)
    assert foreground_darkening_bound(c,b,0).item()>.5
    assert mixed_order_darkening_bound(c,b,torch.ones(3),0).item()==.5


def test_mixed_bound_holds_for_random_multilayer_insertion_without_gradients():
    generator=torch.Generator().manual_seed(192)
    surface_alpha=torch.rand(12,1,100,generator=generator)*.8
    leaf_alpha=torch.rand(12,1,100,generator=generator)*.8
    surface_rgb=torch.rand(12,3,100,generator=generator)*2
    leaf_rgb=torch.rand(12,3,100,generator=generator)*2
    sky=torch.rand(3,100,generator=generator)*2
    t=torch.ones(1,100);tb=t.clone();a=torch.zeros_like(t)
    b=torch.zeros(3,100);c=b.clone()
    for i in range(12):
        b+=tb*surface_alpha[i]*surface_rgb[i];tb*=1-surface_alpha[i]
        c+=t*surface_alpha[i]*surface_rgb[i];t*=1-surface_alpha[i]
        a+=t*leaf_alpha[i];c+=t*leaf_alpha[i]*leaf_rgb[i];t*=1-leaf_alpha[i]
    b+=tb*sky;c+=t*sky
    ceiling=torch.cat((surface_rgb,sky[None]),0).amax(0)[:,None]
    ceiling.requires_grad_(True)
    bound=mixed_order_darkening_bound(c[:,None],b[:,None],ceiling,0)
    assert not bound.requires_grad and (bound<=a+1.e-5).all()
    for cap in (.25,.5,1.,2.):
        tb=torch.ones(1,100);clipped=torch.zeros(3,100)
        for i in range(12):
            clipped+=tb*surface_alpha[i]*surface_rgb[i].clamp_max(cap)
            tb*=1-surface_alpha[i]
        clipped+=tb*sky.clamp_max(cap)
        result=mixed_order_darkening_bound(c[:,None],clipped[:,None],torch.full((3,),cap),0)
        assert (result<=a+1.e-5).all()


def test_mixed_bound_rejects_underestimated_radiance_ceiling():
    with pytest.raises(ValueError):
        mixed_order_darkening_bound(torch.zeros(3,1,1),torch.ones(3,1,1),torch.full((3,),.5))


def test_clamping_final_background_is_not_a_valid_contributor_cap():
    # A near black alpha-.5 surface and far RGB2 surface produce B=1.
    # A black leaf between them has visible alpha .5 and produces C=0.
    c=torch.zeros(3,1,1)
    incorrectly_clamped_composite=torch.ones_like(c)
    contributor_capped_background=torch.full_like(c,.5)
    assert mixed_order_darkening_bound(c,incorrectly_clamped_composite,torch.ones(3),0).item()>.5
    assert mixed_order_darkening_bound(c,contributor_capped_background,torch.ones(3),0).item()==.5
