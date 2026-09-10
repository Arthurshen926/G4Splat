import pytest
import torch
import torch.nn.functional as F
from outdoor.canopy_deformation_diagnostic import BoundedCanopyCenterField
from scripts.canopy_observed_anchor_guidance import centroid_guidance_loss


def test_sparse_centroid_guide_matches_full_field_value_and_gradient():
    xyz=torch.tensor([[.2,.3,.4],[1.1,.4,.2],[.4,.2,.8]])
    field=BoundedCanopyCenterField(xyz,torch.tensor([True,False,True]),[0.,1.,2.],rank=0,radius=3.,spacing=.5)
    with torch.no_grad():field.base.fill_(.12)
    indices=torch.tensor([0,2]);targets=torch.tensor([[.3,.25,.6]])
    guide=dict(indices=indices,active_indices=torch.searchsorted(field.active,indices),
               groups=torch.tensor([0,0]),targets=targets,base_xyz=xyz[indices].clone())
    actual=centroid_guidance_loss(field,guide)
    expected=F.smooth_l1_loss((xyz[indices]+field.offsets()[indices]).mean(0,keepdim=True)/.1,targets/.1,beta=1.)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    ga=torch.autograd.grad(actual,field.base)[0]
    ge=torch.autograd.grad(expected,field.base)[0]
    torch.testing.assert_close(ga,ge,rtol=0,atol=0)
    assert not field.offsets()[1].any()


def test_satisfied_centroid_does_not_force_geometry_and_invalid_scale_rejects():
    xyz=torch.tensor([[.2,.3,.4]])
    field=BoundedCanopyCenterField(xyz,torch.tensor([True]),[0.,1.],rank=0)
    guide=dict(active_indices=torch.tensor([0]),groups=torch.tensor([0]),targets=xyz,base_xyz=xyz)
    loss=centroid_guidance_loss(field,guide)
    assert loss==0
    assert not torch.autograd.grad(loss,field.base)[0].any()
    with pytest.raises(ValueError):centroid_guidance_loss(field,guide,0.)
