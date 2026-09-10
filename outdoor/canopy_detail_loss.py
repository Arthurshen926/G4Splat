"""Local structural supervision without borrowing rigid/unknown pixels."""
import torch
import torch.nn.functional as F


def native_rigid_foliage_extinction_loss(volume_alpha, rigid):
    """Remove visible leaf extinction on known rigid interiors, never gate RGB.

    Native contribution alpha already includes all nearer occluders. Foliage
    hidden behind an opaque wall therefore cannot be turned into free-space
    evidence by this loss. One-pixel semantic boundaries remain unasserted.
    """
    if volume_alpha.shape not in (rigid.shape,(1,*rigid.shape)) or rigid.ndim!=2:
        raise ValueError('Native contribution alpha and rigid image must align')
    core=F.avg_pool2d(rigid.to(volume_alpha)[None,None],3,1,1,count_include_pad=True)[0,0]>=1-1e-6
    alpha=volume_alpha.reshape_as(core).clamp(0,1-1e-5)
    return (-torch.log1p(-alpha)*core).sum()/core.sum().clamp_min(1)


def canopy_owned_color_gradient(canopy_loss,features):
    """Extract material-owned color gradients without altering other gradients."""
    gradient=None
    if canopy_loss.requires_grad:
        gradient=torch.autograd.grad(canopy_loss,features,retain_graph=True,allow_unused=True)[0]
    return torch.zeros_like(features) if gradient is None else gradient


def masked_canopy_ssim_loss(prediction,target,canopy):
    if prediction.shape!=target.shape or prediction.shape!=(3,*canopy.shape):
        raise ValueError('Aligned RGB and canopy mask required')
    core=F.avg_pool2d(canopy.to(prediction)[None,None],11,1,5,count_include_pad=True)[0,0]>=1-1.e-6
    if not core.any():return prediction.sum()*0
    coordinates=torch.arange(11,device=prediction.device,dtype=prediction.dtype)-5
    gaussian=torch.softmax(-coordinates.square()/(2*1.5**2),dim=0)
    kernel=(gaussian[:,None]*gaussian[None,:])[None,None].expand(3,1,11,11).contiguous()
    x=prediction[None];y=target.detach()[None]
    blur=lambda value:F.conv2d(value,kernel,padding=5,groups=3)
    mx=blur(x);my=blur(y)
    vx=blur(x*x)-mx.square();vy=blur(y*y)-my.square()
    delta=x-y;md=blur(delta);vd=(blur(delta.square())-md.square()).clamp_min(0)
    similarity=((1-md.square()/(mx.square()+my.square()+.01**2))*(1-vd/(vx+vy+.03**2).clamp_min(1.e-8))).mean(dim=1)[0]
    return (1-similarity[core]).mean()
