"""Measurement-position uncertainty, independent of optical kernel thickness."""
import math
import torch

CONTRACT='exact_k_pixel_and_log_depth_position_covariance__independent_of_optical_shape_v1'


def camera_ray_position_covariance(uv,depth,log_depth_sigma,camera_to_world,*,fx,fy,cx,cy,pixel_sigma=2.):
    """First-order covariance of exact-K unprojection in world coordinates.

    Two-pixel source localization uncertainty is an explicit conservative
    sampling assumption, not leaf radius. Depth uncertainty moves one surface
    hypothesis along its ray; it never increases rendered material thickness.
    """
    n=len(depth)
    if uv.shape!=(n,2) or log_depth_sigma.shape!=(n,) or camera_to_world.shape!=(3,3):
        raise ValueError('Aligned source bearings, depths and rotation required')
    if not all(math.isfinite(float(v)) and float(v)>0 for v in (fx,fy,pixel_sigma)):
        raise ValueError('Positive finite focal lengths and pixel uncertainty required')
    if not (torch.isfinite(uv).all() and torch.isfinite(depth).all() and (depth>0).all()
            and torch.isfinite(log_depth_sigma).all() and (log_depth_sigma>0).all()
            and torch.isfinite(camera_to_world).all()):
        raise ValueError('Invalid measured position uncertainty')
    ray=torch.stack(((uv[:,0]-cx)/fx,(uv[:,1]-cy)/fy,torch.ones_like(depth)),1)
    sigma_z=depth*log_depth_sigma
    covariance=sigma_z[:,None,None].square()*ray[:,:,None]*ray[:,None,:]
    covariance[:,0,0]+=(depth/fx*pixel_sigma).square()
    covariance[:,1,1]+=(depth/fy*pixel_sigma).square()
    rotation=camera_to_world.to(covariance)
    covariance=rotation[None]@covariance@rotation.T[None]
    return (covariance+covariance.transpose(1,2))*.5


def position_mahalanobis_error(delta,covariance):
    """Use the full covariance: discarding correlation destroys ray direction."""
    if delta.ndim!=2 or delta.shape[1]!=3 or covariance.shape!=(len(delta),3,3):
        raise ValueError('Aligned world position error and covariance required')
    # A small physical numeric floor (0.1mm), not an optical thickness prior.
    covariance=covariance.detach()+torch.eye(3,device=delta.device,dtype=delta.dtype)[None]*1.e-8
    factor,info=torch.linalg.cholesky_ex(covariance)
    if (info!=0).any():raise ValueError('Position covariance must be positive definite')
    solution=torch.cholesky_solve(delta[:,:,None],factor)[:,:,0]
    return (delta*solution).sum(1).clamp_min(0)
