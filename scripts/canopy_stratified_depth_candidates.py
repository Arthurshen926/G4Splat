"""Deterministic low-opacity search opportunities, not measured leaf depths."""
import math
import torch


def stratified_ray_candidates(view, uv, depth, colors, *, seed, strata=3, low=.4, high=1.2, pixel_sigma=1.2):
    if strata != 3 or not (0 < low < high <= 1.5) or not math.isfinite(pixel_sigma) or pixel_sigma <= 0:
        raise ValueError('Three bounded depth strata and positive angular footprint required')
    if uv.shape != (len(depth), 2) or colors.shape != (len(depth), 3) or not torch.isfinite(depth).all() or not (depth > 0).all():
        raise ValueError('Aligned finite positive ray hypotheses required')
    uniform = torch.rand((len(depth), strata), generator=torch.Generator().manual_seed(seed), dtype=depth.dtype)
    unit = (torch.arange(strata, dtype=depth.dtype)[None]+uniform)/strata
    ratios = (math.log(low)+(math.log(high)-math.log(low))*unit).exp().to(depth.device)
    z = depth[:, None]*ratios
    rays = torch.stack(((uv[:, 0]-view.cx)/view.focal_x,
                        (uv[:, 1]-view.cy)/view.focal_y, torch.ones_like(depth)), 1)
    xyz = (rays[:, None]*z[:, :, None])@view.world_view_transform[:3, :3].T+view.camera_center
    scales = z.reshape(-1, 1).expand(-1, 3)*pixel_sigma/math.sqrt(view.focal_x*view.focal_y)
    return xyz.reshape(-1, 3), scales, colors[:, None].expand(-1, strata, -1).reshape(-1, 3)
