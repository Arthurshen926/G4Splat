"""Read-only depth prefixes from ORIGINAL native per-ray surface responsibility.

Depths use the same native surface evaluator, including its low-pass fallback.
This diagnoses the fixed rigid model; it does not certify physical leaf depth.
"""
import torch


def surface_api():
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/'experimental/canopy_optical_depth_rasterizer'
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    from canopy_optical_depth_rasterization import _C
    if Path(_C.__file__).resolve().parent!=root/'canopy_optical_depth_rasterization':
        raise RuntimeError('Surface diagnostic extension from another worktree')
    if not hasattr(_C,'diagnostic_surface_support') or not hasattr(_C,'diagnostic_surface_depths'):
        raise RuntimeError('Rebuild the independent surface diagnostic extension')
    return _C


def ordered_prefix(ids, depths, weights, colors):
    if (ids.ndim!=1 or depths.shape!=ids.shape or weights.shape!=ids.shape
            or colors.shape!=(len(ids),3) or ids.dtype!=torch.long
            or not torch.isfinite(depths).all() or not torch.isfinite(weights).all()
            or not torch.isfinite(colors).all() or (depths<.2).any()
            or (weights<0).any() or (colors<0).any()):
        raise ValueError('Finite aligned native depth, contribution and nonnegative color required')
    # Original mixed ties are ordered by primitive ID. Surface IDs precede
    # volume IDs, so a volume at z sees surface light at depth <= z in front.
    by_id=torch.argsort(ids,stable=True)
    order=by_id[torch.argsort(depths[by_id],stable=True)]
    return dict(ids=ids[order],depths=depths[order],weights=weights[order],
        cumulative_rgb=(weights[order,None]*colors[order]).cumsum(0))


def prefix_rgb(prefix, depth):
    query=torch.as_tensor(depth,device=prefix['depths'].device,dtype=prefix['depths'].dtype)
    count=torch.searchsorted(prefix['depths'],query.contiguous(),right=True)
    cumulative=torch.cat((prefix['cumulative_rgb'].new_zeros(1,3),prefix['cumulative_rgb']),0)
    return cumulative[count]


@torch.no_grad()
def native_prefix(projected, opacity, colors, responsibility, pixel):
    _C=surface_api()
    xy,depth,transforms,normals,radii=projected
    if responsibility.shape!=(len(xy),) or colors.shape!=(len(xy),3):
        raise ValueError('Actual pure-rigid native contribution required for each surface row')
    if not torch.isfinite(responsibility).all() or (responsibility<0).any():
        raise ValueError('Native responsibility must be finite and nonnegative')
    ids=responsibility.gt(0).nonzero().flatten()
    # Positive responsibility certifies native tile inclusion/alpha acceptance;
    # the depth helper alone does not grant visibility to arbitrary primitives.
    if (radii[ids]<=0).any():raise ValueError('Contribution from an unprojected surface')
    pixels=torch.as_tensor(pixel,device=xy.device,dtype=torch.int32).reshape(1,2).expand(len(ids),2).contiguous()
    z,valid=_C.diagnostic_surface_depths(xy,depth,transforms,normals,opacity.contiguous(),ids,pixels)
    if not valid.all():raise ValueError('Native contribution/depth evaluator mismatch')
    return ordered_prefix(ids,z,responsibility[ids],colors[ids])
