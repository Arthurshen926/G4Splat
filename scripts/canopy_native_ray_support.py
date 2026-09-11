"""Sparse selected-ray support from native projected volumes, without alpha caps."""
import numpy as np
import torch
from scipy import sparse


@torch.no_grad()
def sparse_ray_support(projected, uv, surface_depth, width, height, *, minimum_kernel=1e-8):
    xy, depth, conic, radii = projected
    if uv.ndim != 2 or uv.shape[1] != 2 or surface_depth.shape != (len(uv),):
        raise ValueError('Aligned selected rays required')
    if not torch.isfinite(surface_depth).all() or not (surface_depth > 0).all():
        raise ValueError('Positive finite surface depths required')
    # getRect uses truncation toward zero, not floor, at negative coordinates.
    tile_min = torch.trunc((xy-radii[:, None])/16).long().clamp_min(0)
    tile_max = torch.trunc((xy+radii[:, None]+15)/16).long().clamp_min(0)
    grid = xy.new_tensor([(width+15)//16, (height+15)//16], dtype=torch.long)
    tile_min = torch.minimum(tile_min, grid); tile_max = torch.minimum(tile_max, grid)
    rows, cols, values, omitted = [], [], [], []
    for i, pixel in enumerate(uv):
        tile = pixel.long()//16
        eligible = (radii > 0)&(depth < surface_depth[i])&(depth > 0)
        eligible &= ((tile >= tile_min)&(tile < tile_max)).all(1)
        ids = eligible.nonzero().flatten(); delta = xy[ids]-pixel
        power = -.5*(conic[ids, 0]*delta[:, 0].square()+conic[ids, 2]*delta[:, 1].square())-conic[ids, 1]*delta.prod(1)
        G = torch.exp(power); keep = (power <= 0)&(G >= minimum_kernel)
        omitted.append(float(G[~keep].sum()))
        kept = ids[keep].cpu().numpy()
        rows.append(np.full(len(kept), i)); cols.append(kept); values.append(G[keep].cpu().numpy())
    K = sparse.csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(len(uv), len(xy)))
    return K, np.asarray(omitted)
