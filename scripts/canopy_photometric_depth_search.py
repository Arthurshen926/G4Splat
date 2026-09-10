"""Training-view photometric proposal scores, not depth truth or verification."""
import torch
import math
import torch.nn.functional as F


def sample_pixels(image, xy):
    height, width = image.shape[-2:]
    grid = torch.stack((2*xy[..., 0]/(width-1)-1, 2*xy[..., 1]/(height-1)-1), -1)
    values = F.grid_sample(image[None], grid.reshape(1, -1, 1, 2),
                           align_corners=True, padding_mode='zeros')[0, :, :, 0]
    return values.T.reshape(*xy.shape[:-1], image.shape[0])


@torch.no_grad()
def photometric_depth_cost(source, uv, depths, source_rgb, neighbors):
    """N rays x K depth hypotheses, evaluated against >=2 training neighbors.

    neighbors contains (camera, RGB CHW, canopy bool HW). Camera matrices use
    the existing row-vector exact-K convention. Invalid/occluded-by-semantics
    hypotheses are left unsupported, not promoted to negative geometry evidence.
    The two best valid neighbors reduce contamination from unmatched views.
    """
    if uv.shape != (len(depths), 2) or depths.ndim != 2 or len(neighbors) < 2:
        raise ValueError('Aligned rays, depth hypotheses and at least two neighbors required')
    if not torch.isfinite(depths).all() or (depths <= 0).any():
        raise ValueError('Finite positive search depths required')
    offsets = uv.new_tensor([[-2, -2], [0, -2], [2, -2], [-2, 0], [0, 0], [2, 0],
                             [-2, 2], [0, 2], [2, 2]])
    pixels = uv[:, None]+offsets[None]
    source_height, source_width = source_rgb.shape[-2:]
    source_valid = ((pixels[..., 0] >= 0)&(pixels[..., 0] <= source_width-1)
                    &(pixels[..., 1] >= 0)&(pixels[..., 1] <= source_height-1)).all(-1)
    source_patch = sample_pixels(source_rgb, pixels)
    mean = source_patch.mean(1, keepdim=True)
    normalized = (source_patch-mean)/(source_patch.std(1, keepdim=True, unbiased=False)+.05)
    textured = source_patch.std(1, unbiased=False).mean(1) > .02
    rays = torch.stack(((pixels[..., 0]-source.cx)/source.focal_x,
                        (pixels[..., 1]-source.cy)/source.focal_y,
                        torch.ones_like(pixels[..., 0])), -1)
    world = (rays[:, None]*depths[:, :, None, None])@source.world_view_transform[:3, :3].T+source.camera_center
    costs = []
    for camera, rgb, canopy in neighbors:
        points = (world-camera.camera_center)@camera.world_view_transform[:3, :3]
        z = points[..., 2]
        xy = torch.stack((points[..., 0]/z.clamp_min(1e-6)*camera.focal_x+camera.cx,
                          points[..., 1]/z.clamp_min(1e-6)*camera.focal_y+camera.cy), -1)
        height, width = rgb.shape[-2:]
        valid = ((z > 1e-4)&(xy[..., 0] >= 0)&(xy[..., 0] <= width-1)
                 &(xy[..., 1] >= 0)&(xy[..., 1] <= height-1)).all(-1)
        valid &= sample_pixels(canopy[None].to(rgb.dtype), xy[:, :, 4])[..., 0] > .999
        # No depth evidence from flat patches or effectively identical rays.
        parallax = (xy[:, :, 4].amax(1)-xy[:, :, 4].amin(1)).norm(dim=-1) > .5
        valid &= source_valid[:, None]&textured[:, None]&parallax[:, None]
        patch = sample_pixels(rgb, xy)
        patch_mean = patch.mean(2, keepdim=True)
        patch_normalized = (patch-patch_mean)/(patch.std(2, keepdim=True, unbiased=False)+.05)
        error = (patch_normalized-normalized[:, None]).abs().mean((2, 3))
        error += .25*(patch_mean-mean[:, None]).abs().mean((2, 3))
        costs.append(error.masked_fill(~valid, float('inf')))
    ordered = torch.stack(costs).sort(0).values
    return ordered[:2].mean(0), torch.isfinite(ordered).sum(0)


@torch.no_grad()
def select_depth_hypotheses(depths, costs, fallback):
    """Choose one distinct photometric minimum per stratum, else keep fallback."""
    if depths.shape != costs.shape or depths.ndim != 3 or depths.shape[1] != 3 or fallback.shape != depths.shape[:2]:
        raise ValueError('Three explicit depth strata and aligned fallback required')
    ordered, indices = costs.sort(-1)
    accepted = (torch.isfinite(ordered[..., :2]).all(-1)&(ordered[..., 0] < 1.)
                &((ordered[..., 1]-ordered[..., 0]) > .01))
    selected = depths.gather(-1, indices[..., :1]).squeeze(-1)
    return torch.where(accepted, selected, fallback), accepted


@torch.no_grad()
def refine_stratified_proposals(source, uv, prior_depth, source_rgb, neighbors, proposals):
    xyz, scales, colors = proposals
    ratios = (math.log(.4)+(math.log(1.2)-math.log(.4))*(torch.arange(24,
        device=prior_depth.device, dtype=prior_depth.dtype)+.5)/24).exp()
    depths = prior_depth[:, None]*ratios
    costs, support = photometric_depth_cost(source, uv, depths, source_rgb, neighbors)
    fallback = ((xyz-source.camera_center)@source.world_view_transform[:3, :3])[:, 2].reshape(-1, 3)
    selected, accepted = select_depth_hypotheses(depths.reshape(-1, 3, 8),
        costs.reshape(-1, 3, 8), fallback)
    factor = (selected/fallback).reshape(-1, 1)
    moved = (xyz-source.camera_center)*factor+source.camera_center
    refined = (torch.where(accepted.reshape(-1, 1), moved, xyz),
               torch.where(accepted.reshape(-1, 1), scales*factor, scales), colors)
    return refined, dict(accepted_depth_slots=int(accepted.sum()), total_depth_slots=accepted.numel(),
                         hypotheses_with_two_neighbors=int((support >= 2).sum()), geometry_truth=False)
