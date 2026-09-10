"""Read-only local plane-sweep evidence; no rendering gates or learned state.

Monocular uncertainty nominates alternative locations, not optical thickness.
Warp each source patch by its exact camera-tangent plane at each candidate
depth. Correlation is not authority when texture is flat or depth ambiguous.
"""
import torch
import torch.nn.functional as F


def normalized_patch_correlation(reference,warped,minimum_std=.02):
    """Inputs (..., patch_pixels, RGB); reject constant / nonfinite patches."""
    left=reference-reference.mean(dim=-2,keepdim=True)
    right=warped-warped.mean(dim=-2,keepdim=True)
    left_var=left.square().mean(dim=(-2,-1))
    right_var=right.square().mean(dim=(-2,-1))
    score=(left*right).mean(dim=(-2,-1))/(left_var*right_var).sqrt().clamp_min(1e-8)
    valid=(left_var>=minimum_std**2)&(right_var>=minimum_std**2)&torch.isfinite(score)
    return torch.where(valid,score.clamp(-1,1),torch.full_like(score,-1.))


def unambiguous_depth_matches(scores,threshold=.65,uniqueness=.04,separation_bins=2,peak_support_bins=1):
    """Unique peak support, not optical thickness or multiple leaf locations.

    Legacy callers retain one neighboring bin. A physical-grid caller can
    explicitly retain the score-supported part of the uniqueness neighborhood
    to avoid tightening position agreement just by refining sampling density.
    """
    if not isinstance(peak_support_bins,int) or not 0<=peak_support_bins<=separation_bins:
        raise ValueError('Peak support must remain inside the uniqueness neighborhood')
    best,index=scores.max(dim=1)
    bins=torch.arange(scores.shape[1],device=scores.device)[None]
    far=(bins-index[:,None]).abs()>separation_bins
    alternative=torch.where(far,scores,torch.full_like(scores,-1.)).max(dim=1).values
    good=(best>=threshold)&((best-alternative)>=uniqueness)
    return good[:,None]&((bins-index[:,None]).abs()<=peak_support_bins)&(scores>=best[:,None]-.04)&(scores>=threshold)


@torch.no_grad()
def candidate_patch_scores(source_view,target_view,uv,source_depth,log_shifts,*,chunk_size=256):
    """N x D photometric score. Coordinates use exact K / align_corners=False."""
    device=uv.device
    offsets=torch.stack(torch.meshgrid(torch.arange(-2,3,device=device),
                                      torch.arange(-2,3,device=device),indexing='ij'),dim=-1)
    offsets=offsets.reshape(-1,2)[:,[1,0]].float()
    source_image=source_view.original_image.to(device)[None]
    target_image=target_view.original_image.to(device)[None]
    source_rotation=source_view.world_view_transform[:3,:3].to(device)
    target_transform=target_view.world_view_transform.to(device)
    answer=[]
    for start in range(0,len(uv),chunk_size):
        pixels=uv[start:start+chunk_size,None]+offsets[None]
        source_grid=(pixels+.5)/torch.tensor([source_view.image_width,source_view.image_height],device=device)*2-1
        reference=F.grid_sample(source_image,source_grid[None],align_corners=False,padding_mode='zeros')[0].permute(1,2,0)
        rays=torch.stack(((pixels[...,0]-source_view.cx)/source_view.focal_x,
                          (pixels[...,1]-source_view.cy)/source_view.focal_y,torch.ones_like(pixels[...,0])),dim=-1)
        depths=source_depth[start:start+chunk_size,None]*log_shifts.exp()[None]
        camera_points=rays[:,None]*depths[:,:,None,None]
        world=camera_points@source_rotation.T+source_view.camera_center.to(device)
        projected=world@target_transform[:3,:3]+target_transform[3,:3]
        z=projected[...,2]
        u=target_view.focal_x*projected[...,0]/z.clamp_min(1e-6)+target_view.cx
        v=target_view.focal_y*projected[...,1]/z.clamp_min(1e-6)+target_view.cy
        inside=(z>0)&(u>=0)&(v>=0)&(u<target_view.image_width-1)&(v<target_view.image_height-1)
        target_grid=torch.stack(((u+.5)/target_view.image_width*2-1,(v+.5)/target_view.image_height*2-1),dim=-1)
        shape=target_grid.shape
        warped=F.grid_sample(target_image,target_grid.reshape(1,-1,25,2),align_corners=False,padding_mode='zeros')[0]
        warped=warped.permute(1,2,0).reshape(shape[0],shape[1],25,3)
        score=normalized_patch_correlation(reference[:,None],warped)
        valid=inside.all(dim=-1)&(source_grid.abs()<=1).all(dim=-1).all(dim=-1)[:,None]
        answer.append(torch.where(valid,score,torch.full_like(score,-1.)))
    return torch.cat(answer) if answer else torch.empty((0,len(log_shifts)),device=device)
