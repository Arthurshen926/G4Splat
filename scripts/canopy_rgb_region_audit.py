"""Read-only RGB attribution; never changes training/evaluation masks."""
import torch


def rgb_region_audit(source, updated, target, rigid, outer_tree_band):
    if source.shape != updated.shape or source.shape != target.shape or source.shape != (3, *rigid.shape):
        raise ValueError('Aligned CHW RGB required')
    if rigid.dtype != torch.bool or outer_tree_band.dtype != torch.bool or rigid.shape != outer_tree_band.shape:
        raise ValueError('Aligned boolean region masks required')
    source_error = (source-target).square().mean(0)
    updated_error = (updated-target).square().mean(0)
    result = {}
    for name, mask in {'rigid_all': rigid, 'rigid_tree_interface': rigid & outer_tree_band,
                       'rigid_away_from_tree_interface': rigid & ~outer_tree_band}.items():
        count = int(mask.sum())
        result[name] = dict(pixels=count, source_psnr=None, updated_psnr=None,
                            rgb_change_l1=None, worsened_fraction=None)
        if count:
            result[name].update(
                source_psnr=float(-10*source_error[mask].mean().clamp_min(1e-12).log10()),
                updated_psnr=float(-10*updated_error[mask].mean().clamp_min(1e-12).log10()),
                rgb_change_l1=float((updated-source).abs().mean(0)[mask].mean()),
                worsened_fraction=float((updated_error[mask] > source_error[mask]+1e-8).float().mean()))
    return result
