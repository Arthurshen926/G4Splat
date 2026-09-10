"""Read-only source DC / directional SH / opacity swaps; no fitted updates."""
import torch


def region_psnr(rgb, target, regions):
    rgb = rgb.clamp(0, 1)
    return {name: float(-10*(rgb[:, mask]-target[:, mask]).square().mean().clamp_min(1e-12).log10())
            if mask.any() else None for name, mask in regions.items()}


@torch.no_grad()
def source_optics_components(original, refined, render, target, regions):
    if original is refined: raise ValueError('Separate immutable source reference required')
    features = refined.features.detach().clone(); logits = refined.opacity_logits.detach().clone()
    result = {}
    try:
        for mode in ('dc_only', 'directional_sh_only', 'opacity_only'):
            refined.features.copy_(original.features)
            refined.opacity_logits.copy_(original.opacity_logits)
            if mode == 'dc_only': refined.features[:, 0].copy_(features[:, 0])
            elif mode == 'directional_sh_only': refined.features[:, 1:].copy_(features[:, 1:])
            else: refined.opacity_logits.copy_(logits)
            result[mode] = region_psnr(render(), target, regions)
    finally:
        refined.features.copy_(features); refined.opacity_logits.copy_(logits)
    return result
