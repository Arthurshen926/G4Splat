"""Optical existence only on independently certified foreground observations."""
import torch

OCCLUDER_CONSENSUS_CONTRACT=('source_eroded_canopy_stable_moge_entire_interval_before_opaque_rigid__'
    'two_other_calibrated_cameras__minimum_ray_angle025deg__rigid_rgb_disagreement_above_rigid_residual_q90_plus002')


def confirmed_occluder_loss(native_leaf_alpha, confirmed):
    if confirmed.dtype != torch.bool or native_leaf_alpha.numel()!=confirmed.numel():
        raise ValueError('Expected aligned Boolean evidence and native scalar alpha')
    alpha=native_leaf_alpha.reshape_as(confirmed)
    return (-torch.log(alpha.clamp(1.e-6,1.)) * confirmed).sum()/confirmed.sum().clamp_min(1)


def select_occluder_evidence(evidence, scope):
    """Source candidates are an explicit weaker ablation, never certificates."""
    if scope not in ('confirmed', 'source_candidate'):
        raise ValueError('Unknown occluder evidence scope')
    candidate, confirmed = evidence['candidate'], evidence['confirmed']
    if (candidate.dtype != torch.bool or confirmed.dtype != torch.bool
        or candidate.shape != confirmed.shape or bool((confirmed & ~candidate).any())):
        raise ValueError('Invalid occluder evidence masks')
    return confirmed if scope == 'confirmed' else candidate


def validate_consensus_coverage(manifest, train, excluded):
    train=set(train);excluded=set(excluded)
    declared=set(manifest['calibrated_training_views'])
    selected=set(manifest['selected_training_views'])
    canopy=set(manifest['canopy_training_views'])
    if (manifest.get('contract')!=OCCLUDER_CONSENSUS_CONTRACT
        or not manifest['all_canopy_training_views_covered'] or declared!=train
        or selected!=canopy or not selected<=declared or declared&excluded
        or set(manifest['excluded_views'])!=excluded):
        raise ValueError('Occluder evidence must cover the complete matching nonvalidation training cohort')
