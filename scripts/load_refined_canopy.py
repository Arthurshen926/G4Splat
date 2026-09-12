"""Strict read-only restoration of the static v480-shaped diagnostic recipe."""
import hashlib
from pathlib import Path
import torch


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


@torch.no_grad()
def load_refined_canopy(path, teacher, contract):
    from scripts.canopy_candidate_diagnostic import CandidateCloud, NativeCandidateView
    from scripts.canopy_selective_shape import load_shape_cohort
    from scripts.canopy_selective_orientation import OrientedSelectiveCandidates
    from scripts.canopy_static_source_position import StaticSourcePosition, StaticSourcePositionView
    from outdoor.hybrid_gaussian_renderer import VolumetricFoliageModel, persistent_static_evidence_mask
    payload = torch.load(path, map_location='cpu', weights_only=False)
    m = payload['manifest']; a = m['args']; base = teacher.foliage
    if not payload['diagnostic_only'] or any(m[k] != contract[k] for k in ('training_views', 'excluded_views')):
        raise ValueError('Canopy camera contract mismatch')
    if m['source_checkpoint_sha256'] != sha256(contract['args']['checkpoint']):
        raise ValueError('Canopy source checkpoint mismatch')
    if m['masks_sha256'] != sha256(contract['args']['masks']):
        raise ValueError('Canopy mask identity mismatch')
    helpers = {'helper_sha256':'canopy_candidate_diagnostic.py',
        'spatial_helper_sha256':'canopy_candidate_spatial.py',
        'spatial_footprint_helper_sha256':'canopy_spatial_footprint.py',
        'candidate_shape_helper_sha256':'canopy_selective_shape.py',
        'candidate_orientation_helper_sha256':'canopy_selective_orientation.py',
        'joint_optics_helper_sha256':'canopy_candidate_joint_optics.py',
        'source_position_helper_sha256':'canopy_static_source_position.py'}
    for key, name in helpers.items():
        if m[key] != sha256(Path(__file__).with_name(name)):
            raise ValueError(f'Changed restoration helper: {name}')
    if (a['ray_log_radius'] or a['footprint_log_radius'] or not a['candidate_orientation_control']
            or a['leaf_optical_kernel'] != 'native' or a['candidate_gain'] != 1
            or a['candidate_optical_coordinate'] != 'bounded_logit'):
        raise ValueError('Unsupported canopy recipe')
    if m['candidate_shape_cohort_sha256'] != sha256(a['candidate_shape_cohort']):
        raise ValueError('Changed shape cohort')
    state = payload['candidate']
    cloud = CandidateCloud(state['cloud.xyz'], state['cloud.scales'], torch.zeros_like(state['cloud.xyz']))
    selected = load_shape_cohort(a['candidate_shape_cohort'], cloud, m['source_checkpoint_sha256'],
                                m['training_views'], m['excluded_views'])
    candidate = OrientedSelectiveCandidates(cloud, selected, a['position_radius_sigmas'], a['spatial_footprint_log_radius'])
    candidate.load_state_dict(state, strict=True); candidate.cuda().requires_grad_(False)
    refined = VolumetricFoliageModel(base.sh_degree, dynamic_rank=base.dynamic_rank).cuda()
    refined.restore(teacher.state['foliage'])
    eligible = persistent_static_evidence_mask(base) & base.static_leaf_mask
    optics = payload['source_optics']
    if set(optics) != {'features', 'opacity_logits'}:
        raise ValueError('Unexpected source optics')
    for key, value in optics.items():
        target = getattr(refined, key); value = value.to(target)
        if value.shape != target.shape or not torch.isfinite(value).all() or not torch.equal(value[~eligible], target[~eligible]):
            raise ValueError('Invalid source optics')
        target.copy_(value)
    refined.requires_grad_(False)
    position = StaticSourcePosition(base.xyz, base.scales, eligible, a['source_position_radius_sigmas'])
    position.load_verified_state(payload['source_position']); position.code.requires_grad_(False)
    adapter = NativeCandidateView(StaticSourcePositionView(refined, position), candidate)
    return adapter, candidate, dict(path=str(Path(path).resolve()), sha256=sha256(path), step=payload['step'])
