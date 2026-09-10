"""RGB-only, static multi-depth candidate experiment; never a verified map."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'2d-gaussian-splatting')]
import numpy as np
import torch
from PIL import Image
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import render_hybrid, static_detail_forward_visibility_gate
from outdoor.moge3_evidence import sha256_file, canonical_json_sha256
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.canopy_candidate_diagnostic import CandidateCloud, NativeCandidateView, ray_candidates
from scripts.seed_bound_moge_diagnostic import configure_seed_bound_moge
from scripts.evaluate_canopy_validation import FIXED, ADDITIONAL, tensor_digest
from scripts.evaluate_hybrid_teacher import _tree_boundary_masks
from scripts.canopy_candidate_rgb_ownership import candidate_rgb_loss
from scripts.canopy_diagnostic_checkpoint import save_diagnostic_checkpoint, publish_diagnostic_metrics
from scripts.canopy_gradient_accumulation import accumulation_window
from scripts.canopy_color_gradient_ownership import backward_canopy_owned_colors
from scripts.canopy_frozen_reference_cache import FrozenReferenceCache
from scripts.canopy_seed_ray_sampling import select_seed_rays


class Calibration:
    def configure_moge3_metric_scale(self, scale): self.scale = scale
    def configure_moge3_canopy_depth_scales(self, profiles): self.profiles = profiles


def main():
    p = argparse.ArgumentParser(description=__doc__); model = ModelParams(p)
    p.set_defaults(data_device='cpu', resolution=640, white_background=True)
    for key in ('checkpoint', 'cohort', 'masks', 'runtime-cache', 'initialization', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--ratios', type=float, nargs=3, required=True)
    p.add_argument('--rays-per-view', type=int, default=256)
    p.add_argument('--seed-ray-policy', choices=('uniform', 'rgb_floor_priority'), default='uniform')
    p.add_argument('--steps', type=int, default=800)
    p.add_argument('--eval-every', type=int, default=400)
    p.add_argument('--gradient-accumulation', type=int, default=1,
                   help='Views averaged per Adam update; steps remain the total training-image budget')
    p.add_argument('--canopy-only-color-gradients', action='store_true',
                   help='Controlled training-only color ownership; opacity/geometry still use all RGB regions')
    p.add_argument('--gpu-memory-fraction', type=float, default=.42)
    p.add_argument('--image-cache-size', type=int, default=384,
                   help='CPU RGB cache; default fits the 304 training plus 48 evaluation cameras at640')
    p.add_argument('--reference-cache-size', type=int, default=0,
                   help='Optional CPU cache of immutable source RGB; never stores refined model RGB')
    p.add_argument('--opacity-lr', type=float, default=.05)
    p.add_argument('--color-lr', type=float, default=.025)
    p.add_argument('--ray-log-radius', type=float, default=0.)
    p.add_argument('--ray-lr', type=float, default=.02)
    p.add_argument('--rigid-rgb-preservation-weight', type=float, default=0.)
    p.add_argument('--rigid-boundary-preservation-weight', type=float, default=0.)
    p.add_argument('--candidate-color-domain', choices=('unit_rgb', 'legacy_unconstrained'), default='unit_rgb')
    p.add_argument('--search-depth-center', choices=('moge', 'blocking_surface'), default='moge')
    p.add_argument('--depth-search-policy', choices=('fixed_layers', 'stratified_volume', 'photometric_volume'), default='fixed_layers')
    p.add_argument('--noncanopy-rgb-target', choices=('source_reference', 'legacy_ground_truth'), default='source_reference')
    p.add_argument('--footprint-log-radius', type=float, default=0.)
    p.add_argument('--footprint-lr', type=float, default=.02)
    p.add_argument('--position-radius-sigmas', type=float, default=0.)
    p.add_argument('--position-lr', type=float, default=.02)
    p.add_argument('--spatial-footprint-log-radius', type=float, default=0.)
    p.add_argument('--joint-persistent-optics', action='store_true')
    p.add_argument('--candidate-gain', type=int, choices=(0, 1), default=1)
    p.add_argument('--source-color-lr', type=float, default=.0025)
    p.add_argument('--source-opacity-lr', type=float, default=.02)
    p.add_argument('--source-position-radius-sigmas', type=float, default=0.)
    p.add_argument('--source-position-lr', type=float, default=.01)
    p.add_argument('--source-rest-step-multiplier', type=float, default=1.)
    p.add_argument('--surface-rgb-feasibility-weight', type=float, default=0.,
                   help='Optional controlled RGB floor auxiliary; disabled unless explicitly requested')
    p.add_argument('--surface-rgb-feasibility-domain', choices=('absolute', 'relative'), default='absolute')
    a = p.parse_args()
    reference_cache = FrozenReferenceCache(a.reference_cache_size)
    if not 1 <= a.gradient_accumulation <= 8 or a.eval_every % a.gradient_accumulation:
        raise ValueError('Evaluation must occur at complete bounded accumulation windows')
    if not (1 <= a.rays_per_view <= 1024 and a.steps > 0 and a.eval_every > 0 and 1 <= a.image_cache_size <= 512
            and 0 < a.gpu_memory_fraction <= .8 and 0 < a.opacity_lr <= .1
            and 0 < a.color_lr <= .1 and 0 <= a.ray_log_radius <= .3
            and 0 < a.ray_lr <= .1 and 0 <= a.rigid_rgb_preservation_weight <= 10
            and 0 <= a.rigid_boundary_preservation_weight <= 4
            and 0 <= a.footprint_log_radius <= math.log(3.) and 0 < a.footprint_lr <= .1
            and 0 <= a.position_radius_sigmas <= 16 and 0 < a.position_lr <= .1
            and 0 < a.source_color_lr <= .1 and 0 <= a.source_opacity_lr <= .1
            and 0 <= a.source_position_radius_sigmas <= 4 and 0 < a.source_position_lr <= .1
            and 0 <= a.source_rest_step_multiplier <= 1
            and 0 <= a.spatial_footprint_log_radius <= math.log(2.)
            and 0 <= a.surface_rgb_feasibility_weight <= 4):
        raise ValueError('Bounded candidate schedule required')
    if sum(bool(x) for x in (a.ray_log_radius, a.footprint_log_radius, a.position_radius_sigmas)) > 1:
        raise ValueError('Depth, footprint and spatial motion are separate controlled experiments')
    if a.spatial_footprint_log_radius and not a.position_radius_sigmas:
        raise ValueError('Independent combined footprint requires bounded spatial candidates')
    if not a.candidate_gain and not a.joint_persistent_optics:
        raise ValueError('Disabled candidates require the shared-source optics control')
    if a.surface_rgb_feasibility_weight and a.noncanopy_rgb_target != 'source_reference':
        raise ValueError('RGB floor control requires preserved immutable background ownership')
    if a.rigid_boundary_preservation_weight and a.noncanopy_rgb_target != 'source_reference':
        raise ValueError('Boundary control requires immutable background ownership')
    if a.source_rest_step_multiplier != 1 and not a.joint_persistent_optics:
        raise ValueError('Directional source step requires explicit joint optical authority')
    if a.source_position_radius_sigmas and (not a.joint_persistent_optics or a.noncanopy_rgb_target != 'source_reference'):
        raise ValueError('Source positions require supported joint leaves and immutable building ownership')
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(a.gpu_memory_fraction)
    cohort = json.loads(a.cohort.read_text()); holdout = FIXED+ADDITIONAL
    train = sorted(cohort['calibrated_training_views'])
    source_hash = sha256_file(a.checkpoint)
    if (set(train)&set(holdout) or set(cohort['excluded_views']) != set(holdout)
            or cohort['source_checkpoint_sha256'] != source_hash):
        raise ValueError('Source-matched training cohort excluding all 48 views required')
    dataset = model.extract(a); dataset.model_path = str(a.output)
    teacher = load_hybrid_teacher(a.checkpoint, sh_degree=dataset.sh_degree)
    base = teacher.foliage
    frozen = {f'foliage.{k}': v for k, v in base.named_parameters()}
    frozen.update({f'surface.{k}': getattr(teacher.surface, k) for k in
                   ('_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest')})
    for label, module in [('sky', teacher.sky), ('appearance', teacher.appearance)]:
        if hasattr(module, 'named_parameters'):
            frozen.update({f'{label}.{k}': v for k, v in module.named_parameters()})
    for value in frozen.values(): value.requires_grad_(False)
    before = {k: tensor_digest(v) for k, v in frozen.items()}
    views = LazyScene(dataset, GaussianModel(dataset.sh_degree), image_cache_size=a.image_cache_size).getTrainCameras()
    canonical = teacher.state['training_contract']['static_scene_canonical_rgb']['canonical_sequence']
    if any(not views[i].image_name.startswith(canonical+'__') for i in train+holdout):
        raise ValueError('Canonical-only candidate experiment required')
    masks = CambridgeMaskLookup(Path(dataset.source_path), a.masks)
    def regions(v):
        obj, sky, dist, tree = masks.get_index_masks(v.image_name, (0, 1, 2, 3),
            (v.image_height, v.image_width), torch.device('cuda'))
        canopy = obj&sky&dist&~tree; rigid = obj&sky&dist&tree
        inside, outside, _ = _tree_boundary_masks(~canopy)
        return dict(tree=canopy, tree_interior=canopy&~inside, tree_boundary=canopy&inside,
                    rigid=rigid, hard=rigid&outside, sky=obj&dist&~sky)
    cal = Calibration()
    calibration = configure_seed_bound_moge(cal, a.initialization, teacher.state['training_contract'], .8277335147998328)
    cache = json.loads(a.runtime_cache.read_text()); unhashed = dict(cache)
    if unhashed.pop('index_hash') != canonical_json_sha256(unhashed): raise ValueError('Changed runtime index')
    entry = cache['arrays']['depth_m']; depth_path = a.runtime_cache.parent/entry['path']
    if sha256_file(depth_path) != entry['sha256']: raise ValueError('Changed raw depth array')
    raw = np.load(depth_path, mmap_mode='r', allow_pickle=False)
    seeds = []; seed_audit = []; search_audit = []; ray_sampling_audit = []; relative_floor_scales = {}
    black_seed_view = None
    if a.seed_ray_policy == 'rgb_floor_priority' or a.surface_rgb_feasibility_weight or a.surface_rgb_feasibility_domain == 'relative':
        from scripts.canopy_surface_rgb_feasibility import BlackRadianceView
        from scripts.canopy_relative_radiance_feasibility import initial_gradient_normalization
        black_seed_view = BlackRadianceView(base)
    photometric_seed_audit = []; photometric_neighbors = {}
    if a.depth_search_policy == 'photometric_volume':
        from scripts.canopy_photometric_depth_search import refine_stratified_proposals
        centers = torch.stack([views[i].camera_center for i in train])
        neighbor_order = torch.cdist(centers, centers).argsort(1).cpu().tolist()
        photometric_neighbors = {index: [train[j] for j in neighbor_order[k] if train[j] != index][:4]
                                 for k, index in enumerate(train)}
        if any(set(neighbors)&set(holdout) for neighbors in photometric_neighbors.values()):
            raise ValueError('Evaluation cameras cannot select proposal depth')
    search_helper_hash = None
    if a.search_depth_center == 'blocking_surface':
        from scripts.canopy_candidate_search_depth import blocking_aware_search_depth
        search_helper_hash = sha256_file(Path(__file__).with_name('canopy_candidate_search_depth.py'))
    with torch.no_grad():
        for ordinal, index in enumerate(train):
            v = views[index]; seed_regions = regions(v); mask = seed_regions['tree_interior']
            depth = torch.as_tensor(np.array(raw[cache['camera_order'].index(v.image_name)], copy=True),
                                    device='cuda')*cal.scale*cal.profiles[v.image_name]
            if depth.shape != mask.shape: raise ValueError('Depth and native image must align')
            y, x = torch.where(mask & torch.isfinite(depth) & (depth > .2))
            scores = None
            if black_seed_view is not None and seed_regions['tree'].any():
                gate = static_detail_forward_visibility_gate(base, int(v.colmap_id), include_pending_exact=False)
                seed_floor = render_hybrid(v, teacher.surface, black_seed_view,
                    background=torch.zeros(3, device='cuda'), include_dynamic=False,
                    optical_replacement_policy='disabled', structural_trainable_start=None, volume_gate=gate)
                relative_floor_scales[index] = initial_gradient_normalization(
                    seed_floor.render, v.original_image.cuda(), seed_regions['tree'])
                if a.seed_ray_policy == 'rgb_floor_priority':
                    scores = (seed_floor.render-v.original_image.cuda()).clamp_min(0).square().mean(0)[y, x]
                del seed_floor
            if len(y):
                selected = select_seed_rays(len(y), a.rays_per_view, 1701+index, scores).cuda()
                ray_sampling_audit.append(dict(index=index, population=len(y), selected=len(selected),
                    positive_score_population=int((scores > 0).sum()) if scores is not None else None,
                    population_score_mean=float(scores.mean()) if scores is not None else None,
                    selected_positive_score=int((scores[selected] > 0).sum()) if scores is not None else None,
                    selected_score_mean=float(scores[selected].mean()) if scores is not None else None,
                    selected_pixels_sha256=tensor_digest(torch.stack((y[selected], x[selected]), 1)),
                    geometry_truth=False))
                y, x = y[selected], x[selected]
                selected_depth = depth[y, x]
                if a.search_depth_center == 'blocking_surface':
                    wall = render_hybrid(v, teacher.surface, base, background=torch.zeros(3, device='cuda'),
                        include_dynamic=False, optical_replacement_policy='disabled', structural_trainable_start=None,
                        volume_gate=torch.zeros_like(base.opacity_logits.reshape(-1)))
                    selected_depth, changed = blocking_aware_search_depth(selected_depth,
                        wall.median_depth[0, y, x], wall.surface_alpha[0, y, x])
                    search_audit.append(dict(index=index, rays=len(y), closer_blocker_rays=int(changed.sum())))
                    del wall
                if a.depth_search_policy != 'fixed_layers':
                    from scripts.canopy_stratified_depth_candidates import stratified_ray_candidates
                    uv = torch.stack((x, y), 1).float()
                    proposals = stratified_ray_candidates(v, uv, selected_depth,
                        v.original_image.cuda()[:, y, x].T, seed=37001+index)
                    if a.depth_search_policy == 'photometric_volume':
                        neighbor_ids = photometric_neighbors[index]
                        neighbors = [(views[j], views[j].original_image.cuda(), regions(views[j])['tree'])
                                     for j in neighbor_ids]
                        proposals, photo_audit = refine_stratified_proposals(v, uv, selected_depth,
                            v.original_image.cuda(), neighbors, proposals)
                        photometric_seed_audit.append(dict(index=index, neighbors=neighbor_ids, **photo_audit))
                        del neighbors
                    seeds.append(proposals)
                else:
                    seeds.append(ray_candidates(v, torch.stack((x, y), 1).float(), selected_depth,
                                                v.original_image.cuda()[:, y, x].T, a.ratios))
            seed_audit.append(dict(index=index, rays=len(y), image_sha256=tensor_digest(v.original_image)))
            if ordinal % 50 == 0: print(json.dumps(dict(seed_views=ordinal+1)), flush=True)
    if not seeds: raise ValueError('No training canopy rays')
    xyz, scales, colors = [torch.cat([s[i] for s in seeds]) for i in range(3)]
    if len(base)+len(xyz) > 2000000: raise ValueError('Global foliage capacity exceeded')
    candidate = CandidateCloud(xyz, scales, colors)
    refinement_hash = None
    if a.ray_log_radius or a.rigid_rgb_preservation_weight or a.candidate_color_domain == 'unit_rgb':
        from scripts.canopy_candidate_ray_refinement import RayDepthCandidates, rigid_rgb_preservation, project_candidate_dc_
        refinement_hash = sha256_file(Path(__file__).with_name('canopy_candidate_ray_refinement.py'))
    if a.candidate_color_domain == 'unit_rgb': project_candidate_dc_(candidate.dc)
    if a.ray_log_radius:
        origins = torch.cat([views[r['index']].camera_center[None].expand(r['rays']*len(a.ratios), -1)
                             for r in seed_audit if r['rays']])
        candidate = RayDepthCandidates(candidate, origins, a.ray_log_radius)
    footprint_hash = None
    if a.footprint_log_radius:
        from scripts.canopy_candidate_footprint import FootprintCandidates
        footprint_hash = sha256_file(Path(__file__).with_name('canopy_candidate_footprint.py'))
        candidate = FootprintCandidates(candidate, a.footprint_log_radius)
    spatial_hash = None
    if a.position_radius_sigmas:
        from scripts.canopy_candidate_spatial import SpatialCandidates
        spatial_hash = sha256_file(Path(__file__).with_name('canopy_candidate_spatial.py'))
        candidate = SpatialCandidates(candidate, a.position_radius_sigmas)
        if a.spatial_footprint_log_radius:
            from scripts.canopy_spatial_footprint import SpatialFootprintCandidates
            candidate = SpatialFootprintCandidates(candidate.cloud, a.position_radius_sigmas,
                                                   a.spatial_footprint_log_radius)
    joint_hash = None; reference_base = None; eligible = None; source_position = None
    if a.joint_persistent_optics:
        from outdoor.hybrid_gaussian_renderer import VolumetricFoliageModel, persistent_static_evidence_mask
        from scripts.canopy_candidate_joint_optics import JointOpticalCandidateView
        from scripts.canopy_directional_sh_step import rescale_directional_sh_step_
        from scripts.canopy_source_opacity_projection import project_source_opacity_
        joint_hash = sha256_file(Path(__file__).with_name('canopy_candidate_joint_optics.py'))
        reference_base = VolumetricFoliageModel(a.sh_degree, dynamic_rank=base.dynamic_rank).cuda()
        reference_base.restore(teacher.state['foliage'])
        for p_ref in reference_base.parameters(): p_ref.requires_grad_(False)
        eligible = persistent_static_evidence_mask(base)&base.static_leaf_mask
        for key in ('features', 'opacity_logits'):
            frozen.pop('foliage.'+key)
            getattr(base, key).requires_grad_(True)
        optical_base = base
        if a.source_position_radius_sigmas:
            from scripts.canopy_static_source_position import StaticSourcePosition, StaticSourcePositionView
            source_position = StaticSourcePosition(base.xyz, base.scales, eligible, a.source_position_radius_sigmas)
            optical_base = StaticSourcePositionView(base, source_position)
        adapter = JointOpticalCandidateView(optical_base, candidate, eligible)
        before = {k: tensor_digest(v) for k, v in frozen.items()}
        before.update({f'ineligible.{k}': tensor_digest(getattr(base, k)[~eligible])
                       for k in ('features', 'opacity_logits')})
        before.update({f'reference.{k}': tensor_digest(v) for k, v in reference_base.named_parameters()})
        if a.source_rest_step_multiplier == 0:
            before['frozen_source_directional_sh'] = tensor_digest(base.features[:, 1:])
        if a.source_opacity_lr == 0:
            before['frozen_source_opacity'] = tensor_digest(base.opacity_logits)
        if source_position is not None:
            before.update({f'source_position.{k}': tensor_digest(v) for k, v in source_position.named_buffers()})
    else:
        adapter = NativeCandidateView(base, candidate)
    if not a.candidate_gain:
        for p_candidate in candidate.parameters(): p_candidate.requires_grad_(False)
    del seeds, xyz, scales, colors
    groups = [dict(params=[candidate.dc], lr=a.color_lr), dict(params=[candidate.logits], lr=a.opacity_lr)]
    initial_lrs = [a.color_lr, a.opacity_lr]
    if a.ray_log_radius:
        groups.append(dict(params=[candidate.depth_code], lr=a.ray_lr)); initial_lrs.append(a.ray_lr)
    if a.footprint_log_radius:
        groups.append(dict(params=[candidate.scale_code], lr=a.footprint_lr)); initial_lrs.append(a.footprint_lr)
    if a.position_radius_sigmas:
        groups.append(dict(params=[candidate.position_code], lr=a.position_lr)); initial_lrs.append(a.position_lr)
    if a.spatial_footprint_log_radius:
        groups.append(dict(params=[candidate.scale_code], lr=a.footprint_lr)); initial_lrs.append(a.footprint_lr)
    if not a.candidate_gain: groups = []; initial_lrs = []
    if a.joint_persistent_optics:
        groups.extend([dict(params=[base.features], lr=a.source_color_lr),
                       dict(params=[base.opacity_logits], lr=a.source_opacity_lr)])
        initial_lrs.extend([a.source_color_lr, a.source_opacity_lr])
    if source_position is not None:
        groups.append(dict(params=[source_position.code], lr=a.source_position_lr))
        initial_lrs.append(a.source_position_lr)
    optimizer = torch.optim.Adam(groups, eps=1e-15)
    def render(v, enabled=True):
        gate = static_detail_forward_visibility_gate(base, int(v.colmap_id), include_pending_exact=False)
        gate = torch.cat((gate, gate.new_full((len(candidate.xyz),), float(enabled)*a.candidate_gain)))
        out = render_hybrid(v, teacher.surface, adapter, background=torch.zeros(3, device='cuda'),
                            include_dynamic=False, optical_replacement_policy='disabled',
                            structural_trainable_start=None, volume_gate=gate)
        return out.render+(1-out.alpha)*teacher.sky(v), out.volume_alpha
    def compute_immutable_reference(v):
        if reference_base is None: return render(v, False)[0]
        gate = static_detail_forward_visibility_gate(reference_base, int(v.colmap_id), include_pending_exact=False)
        out = render_hybrid(v, teacher.surface, reference_base, background=torch.zeros(3, device='cuda'),
            include_dynamic=False, optical_replacement_policy='disabled', structural_trainable_start=None, volume_gate=gate)
        return out.render+(1-out.alpha)*teacher.sky(v)
    def immutable_reference(v):
        key = (int(v.colmap_id), v.image_height, v.image_width)
        return reference_cache.get(key, lambda: compute_immutable_reference(v), torch.device('cuda'))
    floor_adapter = None
    if a.surface_rgb_feasibility_weight:
        from scripts.canopy_surface_rgb_feasibility import BlackRadianceView, surface_rgb_feasibility_loss
        from scripts.canopy_relative_radiance_feasibility import relative_surface_rgb_feasibility_loss
        floor_adapter = BlackRadianceView(adapter)
    floor_helper_hash = sha256_file(Path(__file__).with_name('canopy_surface_rgb_feasibility.py'))
    from scripts.canopy_boundary_preservation import boundary_rgb_preservation
    manifest = dict(scope=('joint_persistent_optics_and_unverified_candidates__rigid_frozen__optional_bounded_source_positions'
                          if a.joint_persistent_optics else 'unverified_static_candidates__RGB_only__all_existing_parameters_frozen'),
                    args=vars(a), source_checkpoint_sha256=source_hash, training_views=train,
                    excluded_views=holdout, seed_audit=seed_audit, calibration=calibration,
                    ray_sampling_audit=ray_sampling_audit,
                    candidate_count=len(candidate.xyz), initial_opacity=.001,
                    depth_search_contract=(dict(policy=('photometric_stratified_log_depth' if a.depth_search_policy == 'photometric_volume'
                                                       else 'stratified_log_depth'), bounds=[.4, 1.2], strata=3,
                                                seed_rule='37001+training_camera_index', geometry_truth=False)
                                           if a.depth_search_policy != 'fixed_layers' else
                                           dict(policy='fixed_relative_layers', ratios=a.ratios, geometry_truth=False)),
                    helper_sha256=sha256_file(Path(__file__).with_name('canopy_candidate_diagnostic.py')),
                    script_sha256=sha256_file(Path(__file__)), masks_sha256=sha256_file(a.masks),
                    refinement_helper_sha256=refinement_hash,
                    search_helper_sha256=search_helper_hash, search_audit=search_audit,
                    rgb_ownership_helper_sha256=sha256_file(Path(__file__).with_name('canopy_candidate_rgb_ownership.py')),
                    footprint_helper_sha256=footprint_hash,
                    spatial_helper_sha256=spatial_hash,
                    spatial_footprint_helper_sha256=sha256_file(Path(__file__).with_name('canopy_spatial_footprint.py')),
                    joint_optics_helper_sha256=joint_hash,
                    surface_rgb_feasibility_helper_sha256=floor_helper_hash,
                    relative_radiance_helper_sha256=sha256_file(Path(__file__).with_name('canopy_relative_radiance_feasibility.py')),
                    relative_floor_initial_scales=relative_floor_scales,
                    relative_floor_normalization='fixed_original_source_per_training_view__initial_surface_RGB_gradient_L1_matched__epsilon_1_over_255',
                    boundary_preservation_helper_sha256=sha256_file(Path(__file__).with_name('canopy_boundary_preservation.py')),
                    stratified_depth_helper_sha256=sha256_file(Path(__file__).with_name('canopy_stratified_depth_candidates.py')),
                    photometric_depth_helper_sha256=sha256_file(Path(__file__).with_name('canopy_photometric_depth_search.py')),
                    photometric_seed_audit=photometric_seed_audit,
                    diagnostic_checkpoint_helper_sha256=sha256_file(Path(__file__).with_name('canopy_diagnostic_checkpoint.py')),
                    directional_sh_step_helper_sha256=sha256_file(Path(__file__).with_name('canopy_directional_sh_step.py')),
                    source_opacity_projection_helper_sha256=sha256_file(Path(__file__).with_name('canopy_source_opacity_projection.py')),
                    gradient_accumulation_helper_sha256=sha256_file(Path(__file__).with_name('canopy_gradient_accumulation.py')),
                    color_gradient_ownership_helper_sha256=sha256_file(Path(__file__).with_name('canopy_color_gradient_ownership.py')),
                    frozen_reference_cache_helper_sha256=sha256_file(Path(__file__).with_name('canopy_frozen_reference_cache.py')),
                    seed_ray_sampling_helper_sha256=sha256_file(Path(__file__).with_name('canopy_seed_ray_sampling.py')),
                    source_position_helper_sha256=sha256_file(Path(__file__).with_name('canopy_static_source_position.py')),
                    source_position_authority=('persistent_static_evidence_and_static_leaf_only__bounded_by_initial_largest_sigma__static_RGB'),
                    training_budget_unit='rendered_training_images',
                    runtime_index_sha256=sha256_file(a.runtime_cache), depth_sha256=entry['sha256'],
                    limitations=['Historical training images, not blind generalization',
                                 'Candidates are hypotheses, not verified geometry; no production export'])
    (a.output/'manifest.json').write_text(json.dumps(manifest, default=str, indent=2))
    results = []
    @torch.no_grad()
    def evaluate(step):
        rows = []
        for index in holdout:
            v = views[index]; m = regions(v); target = v.original_image.cuda()
            predictions = {'candidate': render(v)[0]}
            if step == 0:
                original = teacher.render(v, task=None, conditioned=False)['rgb']
                zero_error = float((render(v, False)[0].clamp(0, 1)-original).abs().max())
                if zero_error > 1e-6: raise RuntimeError(f'Disabled candidates alter native render: {zero_error}')
                predictions['source'] = original
            row = dict(index=index, modes={})
            for mode, rgb in predictions.items():
                rgb = rgb.clamp(0, 1)
                row['modes'][mode] = {k: float(-10*(rgb[:, mask]-target[:, mask]).square().mean().clamp_min(1e-12).log10())
                                    if mask.any() else None for k, mask in m.items() if k != 'sky'}
                if mode == 'candidate':
                    pair = torch.cat((target, rgb), 2).permute(1, 2, 0).cpu().numpy()
                    Image.fromarray((pair*255).round().astype('uint8')).save(a.output/f'view_{index}_{step:04d}.png')
            rows.append(row)
        summary = {mode: {k: float(np.mean([r['modes'][mode][k] for r in rows if r['modes'][mode][k] is not None]))
                          for k in rows[0]['modes'][mode]} for mode in rows[0]['modes']}
        results.append(dict(step=step, summary=summary, per_view=rows))
        if step:
            parameter_names = {id(p): 'candidate.'+name for name, p in candidate.named_parameters()}
            parameter_names.update({id(p): 'source.'+name for name, p in base.named_parameters()})
            if source_position is not None:
                parameter_names[id(source_position.code)] = 'source_position.code'
            save_diagnostic_checkpoint(a.output/f'candidates_{step:04d}.pth', dict(
                diagnostic_only=True, candidate=candidate.state_dict(), manifest=manifest, step=step,
                source_optics=({k: getattr(base, k).detach().cpu() for k in ('features', 'opacity_logits')}
                               if a.joint_persistent_optics else None),
                source_position=(source_position.state_dict() if source_position is not None else None),
                optimizer_state=optimizer.state_dict(),
                optimizer_parameter_layout=[[parameter_names[id(p)] for p in group['params']] for group in optimizer.param_groups],
                training_order=order, optimizer_schedule_horizon=a.steps,
                completed_training_images=step, completed_optimizer_updates=math.ceil(step/a.gradient_accumulation),
                gradient_accumulation_pending=False,
                torch_rng=torch.random.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                diagnostic_optimizer_state_version=1))
        # Advertise an evaluation only AFTER its immutable snapshot is fully
        # published. Readers cannot discover an iteration with a partial ZIP.
        publish_diagnostic_metrics(a.output/'metrics.json', dict(results=results))
        print(json.dumps(dict(evaluation=step, summary=summary)), flush=True)
    evaluate(0)
    order = [train[i] for i in torch.randperm(len(train), generator=torch.Generator().manual_seed(1701)).tolist()]
    saw_gradient = saw_update = False
    saw_depth_gradient = False
    saw_scale_gradient = False
    saw_position_gradient = False
    saw_source_opacity_update = False
    saw_source_position_gradient = False
    for step in range(1, a.steps+1):
        training_index = order[(step-1) % len(order)]
        v = views[training_index]; m = regions(v); target = v.original_image.cuda()
        first_microview, last_microview, microview_count = accumulation_window(step, a.steps, a.gradient_accumulation)
        if first_microview:
            optimizer.zero_grad(set_to_none=True)
            batch_loss = batch_floor_loss = batch_boundary_loss = 0.
        rgb, _ = render(v)
        # RGB only, including known building and sky pixels: no opacity targets,
        # no MoGe positive hit/empty-space loss, no forced optical mass increase.
        reference = None
        if a.rigid_rgb_preservation_weight or a.noncanopy_rgb_target == 'source_reference':
            with torch.no_grad(): reference = immutable_reference(v)
            if a.reference_cache_size and step == 1:
                if not torch.equal(reference, immutable_reference(v)):
                    raise RuntimeError('Immutable reference cache changed RGB bits')
        loss = candidate_rgb_loss(rgb, target, reference, m, noncanopy_target=a.noncanopy_rgb_target)
        if a.rigid_rgb_preservation_weight:
            loss = loss+a.rigid_rgb_preservation_weight*rigid_rgb_preservation(rgb, reference, m['rigid'])
        boundary_loss = rgb.new_zeros(())
        if a.rigid_boundary_preservation_weight:
            boundary_loss = boundary_rgb_preservation(rgb, reference, m['rigid'], m['hard'])
            loss = loss+a.rigid_boundary_preservation_weight*boundary_loss
        floor_loss = rgb.new_zeros(())
        if floor_adapter is not None:
            gate = static_detail_forward_visibility_gate(base, int(v.colmap_id), include_pending_exact=False)
            floor_render = render_hybrid(v, teacher.surface, floor_adapter,
                background=torch.zeros(3, device='cuda'), include_dynamic=False,
                optical_replacement_policy='disabled', structural_trainable_start=None,
                volume_gate=torch.cat((gate, gate.new_full((len(candidate.xyz),), a.candidate_gain))))
            # Surface radiance through ACTUAL unchanged opacity/order. No sky
            # term, alpha target, geometric pseudo-label or inference-time mask.
            if a.surface_rgb_feasibility_domain == 'relative':
                if m['tree'].any() and training_index not in relative_floor_scales:
                    raise RuntimeError('Missing original-source training-view normalization')
                floor_loss = relative_surface_rgb_feasibility_loss(floor_render.render, target, m['tree'],
                    relative_floor_scales.get(training_index, 1.))
            else:
                floor_loss = surface_rgb_feasibility_loss(floor_render.render, target, m['tree'])
            loss = loss+a.surface_rgb_feasibility_weight*floor_loss
        if a.canopy_only_color_gradients:
            tree_loss = ((rgb-target).abs().mean(0)*m['tree']).sum()/m['tree'].sum().clamp_min(1)
            colors = ([candidate.dc] if a.candidate_gain else [])+([base.features] if a.joint_persistent_optics else [])
            color_ids = {id(parameter) for parameter in colors}
            other_parameters = [parameter for group in optimizer.param_groups for parameter in group['params']
                                if id(parameter) not in color_ids]
            backward_canopy_owned_colors(tree_loss, loss-tree_loss, color_parameters=colors,
                                         optical_geometry_parameters=other_parameters, divisor=microview_count)
        else:
            (loss/microview_count).backward()
        batch_loss += float(loss.detach())/microview_count
        batch_floor_loss += float(floor_loss.detach())/microview_count
        batch_boundary_loss += float(boundary_loss.detach())/microview_count
        if not last_microview:
            # Each backward frees its graph; parameters stay unchanged until
            # the complete multi-view mean gradient has been accumulated.
            continue
        if any(v.grad is not None for v in frozen.values()): raise RuntimeError('Frozen scene received gradient')
        if a.candidate_gain and not all(p.grad is not None and torch.isfinite(p.grad).all() for p in candidate.parameters()):
            raise RuntimeError('Missing or nonfinite candidate gradient')
        grad = float(candidate.logits.grad.abs().sum()) if a.candidate_gain else 0.; saw_gradient |= grad > 0
        depth_grad = float(candidate.depth_code.grad.abs().sum()) if a.ray_log_radius and a.candidate_gain else 0.
        saw_depth_gradient |= depth_grad > 0
        scale_grad = float(candidate.scale_code.grad.abs().sum()) if (a.footprint_log_radius or a.spatial_footprint_log_radius) and a.candidate_gain else 0.
        saw_scale_gradient |= scale_grad > 0
        position_grad = float(candidate.position_code.grad.abs().sum()) if a.position_radius_sigmas and a.candidate_gain else 0.
        saw_position_gradient |= position_grad > 0
        source_position_grad = 0.
        if source_position is not None:
            if source_position.code.grad is None or not torch.isfinite(source_position.code.grad).all():
                raise RuntimeError('Missing/nonfinite supported source-position gradient')
            source_position_grad = float(source_position.code.grad.abs().sum())
            saw_source_position_gradient |= source_position_grad > 0
        previous = candidate.logits.detach().clone()
        previous_source = base.opacity_logits.detach().clone() if a.joint_persistent_optics else None
        previous_rest = base.features[eligible, 1:].detach().clone() if a.joint_persistent_optics else None
        if a.joint_persistent_optics and not all(p.grad is not None and torch.isfinite(p.grad).all()
                                               for p in (base.features, base.opacity_logits)):
            raise RuntimeError('Missing/nonfinite source optical gradient')
        for group, initial in zip(optimizer.param_groups, initial_lrs):
            # Give low-opacity candidates time to be learned before annealing;
            # the paired narrow/wide runs share this exact optimizer schedule.
            group['lr'] = initial*.1**max(0., 2*(step-1)/max(a.steps-1, 1)-1)
        optimizer.step()
        if a.candidate_color_domain == 'unit_rgb': project_candidate_dc_(candidate.dc)
        if a.joint_persistent_optics:
            with torch.no_grad():
                rescale_directional_sh_step_(base.features, previous_rest, eligible, a.source_rest_step_multiplier)
                project_source_opacity_(base.opacity_logits, eligible, enabled=a.source_opacity_lr > 0)
                if a.source_rest_step_multiplier:
                    rest = base.features[eligible, 1:]
                    factor = (4./rest.flatten(1).norm(dim=1).clamp_min(1e-12)).clamp_max(1.)
                    base.features[eligible, 1:] = rest*factor[:, None, None]
                rest_step_mean = float((base.features[eligible, 1:]-previous_rest).abs().mean())
                source_changed = int((previous_source != base.opacity_logits).sum())
                saw_source_opacity_update |= source_changed > 0
        changed = int((previous != candidate.logits).sum()); saw_update |= changed > 0
        if step <= a.gradient_accumulation or step % 25 == 0 or step == a.steps:
            record = dict(step=step, optimizer_updates=math.ceil(step/a.gradient_accumulation),
                          averaged_training_views=microview_count,
                          loss=batch_loss, opacity_gradient_l1=grad, opacity_changed_rows=changed,
                          surface_rgb_feasibility_loss=batch_floor_loss,
                          rigid_boundary_preservation_loss=batch_boundary_loss,
                          opacity_mean=float(candidate.logits.sigmoid().mean()),
                          opacity_max=float(candidate.logits.sigmoid().max()),
                          ray_depth_gradient_l1=depth_grad,
                          footprint_gradient_l1=scale_grad,
                          position_gradient_l1=position_grad,
                          source_position_gradient_l1=source_position_grad,
                          source_opacity_changed_rows=source_changed if a.joint_persistent_optics else 0,
                          source_directional_sh_step_mean_abs=rest_step_mean if a.joint_persistent_optics else 0.,
                          peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
            with (a.output/'trace.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
        if step % a.eval_every == 0 or step == a.steps: evaluate(step)
    after = {k: tensor_digest(v) for k, v in frozen.items()}
    if a.joint_persistent_optics:
        after.update({f'ineligible.{k}': tensor_digest(getattr(base, k)[~eligible]) for k in ('features', 'opacity_logits')})
        after.update({f'reference.{k}': tensor_digest(v) for k, v in reference_base.named_parameters()})
        if a.source_rest_step_multiplier == 0:
            after['frozen_source_directional_sh'] = tensor_digest(base.features[:, 1:])
        if a.source_opacity_lr == 0:
            after['frozen_source_opacity'] = tensor_digest(base.opacity_logits)
        if source_position is not None:
            after.update({f'source_position.{k}': tensor_digest(v) for k, v in source_position.named_buffers()})
    audit = dict(unchanged=before == after, fingerprints=after, saw_opacity_gradient=saw_gradient,
                 immutable_reference_cache=reference_cache.stats(),
                 saw_opacity_update=saw_update, saw_ray_depth_gradient=saw_depth_gradient,
                 saw_footprint_gradient=saw_scale_gradient, saw_position_gradient=saw_position_gradient,
                 saw_source_opacity_update=saw_source_opacity_update)
    if source_position is not None:
        audit['source_position'] = source_position.audit()
        audit['source_position']['saw_gradient'] = saw_source_position_gradient
    (a.output/'frozen_parameter_audit.json').write_text(json.dumps(audit, indent=2))
    if before != after or (a.candidate_gain and not (saw_gradient and saw_update)):
        raise RuntimeError('Candidate causal/frozen audit failed')
    if a.joint_persistent_optics and a.source_opacity_lr > 0 and not saw_source_opacity_update: raise RuntimeError('Source optics never updated')
    if a.candidate_gain and a.ray_log_radius and not saw_depth_gradient: raise RuntimeError('Ray depth never received a gradient')
    if a.candidate_gain and (a.footprint_log_radius or a.spatial_footprint_log_radius) and not saw_scale_gradient: raise RuntimeError('Footprint never received a gradient')
    if a.candidate_gain and a.position_radius_sigmas and not saw_position_gradient: raise RuntimeError('Position never received a gradient')
    if source_position is not None and (not saw_source_position_gradient or not audit['source_position']['finite']
            or not audit['source_position']['moved_rows'] or audit['source_position']['maximum_bound_fraction'] > 1.000001):
        raise RuntimeError('Source-position causal/bound audit failed')


if __name__ == '__main__': main()
