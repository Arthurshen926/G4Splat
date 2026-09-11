"""Read-only native contribution audit of unverified RGB candidates.

Evaluation-camera contribution is NOT training evidence or a promotion rule.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'2d-gaussian-splatting')]
import torch
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import render_hybrid, static_detail_forward_visibility_gate
from outdoor.moge3_evidence import sha256_file
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.canopy_candidate_diagnostic import CandidateCloud, NativeCandidateView
from scripts.evaluate_canopy_validation import FIXED, ADDITIONAL
from scripts.evaluate_hybrid_teacher import _tree_boundary_masks
from scripts.canopy_rgb_region_audit import rgb_region_audit
from scripts.canopy_rgb_feasibility import color_feasibility, nonnegative_color_floor
from scripts.canopy_black_radiance_probe import BlackRadianceCandidateView
from scripts.canopy_candidate_gradient_conflict import training_gradient_conflict
from scripts.canopy_optical_capacity import weighted_opacity_summary, weighted_size_summary
from scripts.canopy_depth_search_counterfactual import reseed_stratified_depth_
from scripts.canopy_source_optics_components import source_optics_components, region_psnr

# Previously used difficult-view crops; evaluation only, never fitting masks.
DENSE_ROIS = {657: (470, 40, 630, 200), 660: (390, 45, 610, 195),
              690: (180, 120, 440, 280), 713: (30, 100, 150, 280),
              768: (90, 210, 170, 270)}
# Additional visually identified failure regions, retained alongside the old
# interior crops. Evaluation only: never used for fitting or leaf truth.
LEAKAGE_ROIS = {657: (420, 170, 610, 270), 660: (470, 205, 630, 335),
                690: (40, 110, 160, 250)}


def scaled_regression_roi(box,width,height):
    return tuple(round(value*scale) for value,scale in zip(box,(width/640,height/360,width/640,height/360)))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__); model = ModelParams(p)
    p.set_defaults(data_device='cpu', resolution=640, white_background=True)
    for key in ('run', 'output'): p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--save-rgb-pairs',action='store_true')
    p.add_argument('--rgb-feasibility-probe', action='store_true')
    p.add_argument('--persistent-black-floor-probe', action='store_true')
    p.add_argument('--building-floor-probe', action='store_true')
    p.add_argument('--training-gradient-conflict-probe', action='store_true')
    p.add_argument('--joint-feasibility-probe', action='store_true',
                   help='Training-only conditional sparse optical feasibility; no model updates')
    p.add_argument('--joint-feasibility-rays', type=int, default=8)
    p.add_argument('--joint-feasibility-violated-only', action='store_true')
    p.add_argument('--joint-feasibility-risk-solution', type=Path)
    p.add_argument('--joint-feasibility-all-training-views', action='store_true')
    p.add_argument('--joint-feasibility-replay', type=Path,
                   help='Read-only native verification of an explicitly supplied LP solution')
    p.add_argument('--joint-replay-kernel',choices=('native','projected_tau'),default='projected_tau',
                   help='Explicit LP replay image model; historical replays used experimental projected_tau')
    p.add_argument('--saved-actual-update-probe', action='store_true')
    p.add_argument('--training-selectivity-cohort', action='store_true')
    p.add_argument('--selected-optical-signal', type=Path,
                   help='Read-only full-training gradients for exact prefix-LP increased candidates')
    p.add_argument('--surface-prefix-replay', type=Path,
                   help='Native surface-prefix diagnosis of a completed joint_native_replay.json')
    p.add_argument('--surface-prefix-constraints', type=Path)
    p.add_argument('--joint-native-prefix', type=Path,
                   help='Build multi-depth necessary constraints from a completed surface_prefix.json')
    p.add_argument('--roi-optical-capacity-probe', action='store_true')
    p.add_argument('--persistent-optical-ceiling-probe', action='store_true',
                   help='Read-only near-opaque persistent leaves plus candidates; never training targets')
    p.add_argument('--optical-ceiling-opacity', type=float, choices=(.99, 1.), default=.99,
                   help='Read-only probe peak; 1 uses finite float32 logit20, never infinite parameters')
    p.add_argument('--stratified-depth-min-ratio-counterfactual', type=float)
    p.add_argument('--source-optics-component-probe', action='store_true')
    p.add_argument('--joint-component-rollback', action='store_true',
                   help='Read-only leave-one-component-out analysis with exact tensor restoration')
    p.add_argument('--growth-direction-audit', action='store_true',
                   help='Training-only current size derivatives; no parameter updates')
    p.add_argument('--growth-counterfactual', type=Path,
                   help='Finite read-only size probes from matching training-only growth directions')
    p.add_argument('--coverage-ceiling-probe', action='store_true',
                   help='Read-only .99 candidate-opacity counterfactual; NEVER a train target or fix')
    p.add_argument('--intrinsic-coverage-probe', action='store_true',
                   help='Also separate missing projected candidates from candidates occluded by the scene')
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    audit_script_hash = sha256_file(Path(__file__))
    rgb_region_helper_hash = sha256_file(Path(__file__).with_name('canopy_rgb_region_audit.py'))
    feasibility_helper_hash = sha256_file(Path(__file__).with_name('canopy_rgb_feasibility.py'))
    black_radiance_helper_hash = sha256_file(Path(__file__).with_name('canopy_black_radiance_probe.py'))
    gradient_helper_hash = sha256_file(Path(__file__).with_name('canopy_candidate_gradient_conflict.py'))
    capacity_helper_hash = sha256_file(Path(__file__).with_name('canopy_optical_capacity.py'))
    reseed_helper_hash = sha256_file(Path(__file__).with_name('canopy_depth_search_counterfactual.py'))
    source_components_helper_hash = sha256_file(Path(__file__).with_name('canopy_source_optics_components.py'))
    if a.intrinsic_coverage_probe and not a.coverage_ceiling_probe:
        raise ValueError('Intrinsic coverage requires the read-only ceiling probe')
    if a.persistent_optical_ceiling_probe and not a.coverage_ceiling_probe:
        raise ValueError('Persistent counterfactual requires the explicit candidate ceiling probe')
    if a.optical_ceiling_opacity != .99 and not a.coverage_ceiling_probe:
        raise ValueError('Peak limit is only meaningful in the explicit read-only counterfactual')
    torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(.25)
    payload_path = a.run/f'candidates_{a.step:04d}.pth'
    # A render audit must not allocate every saved Adam tensor on the GPU.
    payload = torch.load(payload_path, map_location='cpu', weights_only=False)
    manifest = payload['manifest']; args = manifest['args']
    if args.get('moge_position_weight',0) and (a.saved_actual_update_probe or a.selected_optical_signal is not None or a.training_gradient_conflict_probe):
        raise ValueError('Offline optimizer probe does not yet reconstruct MoGe position loss; use in-training full objective audit')
    if args.get('candidate_optical_coordinate','legacy_logit')!='legacy_logit':
        if a.saved_actual_update_probe or a.selected_optical_signal is not None or a.training_gradient_conflict_probe:
            raise ValueError('Optical-coordinate offline optimizer probes require coordinate-aware replay; use the saved in-training actual-update audit')
    if args.get('visible_rigid_alpha_weight',0):
        if manifest.get('observed_background_risk_helper_sha256')!=sha256_file(Path(__file__).with_name('canopy_observed_background_risk.py')):
            raise ValueError('Visible-rigid audit policy changed')
    audit_renderer = render_hybrid
    kernel = args.get('leaf_optical_kernel', 'native')
    if kernel == 'projected_tau':
        if a.training_gradient_conflict_probe or a.saved_actual_update_probe:
            raise ValueError('Saved-gradient probes are not yet kernel-aware; refusing native-kernel reinterpretation')
        from scripts.audit_canopy_optical_depth_kernel import independent_render
        audit_renderer, renderer_hash = independent_render()
        import canopy_optical_depth_rasterization as independent_native
        if (renderer_hash != manifest.get('experimental_renderer_hash')
                or sha256_file(Path(independent_native._C.__file__)) != manifest.get('experimental_binary_sha256')
                or sha256_file(Path(__file__).with_name('audit_canopy_optical_depth_kernel.py')) != manifest.get('kernel_adapter_helper_sha256')):
            raise ValueError('Experimental checkpoint renderer provenance changed')
    elif kernel != 'native':
        raise ValueError('Unknown checkpoint optical kernel')
    if (not payload['diagnostic_only'] or payload['step'] != a.step
            or manifest['helper_sha256'] != sha256_file(Path(__file__).with_name('canopy_candidate_diagnostic.py'))
            or sha256_file(args['checkpoint']) != manifest['source_checkpoint_sha256']
            or sha256_file(args['masks']) != manifest['masks_sha256']):
        raise ValueError('Immutable diagnostic source and helper required')
    if Path(a.source_path).resolve() != Path(args['source_path']).resolve(): raise ValueError('Changed dataset')
    teacher = load_hybrid_teacher(args['checkpoint'], sh_degree=a.sh_degree); base = teacher.foliage
    for parameter in base.parameters(): parameter.requires_grad_(False)
    state = payload['candidate']; ray_radius = args.get('ray_log_radius', 0.)
    footprint_radius = args.get('footprint_log_radius', 0.)
    position_radius = args.get('position_radius_sigmas', 0.)
    prefix = 'cloud.' if ray_radius or footprint_radius or position_radius else ''
    candidate = CandidateCloud(state[prefix+'xyz'], state[prefix+'scales'], torch.zeros_like(state[prefix+'xyz']))
    if ray_radius:
        from scripts.canopy_candidate_ray_refinement import RayDepthCandidates
        if manifest['refinement_helper_sha256'] != sha256_file(Path(__file__).with_name('canopy_candidate_ray_refinement.py')):
            raise ValueError('Changed ray refinement helper')
        candidate = RayDepthCandidates(candidate, state['origins'], ray_radius)
    if footprint_radius:
        from scripts.canopy_candidate_footprint import FootprintCandidates
        if manifest['footprint_helper_sha256'] != sha256_file(Path(__file__).with_name('canopy_candidate_footprint.py')):
            raise ValueError('Changed footprint helper')
        candidate = FootprintCandidates(candidate, footprint_radius)
    if position_radius:
        from scripts.canopy_candidate_spatial import SpatialCandidates
        if manifest['spatial_helper_sha256'] != sha256_file(Path(__file__).with_name('canopy_candidate_spatial.py')):
            raise ValueError('Changed spatial helper')
        candidate = SpatialCandidates(candidate, position_radius)
        if args.get('spatial_footprint_log_radius', 0.):
            from scripts.canopy_spatial_footprint import SpatialFootprintCandidates
            if manifest.get('spatial_footprint_helper_sha256') != sha256_file(Path(__file__).with_name('canopy_spatial_footprint.py')):
                raise ValueError('Changed joint spatial footprint helper')
            candidate = SpatialFootprintCandidates(candidate.cloud, position_radius,
                                                   args['spatial_footprint_log_radius'])
    if args.get('candidate_shape_cohort') is not None:
        from scripts.canopy_selective_shape import load_shape_cohort,SelectiveSpatialFootprintCandidates
        if (manifest.get('candidate_shape_helper_sha256')!=sha256_file(Path(__file__).with_name('canopy_selective_shape.py'))
                or manifest.get('candidate_shape_cohort_sha256')!=sha256_file(args['candidate_shape_cohort'])):
            raise ValueError('Selective shape implementation or cohort changed')
        shape_eligible=load_shape_cohort(args['candidate_shape_cohort'],candidate.cloud,
            manifest['source_checkpoint_sha256'],manifest['training_views'],manifest['excluded_views'])
        candidate=SelectiveSpatialFootprintCandidates(candidate.cloud,shape_eligible,position_radius,
                                                       args['spatial_footprint_log_radius'])
        if args.get('candidate_orientation_control',False):
            from scripts.canopy_selective_orientation import OrientedSelectiveCandidates
            if manifest.get('candidate_orientation_helper_sha256')!=sha256_file(Path(__file__).with_name('canopy_selective_orientation.py')):
                raise ValueError('Candidate orientation implementation changed')
            candidate=OrientedSelectiveCandidates(candidate.cloud,shape_eligible,position_radius,
                args['spatial_footprint_log_radius'])
    candidate.load_state_dict(state)
    candidate = candidate.cuda()
    ceiling_logit = (candidate.logits.new_tensor(20.) if a.optical_ceiling_opacity == 1.
                     else torch.logit(candidate.logits.new_tensor(.99)))
    for parameter in candidate.parameters(): parameter.requires_grad_(False)
    refined_base = base
    source_optics = payload.get('source_optics')
    if bool(args.get('joint_persistent_optics', False)) != (source_optics is not None):
        raise ValueError('Explicit joint source-optics payload required')
    if source_optics is not None:
        from outdoor.hybrid_gaussian_renderer import VolumetricFoliageModel, persistent_static_evidence_mask
        if manifest['joint_optics_helper_sha256'] != sha256_file(Path(__file__).with_name('canopy_candidate_joint_optics.py')):
            raise ValueError('Changed joint optical authority helper')
        refined_base = VolumetricFoliageModel(a.sh_degree, dynamic_rank=base.dynamic_rank).cuda()
        refined_base.restore(teacher.state['foliage'])
        eligible = persistent_static_evidence_mask(base)&base.static_leaf_mask
        if set(source_optics) != {'features', 'opacity_logits'}:
            raise ValueError('Only source colors and opacity may be refined')
        for key, value in source_optics.items():
            target = getattr(refined_base, key); value = value.to(target)
            if target.shape != value.shape or not torch.isfinite(value).all() or not torch.equal(target[~eligible], value[~eligible]):
                raise ValueError('Invalid or unauthorized source optical delta')
            target.copy_(value)
        for p_refined in refined_base.parameters(): p_refined.requires_grad_(False)
    source_position = None
    source_position_radius = args.get('source_position_radius_sigmas', 0.)
    if bool(source_position_radius) != (payload.get('source_position') is not None):
        raise ValueError('Explicit source-position state must match its authorized configuration')
    if source_position_radius:
        from scripts.canopy_static_source_position import StaticSourcePosition, StaticSourcePositionView
        if (source_optics is None or a.source_optics_component_probe
                or manifest.get('source_position_helper_sha256') != sha256_file(Path(__file__).with_name('canopy_static_source_position.py'))):
            raise ValueError('Source-position audit requires matching helper; optical-only component probe is unsupported')
        source_position = StaticSourcePosition(base.xyz, base.scales, eligible, source_position_radius)
        source_position.load_verified_state(payload['source_position'])
        source_position.code.requires_grad_(False)
        refined_base = StaticSourcePositionView(refined_base, source_position)
    adapter = NativeCandidateView(refined_base, candidate)
    black_adapter = BlackRadianceCandidateView(refined_base, candidate) if a.building_floor_probe else None
    if a.persistent_black_floor_probe or a.persistent_optical_ceiling_probe:
        from outdoor.hybrid_gaussian_renderer import persistent_static_evidence_mask
        black_eligible = persistent_static_evidence_mask(refined_base)&refined_base.static_leaf_mask
    candidate_gain = float(args.get('candidate_gain', 1))
    if a.source_optics_component_probe and (source_optics is None or candidate_gain != 0):
        raise ValueError('Source component swaps require the source-optics-only control')
    dataset = model.extract(a); dataset.model_path = str(a.output)
    views = LazyScene(dataset, GaussianModel(dataset.sh_degree), image_cache_size=4).getTrainCameras()
    depth_counterfactual = None
    if a.stratified_depth_min_ratio_counterfactual is not None:
        if (not a.coverage_ceiling_probe or a.training_gradient_conflict_probe
                or args.get('depth_search_policy') != 'stratified_volume'
                or ray_radius or footprint_radius or position_radius):
            raise ValueError('Only unrefined stratified geometry may receive this read-only capacity probe')
        lo, hi = manifest['depth_search_contract']['bounds']
        depth_counterfactual = reseed_stratified_depth_(candidate, views, manifest['seed_audit'],
            old_low=lo, new_low=a.stratified_depth_min_ratio_counterfactual, high=hi)
    masks = CambridgeMaskLookup(Path(dataset.source_path), args['masks'])
    holdout = FIXED+ADDITIONAL
    if set(holdout) != set(manifest['excluded_views']) or set(holdout)&set(manifest['training_views']):
        raise ValueError('All 48 cameras must remain excluded from fitting')
    if a.growth_direction_audit:
        from scripts.canopy_growth_direction_audit import audit_growth
        audit_growth(teacher,base,candidate,adapter,views,masks,manifest,a.output,
                     snapshot_sha256=sha256_file(payload_path))
        return
    if a.joint_component_rollback or a.growth_counterfactual is not None:
        if source_optics is None or source_position is None or candidate_gain != 1:
            raise ValueError('Joint optics, positions and enabled candidates required')
        from scripts.canopy_joint_component_rollback import audit_joint_components
        audit_joint_components(teacher,base,refined_base,source_position,candidate,adapter,
            views,masks,manifest,audit_renderer,a.output,snapshot_sha256=sha256_file(payload_path),
            growth_evidence=a.growth_counterfactual)
        return
    if a.training_selectivity_cohort:
        from scripts.canopy_candidate_selectivity_audit import audit_selectivity
        audit_selectivity(teacher,base,adapter,candidate,views,masks,manifest,a.output,
                         snapshot_sha256=sha256_file(payload_path))
        return
    if a.joint_native_prefix is not None:
        if a.surface_prefix_constraints is None:raise ValueError('Prior constraints required')
        from scripts.canopy_prefix_joint_feasibility import prefix_joint_feasibility
        prefix_joint_feasibility(base,adapter,candidate,views,manifest,a.surface_prefix_constraints,a.joint_native_prefix,a.output,
                                 snapshot_sha256=sha256_file(payload_path))
        return
    if a.selected_optical_signal is not None:
        from scripts.canopy_selected_optical_signal import audit_selected_signal
        audit_selected_signal(teacher,base,adapter,candidate,views,masks,payload,
                              a.selected_optical_signal,a.output,snapshot_sha256=sha256_file(payload_path))
        return
    if a.surface_prefix_replay is not None:
        if a.surface_prefix_constraints is None:
            raise ValueError('Explicit prior prefix constraints required')
        from scripts.canopy_native_surface_prefix_probe import probe_surface_prefix
        probe_surface_prefix(teacher,base,views,manifest,a.surface_prefix_constraints,a.surface_prefix_replay,a.output,
                             snapshot_sha256=sha256_file(payload_path))
        return
    if a.joint_feasibility_probe:
        from scripts.canopy_joint_scene_feasibility import audit_joint_scene
        audit_joint_scene(teacher, base, adapter, candidate, views, masks, manifest, a.output,
                         rays_per_kind=a.joint_feasibility_rays,violated_only=a.joint_feasibility_violated_only,
                         snapshot_sha256=sha256_file(payload_path),risk_solution=a.joint_feasibility_risk_solution,
                         all_training_views=a.joint_feasibility_all_training_views)
        return
    if a.joint_feasibility_replay is not None:
        from scripts.canopy_joint_solution_replay import replay_joint_solution
        replay_joint_solution(teacher,base,adapter,candidate,views,masks,manifest,a.joint_feasibility_replay,a.output,
                              snapshot_sha256=sha256_file(payload_path),kernel=a.joint_replay_kernel)
        return
    if a.saved_actual_update_probe:
        from scripts.canopy_saved_update_probe import probe_saved_updates
        probe_saved_updates(teacher,base,adapter,candidate,views,masks,payload,a.output)
        return
    mass = torch.zeros(len(candidate.xyz), 4, device='cuda')
    visible_views = torch.zeros(len(candidate.xyz), dtype=torch.int32, device='cuda')
    records = []
    roi_weights = {}
    for ordinal, index in enumerate(holdout):
        v = views[index]
        obj, sky, dist, tree = masks.get_index_masks(v.image_name, (0, 1, 2, 3),
            (v.image_height, v.image_width), torch.device('cuda'))
        canopy = obj&sky&dist&~tree; rigid = obj&sky&dist&tree; sky_mask = obj&dist&~sky
        gate = static_detail_forward_visibility_gate(base, int(v.colmap_id), include_pending_exact=False)
        common = dict(background=torch.zeros(3, device='cuda'), include_dynamic=False,
                      optical_replacement_policy='disabled', structural_trainable_start=None)
        original = render_hybrid(v, teacher.surface, base, volume_gate=gate, **common)
        updated = audit_renderer(v, teacher.surface, adapter,
            volume_gate=torch.cat((gate, gate.new_full((len(candidate.xyz),), candidate_gain))),
            audit_fields=torch.stack((canopy, rigid, sky_mask)).float(), **common)
        visible_rigid_risk=None
        if args.get('visible_rigid_alpha_weight',0):
            from scripts.canopy_observed_background_risk import conservative_visible_rigid_mask
            _,risk_boundary,_=_tree_boundary_masks(~canopy)
            reference_rgb=original.render+(1-original.alpha)*teacher.sky(v)
            risk_mask=conservative_visible_rigid_mask(reference_rgb,v.original_image.cuda(),original.surface_alpha,
                dict(rigid=rigid,hard=rigid&risk_boundary))
            drop=(original.surface_alpha-updated.surface_alpha).reshape_as(rigid).clamp_min(0)
            visible_rigid_risk=dict(pixels=int(risk_mask.sum()),
                mean_surface_contribution_drop=float(drop[risk_mask].mean()) if risk_mask.any() else None,
                maximum_surface_contribution_drop=float(drop[risk_mask].max()) if risk_mask.any() else None,
                fraction_drop_above_001=float((drop[risk_mask]>.01).float().mean()) if risk_mask.any() else None,
                scope='conditional_conservative_visible_mask__evaluation_only')
        contribution = updated.responsibility[len(teacher.surface.get_xyz)+len(base):]
        if contribution.shape != mass.shape or not torch.isfinite(contribution).all() or (contribution < 0).any():
            raise RuntimeError('Invalid native candidate contribution')
        mass += contribution; visible_views += contribution[:, 1] > .001
        roi_optics = None
        if (a.training_gradient_conflict_probe or a.roi_optical_capacity_probe) and index in LEAKAGE_ROIS:
            x0, y0, x1, y1 = scaled_regression_roi(LEAKAGE_ROIS[index],v.image_width,v.image_height)
            roi = torch.zeros_like(canopy); roi[y0:y1, x0:x1] = canopy[y0:y1, x0:x1]
            probe = audit_renderer(v, teacher.surface, adapter,
                volume_gate=torch.cat((gate, gate.new_full((len(candidate.xyz),), candidate_gain))),
                audit_fields=torch.stack((roi, rigid, sky_mask)).float(), **common)
            roi_weights[index] = probe.responsibility[len(teacher.surface.get_xyz)+len(base):, 1].clone()
            source_opacity = refined_base.conditioned_state(None, include_dynamic=False)[2]
            roi_optics = dict(candidate=weighted_opacity_summary(candidate.logits.sigmoid(), roi_weights[index]),
                existing=weighted_opacity_summary(source_opacity,
                    probe.responsibility[len(teacher.surface.get_xyz):len(teacher.surface.get_xyz)+len(base), 1]))
            if position_radius:
                ratio = (candidate.scales/candidate.cloud.scales).log().mean(1).exp()
                roi_optics['candidate_size'] = weighted_size_summary(ratio, roi_weights[index])
                if args.get('candidate_shape_cohort') is not None:
                    axis_ratio=candidate.scales/candidate.cloud.scales
                    equivalent=axis_ratio.log().mean(1).exp()
                    roi_optics['candidate_size']=weighted_size_summary(equivalent,roi_weights[index])
                    roi_optics['candidate_size']['scope']='actual_ROI_weighted_geometric_mean_axis_ratio__anisotropic_not_geometry_truth'
                    weights=roi_weights[index];denominator=weights.sum().clamp_min(1e-12)
                    roi_optics['candidate_size']['weighted_aspect_ratio']=float(
                        ((candidate.scales.max(1).values/candidate.scales.min(1).values)*weights).sum()/denominator)
            del probe
        def region_mean(tensor, mask): return float(tensor.reshape_as(mask)[mask].mean()) if mask.any() else None
        records.append(dict(index=index, visible_rigid_risk=visible_rigid_risk,candidate_canopy_mass=float(contribution[:, 1].sum()),
            candidate_rigid_mass=float(contribution[:, 2].sum()), candidate_sky_mass=float(contribution[:, 3].sum()),
            canopy_surface_alpha_before=region_mean(original.surface_alpha, canopy),
            canopy_surface_alpha_after=region_mean(updated.surface_alpha, canopy),
            rigid_surface_alpha_before=region_mean(original.surface_alpha, rigid),
            rigid_surface_alpha_after=region_mean(updated.surface_alpha, rigid)))
        if roi_optics is not None: records[-1]['roi_peak_opacity'] = roi_optics
        inner_tree_band, outer_tree_band, radius = _tree_boundary_masks(~canopy)
        background = teacher.sky(v)
        source_rgb = (original.render+(1-original.alpha)*background).clamp(0, 1)
        updated_rgb = (updated.render+(1-updated.alpha)*background).clamp(0, 1)
        target = v.original_image.cuda()
        records[-1]['image_shape']=[v.image_height,v.image_width]
        all_regions=dict(tree=canopy,tree_interior=canopy&~inner_tree_band,
                         tree_boundary=canopy&inner_tree_band,rigid=rigid,hard=rigid&outer_tree_band)
        records[-1]['rgb_regions']=dict(source=region_psnr(source_rgb,target,all_regions),
                                       updated=region_psnr(updated_rgb,target,all_regions))
        if a.save_rgb_pairs:
            panel=torch.cat((target,source_rgb,updated_rgb),2).permute(1,2,0).cpu().clamp(0,1).numpy()
            Image.fromarray((panel*255).round().astype('uint8')).save(a.output/f'view_{index}.png')
        records[-1]['rigid_rgb_regions'] = rgb_region_audit(source_rgb, updated_rgb, target, rigid, outer_tree_band)
        records[-1]['sky_rgb'] = dict(pixels=int(sky_mask.sum()),
            source_psnr=region_psnr(source_rgb,target,dict(sky=sky_mask))['sky'],
            updated_psnr=region_psnr(updated_rgb,target,dict(sky=sky_mask))['sky'],
            source_volume_alpha=region_mean(original.volume_alpha,sky_mask),
            updated_volume_alpha=region_mean(updated.volume_alpha,sky_mask),
            scope='fixed_view_sky_regression__not_positive_foliage_supervision')
        records[-1]['tree_interface_radius_pixels'] = radius
        if a.source_optics_component_probe:
            component_regions = dict(tree=canopy, tree_interior=canopy & ~inner_tree_band,
                tree_boundary=canopy & inner_tree_band, rigid=rigid, hard=rigid & outer_tree_band)
            def component_render():
                component = audit_renderer(v, teacher.surface, adapter,
                    volume_gate=torch.cat((gate, gate.new_zeros((len(candidate.xyz),)))), **common)
                return component.render+(1-component.alpha)*background
            parts = source_optics_components(base, refined_base, component_render, target, component_regions)
            parts['original_source'] = region_psnr(source_rgb, target, component_regions)
            parts['joint_source'] = region_psnr(updated_rgb, target, component_regions)
            if (component_render().clamp(0, 1)-updated_rgb).abs().max() > 2e-6:
                raise RuntimeError('Component probe did not restore the refined native render')
            records[-1]['source_optics_components'] = parts
        if black_adapter is not None:
            # All leaf radiance and sky are absent, but ALL actual opacity,
            # positions and ordering remain. No model parameter is modified.
            probe = audit_renderer(v, teacher.surface, black_adapter,
                volume_gate=torch.cat((gate, gate.new_full((len(candidate.xyz),), candidate_gain))), **common)
            if (probe.alpha-updated.alpha).abs().max() > 2e-5:
                raise RuntimeError('Black-radiance adapter changed optical weights')
            floor_regions = {'canopy': canopy}
            if index in LEAKAGE_ROIS:
                x0, y0, x1, y1 = scaled_regression_roi(LEAKAGE_ROIS[index],v.image_width,v.image_height)
                roi = torch.zeros_like(canopy); roi[y0:y1, x0:x1] = canopy[y0:y1, x0:x1]
                floor_regions['supplemental_leakage_roi'] = roi
            current_raw = updated.render+(1-updated.alpha)*background
            floor_records = {}
            for name, mask in floor_regions.items():
                value = nonnegative_color_floor(probe.render, current_raw, target, mask,metric_domain='clamped_unit_rgb')
                value['raw_domain'] = nonnegative_color_floor(probe.render,current_raw,target,mask)
                value['scope'] = 'unaltered_surface_radiance_through_actual_opacity__all_foliage_and_sky_radiance_zero__read_only'
                floor_records[name] = value
            records[-1]['building_radiance_floor'] = floor_records
            del probe
        if a.persistent_black_floor_probe:
            saved_dc = candidate.dc.clone()
            saved_features = refined_base.features[black_eligible].clone()
            try:
                candidate.dc.fill_(-.5/.28209479177387814)
                refined_base.features[black_eligible] = 0
                refined_base.features[black_eligible, 0] = -.5/.28209479177387814
                probe = audit_renderer(v, teacher.surface, adapter,
                    volume_gate=torch.cat((gate, gate.new_full((len(candidate.xyz),), candidate_gain))), **common)
                black_rgb = probe.render+(1-probe.alpha)*background
                current_raw = updated.render+(1-updated.alpha)*background
                floor_regions = {'canopy': canopy}
                if index in LEAKAGE_ROIS:
                    x0, y0, x1, y1 = scaled_regression_roi(LEAKAGE_ROIS[index],v.image_width,v.image_height)
                    roi = torch.zeros_like(canopy); roi[y0:y1, x0:x1] = canopy[y0:y1, x0:x1]
                    floor_regions['supplemental_leakage_roi'] = roi
                records[-1]['persistent_black_floor'] = {name: nonnegative_color_floor(
                    black_rgb, current_raw, target, mask,metric_domain='clamped_unit_rgb') for name, mask in floor_regions.items()}
                del probe
            finally:
                candidate.dc.copy_(saved_dc)
                refined_base.features[black_eligible] = saved_features
        if a.rgb_feasibility_probe:
            saved_dc = candidate.dc.clone()
            try:
                endpoints = []
                for color in (0., 1.):
                    candidate.dc.fill_((color-.5)/.28209479177387814)
                    probe = audit_renderer(v, teacher.surface, adapter,
                        volume_gate=torch.cat((gate, gate.new_full((len(candidate.xyz),), candidate_gain))), **common)
                    endpoints.append(probe.render+(1-probe.alpha)*background)
                    del probe
                current_raw = updated.render+(1-updated.alpha)*background
                probe_regions = {'canopy': canopy, 'rigid': rigid}
                if index in LEAKAGE_ROIS:
                    x0, y0, x1, y1 = scaled_regression_roi(LEAKAGE_ROIS[index],v.image_width,v.image_height)
                    roi = torch.zeros_like(canopy); roi[y0:y1, x0:x1] = canopy[y0:y1, x0:x1]
                    probe_regions['supplemental_leakage_roi'] = roi
                records[-1]['color_feasibility'] = {name: color_feasibility(
                    endpoints[0], endpoints[1], current_raw, target, mask) for name, mask in probe_regions.items()}
                del endpoints
            finally:
                candidate.dc.copy_(saved_dc)
        for roi_name, roi_map in (('preset_dense_roi', DENSE_ROIS), ('supplemental_leakage_roi', LEAKAGE_ROIS)):
            if index not in roi_map:
                continue
            x0, y0, x1, y1 = scaled_regression_roi(roi_map[index],v.image_width,v.image_height)
            roi = torch.zeros_like(canopy); roi[y0:y1, x0:x1] = canopy[y0:y1, x0:x1]
            def psnr(rgb):
                return float(-10*(rgb[:, roi]-target[:, roi]).square().mean().clamp_min(1e-12).log10()) if roi.any() else None
            records[-1][roi_name] = dict(xyxy=[x0,y0,x1,y1], canopy_pixels=int(roi.sum()),
                source_psnr=psnr(source_rgb), updated_psnr=psnr(updated_rgb),
                source_surface_alpha=region_mean(original.surface_alpha, roi),
                updated_surface_alpha=region_mean(updated.surface_alpha, roi),
                source_volume_alpha=region_mean(original.volume_alpha, roi),
                updated_volume_alpha=region_mean(updated.volume_alpha, roi))
        if a.coverage_ceiling_probe:
            saved_logits = candidate.logits.clone()
            try:
                candidate.logits.fill_(ceiling_logit)
                ceiling = audit_renderer(v, teacher.surface, adapter,
                    volume_gate=torch.cat((gate, gate.new_full((len(candidate.xyz),), candidate_gain))), **common)
                blocked = canopy & (original.surface_alpha[0] > .25)
                records[-1]['coverage_ceiling'] = dict(
                    probe_peak_opacity=a.optical_ceiling_opacity,
                    canopy_surface_alpha=region_mean(ceiling.surface_alpha, canopy),
                    initial_high_surface_canopy_pixels=int(blocked.sum()),
                    high_surface_pixels_remaining=int((blocked & (ceiling.surface_alpha[0] > .25)).sum()),
                    high_surface_pixels_reduced_below_01=int((blocked & (ceiling.surface_alpha[0] < .1)).sum()),
                    scope='geometric_coverage_counterfactual_not_physical_leaf_truth')
                if index in LEAKAGE_ROIS:
                    x0, y0, x1, y1 = scaled_regression_roi(LEAKAGE_ROIS[index],v.image_width,v.image_height)
                    roi = torch.zeros_like(canopy); roi[y0:y1, x0:x1] = canopy[y0:y1, x0:x1]
                    records[-1]['coverage_ceiling']['supplemental_roi_surface_alpha'] = region_mean(ceiling.surface_alpha, roi)
                    records[-1]['coverage_ceiling']['supplemental_roi_volume_alpha'] = region_mean(ceiling.volume_alpha, roi)
                if a.persistent_optical_ceiling_probe:
                    saved_persistent_logits = refined_base.opacity_logits[black_eligible].clone()
                    try:
                        near_opaque = ceiling_logit
                        candidate.logits.copy_(torch.maximum(saved_logits, near_opaque))
                        refined_base.opacity_logits[black_eligible] = torch.maximum(saved_persistent_logits, near_opaque)
                        persistent_ceiling = audit_renderer(v, teacher.surface, adapter,
                            volume_gate=torch.cat((gate, gate.new_full((len(candidate.xyz),), candidate_gain))), **common)
                        ceiling_record = dict(canopy_surface_alpha=region_mean(persistent_ceiling.surface_alpha, canopy),
                            canopy_volume_alpha=region_mean(persistent_ceiling.volume_alpha, canopy),
                            probe_peak_opacity=a.optical_ceiling_opacity,
                            scope='read_only_peak_opacity_counterfactual__existing_verified_persistent_and_candidates__not_a_fix')
                        if index in LEAKAGE_ROIS:
                            ceiling_record['supplemental_roi_surface_alpha'] = region_mean(persistent_ceiling.surface_alpha, roi)
                            ceiling_record['supplemental_roi_volume_alpha'] = region_mean(persistent_ceiling.volume_alpha, roi)
                            black_probe = audit_renderer(v, teacher.surface, BlackRadianceCandidateView(refined_base, candidate),
                                volume_gate=torch.cat((gate, gate.new_full((len(candidate.xyz),), candidate_gain))), **common)
                            ceiling_record['supplemental_roi_building_floor'] = nonnegative_color_floor(
                                black_probe.render, persistent_ceiling.render+(1-persistent_ceiling.alpha)*background,
                                target, roi,metric_domain='clamped_unit_rgb')
                            ceiling_record['supplemental_roi_building_floor']['scope'] = 'surface_only_floor_at_near_opaque_persistent_and_candidate_counterfactual'
                            del black_probe
                        records[-1]['persistent_optical_ceiling'] = ceiling_record
                        del persistent_ceiling
                    finally:
                        refined_base.opacity_logits[black_eligible] = saved_persistent_logits
                        candidate.logits.fill_(ceiling_logit)
                if a.intrinsic_coverage_probe:
                    intrinsic = audit_renderer(v, teacher.surface, adapter,
                        surface_gate=torch.zeros(len(teacher.surface.get_xyz), device='cuda'),
                        volume_gate=torch.cat((torch.zeros_like(gate), gate.new_full((len(candidate.xyz),), candidate_gain))), **common)
                    alpha = intrinsic.volume_alpha[0]
                    records[-1]['coverage_ceiling'].update(
                        missing_projected_coverage_pixels=int((blocked & (alpha < .25)).sum()),
                        intrinsically_dense_pixels=int((blocked & (alpha >= .95)).sum()),
                        intrinsically_dense_but_wall_remains=int((blocked & (alpha >= .95)
                                                                  & (ceiling.surface_alpha[0] > .25)).sum()))
                    if index in LEAKAGE_ROIS:
                        records[-1]['coverage_ceiling'].update(
                            supplemental_roi_missing_projected_pixels=int((roi & (alpha < .25)).sum()),
                            supplemental_roi_intrinsically_dense_pixels=int((roi & (alpha >= .95)).sum()),
                            supplemental_roi_dense_but_wall_remains=int((roi & (alpha >= .95)
                                & (ceiling.surface_alpha[0] > .25)).sum()))
                    if index in (657, 660, 690, 713, 768, 909):
                        np.savez_compressed(a.output/f'coverage_{index}.npz', canopy=canopy.cpu().numpy(),
                            source_surface_alpha=original.surface_alpha[0].cpu().numpy(),
                            ceiling_surface_alpha=ceiling.surface_alpha[0].cpu().numpy(),
                            intrinsic_candidate_alpha=alpha.cpu().numpy())
                    del intrinsic
                del ceiling
            finally:
                candidate.logits.copy_(saved_logits)
        if ordinal % 8 == 0: print(json.dumps(dict(completed_views=ordinal+1)), flush=True)
    opacity = candidate.logits.sigmoid()
    gradient_report = None
    if a.training_gradient_conflict_probe:
        from scripts.canopy_candidate_gradient_conflict import candidate_adam_first_moment
        moment = candidate_adam_first_moment(payload)
        if moment is not None: moment = moment.to(candidate.logits)
        gradient_report = training_gradient_conflict(teacher, base, adapter, candidate, views, masks,
                                                     manifest, roi_weights, a.output, moment)
    summary = dict(candidates=len(opacity), opacity_above_01=int((opacity > .01).sum()),
        opacity_above_1=int((opacity > .1).sum()), mean_opacity=float(opacity.mean()),
        canopy_contributing_in_two_eval_views=int((visible_views >= 2).sum()),
        canopy_mass=float(mass[:, 1].sum()), rigid_mass=float(mass[:, 2].sum()), sky_mass=float(mass[:, 3].sum()))
    torch.save(dict(contribution_mass=mass.cpu(), evaluation_canopy_view_count=visible_views.cpu()),
               a.output/'contribution.pth')
    report = dict(scope='evaluation_only__not_verified_support_or_promotion_authority',
                  leaf_optical_kernel=kernel,
                  experimental_binary_sha256=manifest.get('experimental_binary_sha256'),
                  source_optics_refined=source_optics is not None,
                  source_position=(source_position.audit() if source_position is not None else None),
                  source_position_helper_sha256=manifest.get('source_position_helper_sha256'),
                  checkpoint_sha256=sha256_file(payload_path), audit_script_sha256=audit_script_hash,
                  rgb_region_helper_sha256=rgb_region_helper_hash,
                  rgb_feasibility_helper_sha256=feasibility_helper_hash,
                  black_radiance_helper_sha256=black_radiance_helper_hash,
                  training_gradient_helper_sha256=gradient_helper_hash,
                  optical_capacity_helper_sha256=capacity_helper_hash,
                  depth_counterfactual_helper_sha256=reseed_helper_hash,
                  depth_search_counterfactual=depth_counterfactual,
                  source_components_helper_sha256=source_components_helper_hash,
                  training_gradient_conflict=gradient_report,
                  summary=summary, records=records)
    (a.output/'audit.json').write_text(json.dumps(report, indent=2)); print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
