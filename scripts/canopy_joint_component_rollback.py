"""Temporary leave-one-component-out diagnostics; never a fitted/exported model.

Differences are conditional on all other learned components, not additive causal
credits. The context restores every touched tensor, including on render failure.
"""
from contextlib import contextmanager
import torch

MODES = ('full', 'source_dc_original', 'source_sh_original',
         'source_opacity_original', 'source_position_original',
         'candidates_disabled', 'candidate_position_initial', 'candidate_size_initial')
SHAPE_MODES = ('candidate_isotropic_code_initial', 'candidate_shape_initial')


@contextmanager
def component_rollback(mode, original, refined, source_position, candidate):
    if mode not in MODES + SHAPE_MODES:
        raise ValueError('Explicit leave-one-component-out mode required')
    if refined is original or source_position is None:
        raise ValueError('Separate source optics and static position required')
    if mode in SHAPE_MODES and not hasattr(candidate, 'shape_code'):
        raise ValueError('Shape decomposition requires anisotropic candidates')
    changes = {
        'source_dc_original': (refined.features[:, :1], original.features[:, :1]),
        'source_sh_original': (refined.features[:, 1:], original.features[:, 1:]),
        'source_opacity_original': (refined.opacity_logits, original.opacity_logits),
        'source_position_original': (source_position.code, None),
        'candidate_position_initial': (candidate.position_code, None),
        'candidate_size_initial': (candidate.scale_code, None),
    }
    changed = changes.get(mode)
    changes_to_apply = [] if changed is None else [changed]
    if mode == 'candidate_size_initial' and hasattr(candidate, 'shape_code'):
        changes_to_apply.append((candidate.shape_code, None))
    elif mode == 'candidate_isotropic_code_initial':
        changes_to_apply = [(candidate.scale_code, None)]
    elif mode == 'candidate_shape_initial':
        changes_to_apply = [(candidate.shape_code, None)]
    saved = []
    try:
        for tensor, replacement in changes_to_apply:
            saved.append((tensor, tensor.detach().clone()))
            with torch.no_grad():
                if replacement is None:
                    tensor.zero_()
                else:
                    tensor.copy_(replacement)
        yield 0. if mode == 'candidates_disabled' else 1.
    finally:
        for tensor, snapshot in saved:
            with torch.no_grad():
                tensor.copy_(snapshot)
            if not torch.equal(tensor, snapshot):
                raise RuntimeError('Read-only component rollback failed to restore state')


@torch.no_grad()
def audit_joint_components(teacher, original, refined, source_position, candidate,
                           adapter, views, masks, manifest, renderer, output, *, snapshot_sha256,
                           growth_evidence=None):
    import json
    from outdoor.hybrid_gaussian_renderer import static_detail_forward_visibility_gate
    from scripts.evaluate_hybrid_teacher import _tree_boundary_masks
    from scripts.canopy_source_optics_components import region_psnr
    from scripts.canopy_observed_background_risk import conservative_visible_rigid_mask
    from outdoor.moge3_evidence import sha256_file
    from pathlib import Path
    records = []
    modes = MODES + (SHAPE_MODES if hasattr(candidate, 'shape_code') else ())
    growth_contract = None
    if growth_evidence is not None:
        from scripts.canopy_growth_counterfactual import eligible_shrink, shrink_probe
        data = torch.load(growth_evidence, map_location='cpu', weights_only=True)
        selection = eligible_shrink(data, candidate.scale_code, snapshot_sha256,
                                    manifest['training_views'], manifest['excluded_views'])
        modes = ('full', 'joint_shrink_quarter', 'joint_shrink_initial')
        growth_contract = dict(evidence_sha256=sha256_file(growth_evidence),
            intervention_helper_sha256=sha256_file(Path(__file__).with_name('canopy_growth_counterfactual.py')),
            selected_count=int(selection.sum()),
            criteria='expanded shared code; both net derivatives positive; each nonzero in >=2 training views',
            warning='Local derivative sign is not a guarantee for either finite step or regression views')
    eval_views = manifest['excluded_views']
    if set(eval_views) & set(manifest['training_views']):
        raise ValueError('Regression views must remain excluded from training')
    common = dict(background=torch.zeros(3, device='cuda'), include_dynamic=False,
                  optical_replacement_policy='disabled', structural_trainable_start=None)
    # These historical crops are evaluation only, never training evidence.
    crops = {660: (470,205,630,335), 657: (420,170,610,270), 690: (40,110,160,250)}
    for ordinal, index in enumerate(eval_views):
        v = views[index]
        obj, ns, dist, nt = masks.get_index_masks(v.image_name, (0,1,2,3),
            (v.image_height,v.image_width), torch.device('cuda'))
        tree = obj & ns & dist & ~nt; rigid = obj & ns & dist & nt
        inner, outer, _ = _tree_boundary_masks(~tree)
        regions = dict(tree=tree, tree_interior=tree & ~inner, tree_boundary=tree & inner,
                       rigid=rigid, hard=rigid & outer, sky=obj & dist & ~ns)
        if index in crops:
            x0,y0,x1,y1 = crops[index]; roi = torch.zeros_like(tree)
            roi[y0:y1,x0:x1] = tree[y0:y1,x0:x1]; regions['leakage_roi'] = roi
        gate = static_detail_forward_visibility_gate(original, int(v.colmap_id), include_pending_exact=False)
        background = teacher.sky(v); target = v.original_image.cuda()
        from outdoor.hybrid_gaussian_renderer import render_hybrid
        reference = render_hybrid(v,teacher.surface,original,volume_gate=gate,**common)
        reference_rgb = reference.render+(1-reference.alpha)*background
        guard = conservative_visible_rigid_mask(reference_rgb,target,reference.surface_alpha,regions)
        def measure(package):
            rgb = package.render+(1-package.alpha)*background
            drop = (reference.surface_alpha-package.surface_alpha).reshape_as(tree).clamp_min(0)
            return dict(psnr=region_psnr(rgb,target,regions),
                surface_alpha={k:float(package.surface_alpha.reshape_as(tree)[m].mean()) if m.any() else None
                               for k,m in regions.items()},
                visible_guard_pixels=int(guard.sum()),
                visible_guard_mean_drop=float(drop[guard].mean()) if guard.any() else None)
        result = dict(index=index, original=measure(reference), modes={})
        for mode in modes:
            context = (component_rollback(mode,original,refined,source_position,candidate)
                       if growth_evidence is None else shrink_probe(candidate.scale_code,selection,
                           {'full': 0., 'joint_shrink_quarter': .25, 'joint_shrink_initial': 1.}[mode]))
            with context as gain:
                package = renderer(v,teacher.surface,adapter,
                    volume_gate=torch.cat((gate,gate.new_full((len(candidate.xyz),),gain))),**common)
                result['modes'][mode] = measure(package)
                if growth_evidence is not None and index in (660, 657, 690, 738, 674):
                    from PIL import Image
                    rgb = package.render + (1 - package.alpha) * background
                    pair = torch.cat((target, rgb), dim=2).clamp(0, 1)
                    pixels = (pair.permute(1, 2, 0) * 255).round().byte().cpu().numpy()
                    Image.fromarray(pixels).save(output / f'view_{index}_{mode}.png')
                del package
        del reference
        records.append(result)
        if ordinal % 8 == 0:
            print(json.dumps(dict(component_views=ordinal+1)),flush=True)
    report = dict(scope='read_only_leave_one_component_out__nonadditive__evaluation_not_supervision',
        growth_contract=growth_contract,
        checkpoint_sha256=snapshot_sha256, helper_sha256=sha256_file(Path(__file__)),
        training_views=manifest['training_views'],excluded_views=eval_views,
        warning='Restored initial/reference components are counterfactuals, not accepted repairs or geometric certificates',
        records=records)
    (output/'joint_components.json').write_text(json.dumps(report,indent=2))
