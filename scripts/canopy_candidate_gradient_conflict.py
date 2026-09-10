"""Read-only training-gradient balance at a fixed diagnostic state."""
import json
import torch
from outdoor.hybrid_gaussian_renderer import render_hybrid, static_detail_forward_visibility_gate


def summarize_gradient_balance(components, weights, adam_first_moment=None):
    denominator = weights.sum().clamp_min(1e-12)
    means = {name: float((value*weights).sum()/denominator) for name, value in components.items()}
    net = components['canopy_positive']-components['canopy_negative']+components['background_positive']-components['background_negative']
    means.update(evaluation_weight=float(weights.sum()),
                 opacity_increase_net_weight_fraction=float((weights*(net < -1e-12)).sum()/denominator),
                 opacity_decrease_net_weight_fraction=float((weights*(net > 1e-12)).sum()/denominator))
    if adam_first_moment is not None:
        if adam_first_moment.shape != net.shape or not torch.isfinite(adam_first_moment).all():
            raise ValueError('Aligned finite candidate Adam moment required')
        means.update(last_adam_moment_increase_weight_fraction=float((weights*(adam_first_moment < -1e-12)).sum()/denominator),
                     net_gradient_opposes_last_adam_moment_weight_fraction=float((weights*(
                         ((net > 1e-12)&(adam_first_moment < -1e-12))|
                         ((net < -1e-12)&(adam_first_moment > 1e-12)))).sum()/denominator))
    return means


def candidate_adam_first_moment(payload):
    optimizer = payload.get('optimizer_state'); layout = payload.get('optimizer_parameter_layout')
    if optimizer is None or layout is None:
        return None
    if len(layout) != len(optimizer['param_groups']): raise ValueError('Invalid optimizer layout')
    matches = []
    for names, group in zip(layout, optimizer['param_groups']):
        if len(names) != len(group['params']): raise ValueError('Invalid optimizer parameter mapping')
        for name, key in zip(names, group['params']):
            if name.startswith('candidate.') and name.endswith('.logits'):
                state = optimizer['state'].get(key, {})
                if 'exp_avg' in state: matches.append(state['exp_avg'])
    if len(matches) != 1: raise ValueError('Exactly one saved candidate opacity moment required')
    return matches[0]


def training_gradient_conflict(teacher, base, adapter, candidate, views, masks, manifest, roi_weights, output,
                              adam_first_moment=None):
    if any(manifest['args'].get(key, 0.) for key in
           ('rigid_boundary_preservation_weight', 'surface_rgb_feasibility_weight')):
        raise ValueError('Gradient-balance audit does not yet include these auxiliary losses')
    if manifest['args'].get('candidate_gain', 1) != 1:
        raise ValueError('Enabled candidates required; audit must not turn disabled candidates on')
    if manifest['args']['noncanopy_rgb_target'] != 'source_reference':
        raise ValueError('Explicit immutable background ownership required')
    if any(p.requires_grad for p in adapter.base.parameters()):
        raise ValueError('Source optical state must be frozen during the candidate-only audit')
    for name in ('_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest'):
        getattr(teacher.surface, name).requires_grad_(False)
    for p in teacher.sky.parameters(): p.requires_grad_(False)
    components = {name: torch.zeros_like(candidate.logits) for name in
                  ('canopy_positive', 'canopy_negative', 'background_positive', 'background_negative')}
    original_logits = candidate.logits.detach().clone()
    original_requires_grad = candidate.logits.requires_grad
    guard_weight = 1+manifest['args']['rigid_rgb_preservation_weight']
    inactive_momentum = []
    common = dict(background=torch.zeros(3, device='cuda'), include_dynamic=False,
                  optical_replacement_policy='disabled', structural_trainable_start=None)
    try:
        candidate.logits.requires_grad_(True)
        with torch.enable_grad():
            for ordinal, index in enumerate(manifest['training_views']):
                v = views[index]
                obj, nonsky, dist, nontree = masks.get_index_masks(v.image_name, (0, 1, 2, 3),
                    (v.image_height, v.image_width), torch.device('cuda'))
                canopy = obj & nonsky & dist & ~nontree
                rigid = obj & nonsky & dist & nontree
                sky = obj & dist & ~nonsky
                gate = static_detail_forward_visibility_gate(base, int(v.colmap_id), include_pending_exact=False)
                with torch.no_grad():
                    original = render_hybrid(v, teacher.surface, base, volume_gate=gate, **common)
                    sky_rgb = teacher.sky(v)
                    reference = original.render+(1-original.alpha)*sky_rgb
                updated = render_hybrid(v, teacher.surface, adapter,
                    volume_gate=torch.cat((gate, gate.new_ones((len(candidate.xyz),)))), **common)
                rgb = updated.render+(1-updated.alpha)*sky_rgb
                canopy_loss = ((rgb-v.original_image.cuda()).abs().mean(0)*canopy).sum()/canopy.sum().clamp_min(1)
                background_error = (rgb-reference).abs().mean(0)
                background_loss = guard_weight*(background_error*rigid).sum()/rigid.sum().clamp_min(1)
                background_loss = background_loss+(background_error*sky).sum()/sky.sum().clamp_min(1)
                net_view_gradient = torch.zeros_like(candidate.logits)
                for name, loss, retain in [('canopy', canopy_loss, True), ('background', background_loss, False)]:
                    grad, = torch.autograd.grad(loss, candidate.logits, retain_graph=retain)
                    if not torch.isfinite(grad).all(): raise RuntimeError('Nonfinite optical gradient')
                    components[name+'_positive'] += grad.detach().clamp_min(0)
                    components[name+'_negative'] += (-grad.detach()).clamp_min(0)
                    net_view_gradient += grad.detach()
                if adam_first_moment is not None:
                    paused = net_view_gradient == 0
                    stale = paused & (adam_first_moment != 0)
                    inactive_momentum.append(dict(index=index,
                        zero_gradient_rows=int(paused.sum()), zero_gradient_with_saved_momentum_rows=int(stale.sum()),
                        roi_weight_fractions={str(key): float((weight*stale).sum()/weight.sum().clamp_min(1e-12))
                                              for key, weight in roi_weights.items()}))
                del original, updated, rgb, canopy_loss, background_loss
                if ordinal % 25 == 0:
                    print(json.dumps(dict(gradient_training_views=ordinal+1)), flush=True)
    finally:
        candidate.logits.requires_grad_(original_requires_grad)
    if not torch.equal(original_logits, candidate.logits):
        raise RuntimeError('Read-only gradient audit modified opacity')
    result = dict(scope='fixed_state_candidate_opacity_gradient_only__evaluation_weights_are_not_optimization_authority',
                  train_views=len(manifest['training_views']), gradient_sign='positive decreases opacity; negative increases it',
                  source_optics_refined=bool(manifest['args'].get('joint_persistent_optics', False)),
                  adam_moment_available=adam_first_moment is not None,
                  source_position_refined=bool(manifest['args'].get('source_position_radius_sigmas', 0.)),
                  zero_gradient_saved_momentum=inactive_momentum,
                  zero_gradient_warning='Each training view replayed at ONE FIXED final state and SAME saved moment; not the actual historical update sequence or a visibility oracle',
                  warning='Last saved Adam moment is not the next step or a proof of an optimizer bug; other parameters are fixed in this audit',
                  total_rigid_weight=guard_weight, unchanged_opacity=True,
                  regions={str(key): summarize_gradient_balance(components, value, adam_first_moment) for key, value in roi_weights.items()})
    torch.save({name: value.cpu() for name, value in components.items()}, output/'training_gradient_components.pth')
    return result
