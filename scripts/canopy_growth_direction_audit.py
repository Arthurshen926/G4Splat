"""Current-state training-only size derivatives, not geometric truth or a fix."""
import json
from pathlib import Path
import torch


def summarize_growth(tree, risk, code):
    if tree.shape != risk.shape or tree.shape != code.shape:
        raise ValueError('Aligned per-candidate derivatives required')
    if not all(torch.isfinite(x).all() for x in (tree, risk, code)):
        raise ValueError('Finite derivatives required')
    # A positive derivative means decreasing the shared scale code reduces the
    # chosen loss locally. This is not an Adam update or finite-change guarantee.
    expanded = code > 0
    conflict = expanded & (tree < 0) & (risk > 0)
    jointly_shrink = expanded & (tree > 0) & (risk > 0)
    return dict(expanded=int(expanded.sum()),
                expansion_conflict=int(conflict.sum()),
                joint_local_shrink=int(jointly_shrink.sum()),
                tree_gradient_l1=float(tree.abs().sum()),
                risk_gradient_l1=float(risk.abs().sum()))


def audit_growth(teacher, original, candidate, adapter, views, masks, manifest,
                 output, *, snapshot_sha256):
    from outdoor.hybrid_gaussian_renderer import render_hybrid, static_detail_forward_visibility_gate
    from scripts.canopy_observed_background_risk import observed_rgb_risk
    from outdoor.moge3_evidence import sha256_file
    training = manifest['training_views']
    if (not training or len(set(training)) != len(training)
            or set(training) & set(manifest['excluded_views'])
            or manifest['args'].get('leaf_optical_kernel', 'native') != 'native'
            or manifest['args']['candidate_gain'] != 1):
        raise ValueError('Unique training-only views and enabled native candidates required')
    parameter = candidate.scale_code
    before = parameter.detach().clone()
    old_requires_grad = parameter.requires_grad
    totals = [torch.zeros_like(parameter, dtype=torch.float64) for _ in range(2)]
    counts = [torch.zeros_like(parameter, dtype=torch.int32) for _ in range(2)]
    common = dict(background=torch.zeros(3, device=parameter.device), include_dynamic=False,
                  optical_replacement_policy='disabled', structural_trainable_start=None)
    try:
        parameter.requires_grad_(True)
        for ordinal, index in enumerate(training):
            view = views[index]
            obj, ns, dist, nt = masks.get_index_masks(view.image_name, (0, 1, 2, 3),
                (view.image_height, view.image_width), parameter.device)
            tree = obj & ns & dist & ~nt
            rigid = obj & ns & dist & nt
            with torch.no_grad():
                gate = static_detail_forward_visibility_gate(original, int(view.colmap_id),
                                                             include_pending_exact=False)
                sky = teacher.sky(view)
                target = view.original_image.to(parameter.device)
                ref = render_hybrid(view, teacher.surface, original, volume_gate=gate, **common)
                reference = ref.render + (1 - ref.alpha) * sky
                del ref
            with torch.enable_grad():
                out = render_hybrid(view, teacher.surface, adapter,
                    volume_gate=torch.cat((gate, gate.new_ones(len(candidate.xyz)))), **common)
                rgb = out.render + (1 - out.alpha) * sky
                tree_loss = ((rgb - target).abs().mean(0) * tree).sum() / tree.sum().clamp_min(1)
                risk_loss = observed_rgb_risk(rgb, target, reference, rigid)
                for j, loss in enumerate((tree_loss, risk_loss)):
                    gradient, = torch.autograd.grad(loss, parameter, retain_graph=(j == 0))
                    totals[j] += gradient.detach().double()
                    counts[j] += gradient.detach().ne(0)
                del out, rgb, tree_loss, risk_loss
            if ordinal % 25 == 0:
                print(json.dumps(dict(growth_training_views=ordinal + 1)), flush=True)
    finally:
        parameter.requires_grad_(old_requires_grad)
        if not torch.equal(parameter.detach(), before):
            raise RuntimeError('Read-only growth audit changed candidate scales')
    totals = [x / len(training) for x in totals]
    report = dict(scope='training_only_current_state_derivatives__not_optimization_or_truth',
        snapshot_sha256=snapshot_sha256, helper_sha256=sha256_file(Path(__file__)),
        training_views=training, excluded_views=manifest['excluded_views'],
        summary=summarize_growth(*totals, before),
        losses=['per-view mean tree L1', 'per-view mean rigid RGB risk above original-source error'],
        limitations=['Not the full training objective or Adam direction',
                     'Local derivatives need finite-change multi-view verification',
                     'No evaluation pixels were used to compute these signals'])
    torch.save(dict(tree_gradient=totals[0].cpu(), risk_gradient=totals[1].cpu(),
                    tree_nonzero_views=counts[0].cpu(), risk_nonzero_views=counts[1].cpu(),
                    scale_code=before.cpu(), report=report), output / 'growth_directions.pth')
    (output / 'growth_directions.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report['summary']), flush=True)
