"""Fixed-state, training-only signals for native-replayed prefix candidates.

No parameter updates. LP-selected identities are diagnostic selections, never
opacity targets. Visibility counts below are measured at ONE fixed state, not
historical effective-update counts.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def summarize_signals(rows):
    total = np.asarray([r['total_gradient'] for r in rows], dtype=np.float64)
    floor = np.asarray([r['weighted_floor_gradient'] for r in rows], dtype=np.float64)
    mass = np.asarray([r['responsibility'] for r in rows], dtype=np.float64)
    if total.ndim != 2 or floor.shape != total.shape or mass.shape != total.shape+(4,):
        raise ValueError('Aligned per-view gradients and native responsibility required')
    if not all(np.isfinite(a).all() for a in (total, floor, mass)):
        raise ValueError('Finite fixed-state signals required')
    return dict(total_gradient_sum=total.sum(0).tolist(),
                weighted_floor_gradient_sum=floor.sum(0).tolist(),
                rgb_and_guard_gradient_sum=(total-floor).sum(0).tolist(),
                increasing_gradient_views=(total < 0).sum(0).tolist(),
                decreasing_gradient_views=(total > 0).sum(0).tolist(),
                nonzero_gradient_views=(total != 0).sum(0).tolist(),
                contributing_views=(mass[:, :, 0] > 1e-5).sum(0).tolist(),
                regional_responsibility_sum=mass.sum(0).tolist())


def audit_selected_signal(teacher, base, adapter, candidate, views, masks, payload,
                          solution_dir, output, *, snapshot_sha256):
    from outdoor.hybrid_gaussian_renderer import render_hybrid, static_detail_forward_visibility_gate
    from scripts.canopy_declared_training_objective import declared_objective
    from scripts.canopy_surface_rgb_feasibility import BlackRadianceView
    from scripts.canopy_candidate_gradient_conflict import candidate_adam_first_moment
    from scripts.evaluate_hybrid_teacher import _tree_boundary_masks
    from scripts.canopy_surface_rgb_feasibility import surface_rgb_feasibility_loss
    from scripts.canopy_boundary_preservation import boundary_rgb_preservation
    manifest=payload['manifest']; args=SimpleNamespace(**manifest['args'])
    if getattr(args,'visible_rigid_alpha_weight',0):
        raise ValueError('Selected-gradient probe does not reconstruct the visible-wall reference; refusing an incomplete objective')
    info=json.loads((Path(solution_dir)/'report.json').read_text())
    constraints=json.loads((Path(info['source'])/'joint_feasibility.json').read_text())
    if (constraints['snapshot_sha256'] != snapshot_sha256
            or constraints['source_checkpoint_sha256'] != manifest['source_checkpoint_sha256']
            or getattr(args,'leaf_optical_kernel','native') != 'native'
            or args.candidate_gain != 1 or args.canopy_only_color_gradients):
        raise ValueError('Exact native scene and declared full RGB recipe required')
    data=np.load(Path(solution_dir)/'joint_solution.npz')
    ids=data['primitive_ids'][(data['tau']-data['current_tau']) > 1e-6]
    if not len(ids) or (ids < len(base)).any() or (ids >= len(adapter)).any():
        raise ValueError('This read-only audit selects increased candidate rows only')
    if set(manifest['training_views']) & set(manifest['excluded_views']):
        raise ValueError('Excluded cameras cannot supply gradient evidence')
    local=torch.as_tensor(ids-len(base),device='cuda',dtype=torch.long)
    for name in ('_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest'):
        getattr(teacher.surface,name).requires_grad_(False)
    for p in teacher.sky.parameters():p.requires_grad_(False)
    original=candidate.logits.detach().clone(); old_flag=candidate.logits.requires_grad
    black=BlackRadianceView(adapter)
    common=dict(background=torch.zeros(3,device='cuda'),include_dynamic=False,
                optical_replacement_policy='disabled',structural_trainable_start=None)
    records=[]
    try:
        candidate.logits.requires_grad_(True)
        for ordinal,index in enumerate(manifest['training_views']):
            v=views[index]
            obj,ns,dist,nt=masks.get_index_masks(v.image_name,(0,1,2,3),
                (v.image_height,v.image_width),torch.device('cuda'))
            tree=obj&ns&dist&~nt; rigid=obj&ns&dist&nt; sky=obj&dist&~ns
            _,boundary,_=_tree_boundary_masks(~tree)
            regions=dict(tree=tree,rigid=rigid,hard=rigid&boundary,sky=sky)
            gate=static_detail_forward_visibility_gate(base,int(v.colmap_id),include_pending_exact=False)
            fullgate=torch.cat((gate,torch.ones(len(candidate.xyz),device='cuda')))
            with torch.no_grad():
                source=render_hybrid(v,teacher.surface,base,volume_gate=gate,**common)
                reference=source.render+(1-source.alpha)*teacher.sky(v)
                target=v.original_image.cuda()
                del source
            with torch.enable_grad():
                out=render_hybrid(v,teacher.surface,adapter,volume_gate=fullgate,
                    audit_fields=torch.stack((tree,rigid,sky)).float(),**common)
                rgb=out.render+(1-out.alpha)*teacher.sky(v)
                floor_rgb=render_hybrid(v,teacher.surface,black,volume_gate=fullgate,**common).render
                scales=manifest.get('relative_floor_initial_scales',{})
                total,_,floor=declared_objective(rgb,target,reference,regions,args,
                    floor_rgb=floor_rgb,relative_scale=scales.get(index,scales.get(str(index),1.)))
                gf=(torch.autograd.grad(args.surface_rgb_feasibility_weight*floor,candidate.logits,retain_graph=True)[0]
                    if args.surface_rgb_feasibility_weight else torch.zeros_like(candidate.logits))
                tree_loss=((rgb-target).abs().mean(0)*tree).sum()/tree.sum().clamp_min(1)
                tree_grad=torch.autograd.grad(tree_loss,candidate.logits,retain_graph=True)[0]
                # Explicit additional counterfactual signal, NOT included in
                # the saved recipe's loss if its auxiliary weight was zero.
                potential_floor=surface_rgb_feasibility_loss(floor_rgb,target,tree)
                potential_grad=torch.autograd.grad(potential_floor,candidate.logits,retain_graph=True)[0]
                error=(rgb-target).abs().mean(0)
                gt_background=((1+args.rigid_rgb_preservation_weight)*(error*rigid).sum()/rigid.sum().clamp_min(1)
                               +(error*sky).sum()/sky.sum().clamp_min(1))
                if args.rigid_boundary_preservation_weight:
                    gt_background=gt_background+args.rigid_boundary_preservation_weight*boundary_rgb_preservation(
                        rgb,target,rigid,regions['hard'])
                gt_background_grad=torch.autograd.grad(gt_background,candidate.logits,retain_graph=True)[0]
                gt=torch.autograd.grad(total,candidate.logits)[0]
            mass=out.responsibility[len(teacher.surface.get_xyz)+len(base):][local]
            row=dict(view=index,loss=float(total),total_gradient=gt[local].cpu().tolist(),
                weighted_floor_gradient=gf[local].cpu().tolist(),responsibility=mass.cpu().tolist(),
                tree_rgb_gradient=tree_grad[local].cpu().tolist(),
                background_and_guard_gradient=(gt-gf-tree_grad)[local].cpu().tolist(),
                counterfactual_background_GT_gradient=gt_background_grad[local].cpu().tolist(),
                counterfactual_unit_absolute_floor_gradient=potential_grad[local].cpu().tolist())
            records.append(row)
            del out,rgb,floor_rgb,total,floor,gf,gt,tree_loss,tree_grad,potential_floor,potential_grad,error,gt_background,gt_background_grad
            if ordinal%25==0:print(json.dumps(dict(selected_signal_views=ordinal+1)),flush=True)
    finally:
        candidate.logits.requires_grad_(old_flag)
    if not torch.equal(original,candidate.logits):raise RuntimeError('Read-only audit changed candidate opacity')
    moment=candidate_adam_first_moment(payload)
    components={key:np.asarray([row[key] for row in records],dtype=np.float64).sum(0).tolist()
        for key in ('tree_rgb_gradient','background_and_guard_gradient','counterfactual_unit_absolute_floor_gradient',
                    'counterfactual_background_GT_gradient')}
    report=dict(scope='fixed_state_full_training_objective__no_updates__not_historical_observation_counts',
        solution=str(solution_dir),snapshot_sha256=snapshot_sha256,primitive_ids=ids.tolist(),
        candidate_opacity=original[local].sigmoid().cpu().tolist(),
        saved_first_moment=(moment[local.cpu()].tolist() if moment is not None else None),
        summary=summarize_signals(records),components=components,records=records,
        saved_auxiliary_weight=args.surface_rgb_feasibility_weight,
        limitations=['Only LP-increased candidates from eight training rays; not all canopy geometry',
                    'Positive derivative means decreasing opacity; no optimizer step is performed',
                    'Counterfactual GT background derivatives test reference-target bias; they grant no noncanopy foliage authority',
                    'Conditional sparse feasibility does not establish safety across every image'])
    (output/'selected_optical_signal.json').write_text(json.dumps(report,indent=2))
