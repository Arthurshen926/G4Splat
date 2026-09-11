"""Read-only one-step counterfactuals from saved late-stage optimizer state."""
import copy
import json
from types import SimpleNamespace
import torch
from outdoor.hybrid_gaussian_renderer import render_hybrid, persistent_static_evidence_mask, static_detail_forward_visibility_gate
from scripts.canopy_candidate_joint_optics import JointOpticalCandidateView
from scripts.canopy_actual_update_audit import ActualUpdateAudit
from scripts.canopy_declared_training_objective import declared_objective
from scripts.canopy_surface_rgb_feasibility import BlackRadianceView
from scripts.canopy_candidate_ray_refinement import project_candidate_dc_
from scripts.canopy_directional_sh_step import rescale_directional_sh_step_
from scripts.canopy_source_opacity_projection import project_source_opacity_
from scripts.evaluate_hybrid_teacher import _tree_boundary_masks


def probe_saved_updates(teacher,base,adapter,candidate,views,masks,payload,output):
    args=SimpleNamespace(**payload['manifest']['args']); manifest=payload['manifest']
    if getattr(args,'visible_rigid_alpha_weight',0):
        raise ValueError('Visible-wall recipes require the trainer exact-continuation audit with its immutable alpha reference')
    from pathlib import Path
    from outdoor.moge3_evidence import sha256_file
    if manifest.get('source_opacity_projection_helper_sha256') != sha256_file(Path(__file__).with_name('canopy_source_opacity_projection.py')):
        raise ValueError('Saved update replay requires the original opacity postprocessing implementation')
    if (getattr(args,'optimizer_policy','adam')!='adam' or not args.joint_persistent_optics or args.gradient_accumulation!=1
            or args.canopy_only_color_gradients or args.candidate_gain!=1):
        raise ValueError('Only the matched static joint full-objective single-view recipe is supported')
    torch.cuda.set_per_process_memory_fraction(.42)
    refined=adapter.base;eligible=persistent_static_evidence_mask(base)&base.static_leaf_mask
    for p in candidate.parameters():p.requires_grad_(True)
    for key in ('features','opacity_logits'):getattr(refined,key).requires_grad_(True)
    position=getattr(refined,'position',None)
    if position is not None:position.code.requires_grad_(True)
    joint=JointOpticalCandidateView(refined,candidate,eligible)
    parameters={'candidate.'+k:p for k,p in candidate.named_parameters()}
    parameters.update({'source.'+k:p for k,p in refined.named_parameters() if p.requires_grad})
    if position is not None:parameters['source_position.code']=position.code
    layout=payload['optimizer_parameter_layout'];saved=payload['optimizer_state']
    if len(layout)!=len(saved['param_groups']) or set(sum(layout,[]))!=set(parameters):
        raise ValueError('Saved optimizer row/parameter identities do not match live state')
    groups=[]
    for names,group in zip(layout,saved['param_groups']):
        groups.append(dict(**{k:v for k,v in group.items() if k!='params'},params=[parameters[k] for k in names]))
    optimizer=torch.optim.Adam(groups,eps=1e-15)
    initial={k:p.detach().cpu().clone() for k,p in parameters.items()}
    black=BlackRadianceView(joint)
    common=dict(background=torch.zeros(3,device='cuda'),include_dynamic=False,optical_replacement_policy='disabled',structural_trainable_start=None)
    order=payload['training_order'];chosen=[order[(payload['step']+k)%len(order)] for k in range(8)]
    if set(chosen)&set(manifest['excluded_views']):raise ValueError('Training views only')
    records=[]
    try:
        for index in chosen:
            with torch.no_grad():
                for k,p in parameters.items():p.copy_(initial[k])
            optimizer.load_state_dict(copy.deepcopy(saved));optimizer.zero_grad(set_to_none=True)
            v=views[index];obj,ns,dist,nt=masks.get_index_masks(v.image_name,(0,1,2,3),(v.image_height,v.image_width),torch.device('cuda'))
            tree=obj&ns&dist&~nt;rigid=obj&ns&dist&nt;_,boundary,_=_tree_boundary_masks(~tree)
            regions=dict(tree=tree,rigid=rigid,hard=rigid&boundary,sky=obj&dist&~ns)
            gate=static_detail_forward_visibility_gate(base,int(v.colmap_id),include_pending_exact=False)
            fullgate=torch.cat((gate,torch.ones(len(candidate.xyz),device='cuda')))
            with torch.no_grad():
                original=render_hybrid(v,teacher.surface,base,volume_gate=gate,**common)
                reference=original.render+(1-original.alpha)*teacher.sky(v);target=v.original_image.cuda()
            def objective():
                out=render_hybrid(v,teacher.surface,joint,volume_gate=fullgate,**common)
                rgb=out.render+(1-out.alpha)*teacher.sky(v)
                floor=(render_hybrid(v,teacher.surface,black,volume_gate=fullgate,**common).render
                       if args.surface_rgb_feasibility_weight else None)
                scales=manifest.get('relative_floor_initial_scales',{})
                scale=scales.get(index,scales.get(str(index),1.))
                return declared_objective(rgb,target,reference,regions,args,floor_rgb=floor,relative_scale=scale)[0]
            with torch.enable_grad():
                loss=objective();loss.backward()
            audit=ActualUpdateAudit(parameters.items());before=float(loss.detach())
            previous_rest=refined.features[eligible,1:].detach().clone()
            optimizer.step();raw=audit.measure()
            with torch.no_grad():
                if args.candidate_color_domain=='unit_rgb':project_candidate_dc_(candidate.dc)
                rescale_directional_sh_step_(refined.features,previous_rest,eligible,args.source_rest_step_multiplier)
                project_source_opacity_(refined.opacity_logits,eligible,enabled=args.source_opacity_lr>0)
                if args.source_rest_step_multiplier:
                    rest=refined.features[eligible,1:];factor=(4./rest.flatten(1).norm(dim=1).clamp_min(1e-12)).clamp_max(1.)
                    refined.features[eligible,1:]=rest*factor[:,None,None]
                final=audit.measure();after=float(objective())
            row=dict(view=index,loss_before=before,loss_after=after,loss_delta=after-before,adam=raw,after_postprocessing=final)
            records.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ['adam','after_postprocessing']}),flush=True)
            del audit,loss,previous_rest
    finally:
        with torch.no_grad():
            for k,p in parameters.items():p.copy_(initial[k]);p.grad=None;p.requires_grad_(False)
        for hook in joint.optical_hooks:hook.remove()
    report=dict(scope='same_saved_state_independent_one_step_counterfactuals__not_resume_or_production',
        saved_step=payload['step'],learning_rates=[g['lr'] for g in saved['param_groups']],
        source_reference_immutable=True,records=records,
        limitations=['Eight fixed-state training-view probes, not all-view objective descent',
                     'Uses saved final learning rates and moments; not a new LR schedule or full training continuation'])
    (output/'saved_actual_updates.json').write_text(json.dumps(report,indent=2))
