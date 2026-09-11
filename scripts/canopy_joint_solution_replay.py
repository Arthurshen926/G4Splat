"""Native verification of a conditional LP counterfactual; never exports a map."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scripts.canopy_surface_rgb_feasibility import BlackRadianceView
from outdoor.hybrid_gaussian_renderer import static_detail_forward_visibility_gate, persistent_static_evidence_mask
from scripts.evaluate_hybrid_teacher import _tree_boundary_masks


@torch.no_grad()
def replay_joint_solution(teacher,base,adapter,candidate,views,masks,manifest,solution_dir,output,*,snapshot_sha256,kernel='projected_tau'):
    solution_dir=Path(solution_dir); info=json.loads((solution_dir/'report.json').read_text())
    constraints=json.loads((Path(info['source'])/'joint_feasibility.json').read_text())
    if constraints.get('snapshot_sha256') != snapshot_sha256:
        raise ValueError('LP constraints must bind the exact candidate/geometry checkpoint')
    solution=np.load(solution_dir/'joint_solution.npz')
    ids=torch.from_numpy(solution['primitive_ids']).cuda(); tau=torch.from_numpy(solution['tau']).cuda()
    if (ids.dtype!=torch.int64 or ids.ndim!=1 or tau.shape!=ids.shape or not torch.isfinite(tau).all()
            or (tau<0).any() or ids.unique().numel()!=ids.numel() or (ids<0).any()
            or (ids>=len(base)+len(candidate.xyz)).any()):
        raise ValueError('Invalid LP primitive identities or optical depths')
    old_source=adapter.base.opacity_logits.detach().clone();old_candidate=candidate.logits.detach().clone()
    all_logits=torch.cat((old_source.reshape(-1),old_candidate.reshape(-1)))
    if not np.allclose(torch.nn.functional.softplus(all_logits[ids]).cpu().numpy(),solution['current_tau'],rtol=2e-5,atol=2e-6):
        raise ValueError('LP state does not match the restored checkpoint')
    eligible=torch.cat((persistent_static_evidence_mask(base)&base.static_leaf_mask,torch.ones(len(candidate.xyz),device='cuda',dtype=torch.bool)))
    if not eligible[ids].all():raise ValueError('LP cannot modify unsupported source leaves')
    new_logits=all_logits.clone();new_logits[ids]=torch.log(torch.expm1(tau).clamp_min(1e-12)).to(new_logits)
    from outdoor.moge3_evidence import sha256_file
    if kernel=='projected_tau':
        from scripts.audit_canopy_optical_depth_kernel import independent_render
        import canopy_optical_depth_rasterization as backend
        render,render_hash=independent_render()
    elif kernel=='native':
        import inspect
        import hashlib
        import diff_surfel_rasterization as backend
        from outdoor.hybrid_gaussian_renderer import render_hybrid
        render=render_hybrid;render_hash=hashlib.sha256(inspect.getsource(render).encode()).hexdigest()
    else:raise ValueError('Explicit supported replay kernel required')
    binary_hash=sha256_file(backend._C.__file__)
    black=BlackRadianceView(adapter)
    common=dict(background=torch.zeros(3,device='cuda'),include_dynamic=False,optical_replacement_policy='disabled',structural_trainable_start=None)
    records=[];ray_records=[]
    try:
        for index in sorted(set(manifest['excluded_views'])|set(constraints['training_views'])):
            v=views[index];obj,ns,dist,nt=masks.get_index_masks(v.image_name,(0,1,2,3),(v.image_height,v.image_width),torch.device('cuda'))
            tree=obj&ns&dist&~nt;rigid=obj&ns&dist&nt;_,boundary,_=_tree_boundary_masks(~tree)
            gate=static_detail_forward_visibility_gate(base,int(v.colmap_id),include_pending_exact=False)
            gate=torch.cat((gate,torch.ones(len(candidate.xyz),device='cuda')))
            modes={};images=[]
            for name,logits in [('current',all_logits),('lp_counterfactual',new_logits)]:
                adapter.base.opacity_logits.copy_(logits[:len(base)].reshape_as(old_source));candidate.logits.copy_(logits[len(base):].reshape_as(old_candidate))
                out=render(v,teacher.surface,adapter,volume_gate=gate,**common)
                rgb=(out.render+(1-out.alpha)*teacher.sky(v)).clamp(0,1);target=v.original_image.cuda()
                modes[name]={k:float(-10*(rgb[:,mask]-target[:,mask]).square().mean().clamp_min(1e-12).log10()) if mask.any() else None
                             for k,mask in [('tree',tree),('rigid',rigid),('hard',rigid&boundary)]}
                if index in [660,738,674]: images.append(rgb.cpu())
                if index in constraints['training_views']:
                    floor=render(v,teacher.surface,black,volume_gate=gate,**common)
                    for ray in constraints['rays']:
                        if ray['view']!=index:continue
                        x,y=ray['x'],ray['y']
                        ray_records.append(dict({k:v for k,v in ray.items() if k!='target'},mode=name,surface_rgb=floor.render[:,y,x].cpu().tolist(),
                                                target=target[:,y,x].cpu().tolist(),rgb=rgb[:,y,x].cpu().tolist(),
                                                rgb_abs_error=float((rgb[:,y,x]-target[:,y,x]).abs().max())))
            records.append(dict(index=index,evaluation=index in manifest['excluded_views'],modes=modes))
            if images:
                pixels=torch.cat([v.original_image.cpu()]+images,2).permute(1,2,0).numpy()
                Image.fromarray((pixels.clip(0,1)*255).round().astype('uint8')).save(output/f'counterfactual_{index}.png')
            if len(records)%8==0:print(json.dumps(dict(replayed_views=len(records))),flush=True)
    finally:
        adapter.base.opacity_logits.copy_(old_source);candidate.logits.copy_(old_candidate)
    summary={k:float(np.mean([r['modes']['lp_counterfactual'][k]-r['modes']['current'][k] for r in records if r['evaluation'] and r['modes']['current'][k] is not None])) for k in ['tree','rigid','hard']}
    report=dict(scope='read_only_conditional_LP_counterfactual__not_accepted_model',summary=summary,records=records,rays=ray_records,adapter_hash=render_hash,solution=str(solution_dir),
                leaf_optical_kernel=kernel,binary_sha256=binary_hash,
                source_checkpoint_sha256=manifest['source_checkpoint_sha256'],snapshot_sha256=snapshot_sha256,
                limitations=['Historical unlabelled LP replays used projected_tau, not the production alpha_peak*G kernel',
                             'Conditional finite-ray solve; no model export or visibility guarantee'])
    (output/'joint_native_replay.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary),flush=True)
