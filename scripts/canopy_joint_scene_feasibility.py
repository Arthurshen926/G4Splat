"""Training-only, conditional joint optical feasibility. No model mutation."""
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
from scipy import sparse
from scripts.canopy_joint_optical_feasibility import solve_optical_intervals
from scripts.canopy_native_ray_support import sparse_ray_support
from outdoor.hybrid_gaussian_renderer import render_hybrid, persistent_static_evidence_mask, static_detail_forward_visibility_gate


def erode(mask, radius=4):
    return ~torch.nn.functional.max_pool2d((~mask)[None,None].float(), 2*radius+1, stride=1, padding=radius)[0,0].bool()


@torch.no_grad()
def audit_joint_scene(teacher, base, adapter, candidate, views, masks, manifest, output, *, rays_per_kind=8, violated_only=False, snapshot_sha256=None, risk_solution=None, all_training_views=False):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'experimental/canopy_optical_depth_rasterizer'))
    import canopy_optical_depth_rasterization as native
    from outdoor.moge3_evidence import sha256_file
    from scripts.audit_canopy_optical_depth_kernel import independent_render
    from scripts.canopy_surface_rgb_feasibility import BlackRadianceView
    if not 1 <= rays_per_kind <= 128:raise ValueError('Bounded sparse ray budget required')
    experimental_render,_=independent_render();black=BlackRadianceView(adapter)
    train = manifest['training_views']; excluded = set(manifest['excluded_views'])
    if set(train)&excluded or manifest['args'].get('candidate_gain',1) != 1:
        raise ValueError('Enabled static candidates and training-only rays required')
    scores = {r['index']:r.get('selected_score_mean') or 0. for r in manifest['ray_sampling_audit']}
    selected = sorted(set(sorted(train, key=lambda i: (-scores.get(i,0.),i))[:8]
                          +[train[k] for k in np.linspace(0,len(train)-1,8).astype(int)]))
    if all_training_views:selected=list(train)
    centers = torch.stack([views[i].camera_center for i in train])
    neighbors = {}
    for i in selected:
        ids = (centers-views[i].camera_center).norm(dim=1).argsort().cpu().tolist()
        neighbors[i] = [train[j] for j in ids if train[j] != i][:3]
    needed = sorted(set(selected).union(*(set(x) for x in neighbors.values())))
    common = dict(background=torch.zeros(3,device='cuda'), include_dynamic=False,
                  optical_replacement_policy='disabled', structural_trainable_start=None)
    cache = {}
    for cache_ordinal,i in enumerate(needed):
        v=views[i]; obj,non_sky,dist,non_tree=masks.get_index_masks(v.image_name,(0,1,2,3),
            (v.image_height,v.image_width),torch.device('cuda'))
        wall=render_hybrid(v,teacher.surface,base,volume_gate=torch.zeros(len(base),device='cuda'),**common)
        depth=wall.median_depth[0]; rgb=wall.render
        valid=(wall.surface_alpha[0]>.99)&torch.isfinite(depth)&(depth>.2)
        target=v.original_image.cuda()
        rigid=erode(obj&non_sky&dist&non_tree)&valid&((rgb-target).abs().amax(0)<.04)
        tree=erode(obj&non_sky&dist&~non_tree)&valid
        cache[i]=dict(depth=depth.cpu(),rgb=rgb.cpu(),target=target.cpu(),rigid=rigid.cpu(),tree=tree.cpu())
        if cache_ordinal%25==0:print(json.dumps(dict(joint_background_views=cache_ordinal+1,total=len(needed))),flush=True)
    xyz,_,alpha=adapter.conditioned_state(None,include_dynamic=False)
    scales=adapter.scales; quat=adapter.normalized_quaternions
    eligible=torch.cat((persistent_static_evidence_mask(base)&base.static_leaf_mask,
                        torch.ones(len(candidate.xyz),dtype=torch.bool,device='cuda'))).cpu().numpy()
    tau0=-np.log1p(-alpha.cpu().numpy().clip(0,1-1e-6))
    prior_rays=[];risk_logits=None
    if risk_solution is not None:
        risk_solution=Path(risk_solution);risk_info=json.loads((risk_solution/'report.json').read_text())
        previous=json.loads((Path(risk_info['source'])/'joint_feasibility.json').read_text())
        if previous.get('snapshot_sha256')!=snapshot_sha256:raise ValueError('Risk probe must bind the exact checkpoint')
        prior_rays=previous['rays'];solution=np.load(risk_solution/'joint_solution.npz')
        ids=solution['primitive_ids'];tau=solution['tau']
        if (ids.dtype!=np.int64 or len(np.unique(ids))!=len(ids) or (ids<0).any() or (ids>=len(alpha)).any()
                or not eligible[ids].all() or not np.isfinite(tau).all() or (tau<0).any()
                or not np.allclose(tau0[ids],solution['current_tau'],rtol=2e-5,atol=2e-6)):
            raise ValueError('Invalid risk probe state/authority')
        risk_logits=torch.cat((adapter.base.opacity_logits.reshape(-1),candidate.logits.reshape(-1))).clone()
        risk_logits[torch.from_numpy(ids).cuda()]=torch.log(torch.expm1(torch.from_numpy(tau).cuda()).clamp_min(1e-12)).to(risk_logits)
    kernels=[]; lower=[]; upper=[]; metadata=[]; discarded=[]
    for i in selected:
        v=views[i]; c={k:t.cuda() for k,t in cache[i].items()}
        # Conditional background radiance lower bound; .03 is a tolerance sweep
        # centre, not a fixed opacity target and not a geometric depth label.
        L=torch.log((c['rgb'].clamp_min(1e-6)/(c['target']+.03).clamp_min(1e-6))).amax(0).clamp_min(0)
        pos=c['tree']&(L>.05)
        if violated_only:
            active_gate=static_detail_forward_visibility_gate(base,int(v.colmap_id),include_pending_exact=False)
            floor=experimental_render(v,teacher.surface,black,
                volume_gate=torch.cat((active_gate,torch.ones(len(candidate.xyz),device='cuda'))),**common)
            pos &= (floor.render-c['target']-.03).amax(0)>0
        chosen=[]
        risk=None
        if risk_logits is not None:
            active_gate=static_detail_forward_visibility_gate(base,int(v.colmap_id),include_pending_exact=False)
            fullgate=torch.cat((active_gate,torch.ones(len(candidate.xyz),device='cuda')))
            before=experimental_render(v,teacher.surface,adapter,volume_gate=fullgate,**common)
            before_rgb=(before.render+(1-before.alpha)*teacher.sky(v)).clamp(0,1)
            saved_source=adapter.base.opacity_logits.clone();saved_candidates=candidate.logits.clone()
            try:
                adapter.base.opacity_logits.copy_(risk_logits[:len(base)].reshape_as(saved_source))
                candidate.logits.copy_(risk_logits[len(base):].reshape_as(saved_candidates))
                after=experimental_render(v,teacher.surface,adapter,volume_gate=fullgate,**common)
                after_rgb=(after.render+(1-after.alpha)*teacher.sky(v)).clamp(0,1)
                risk=((after_rgb-c['target']).abs()-(before_rgb-c['target']).abs()).amax(0).clamp_min(0)
            finally:
                adapter.base.opacity_logits.copy_(saved_source);candidate.logits.copy_(saved_candidates)
        for kind,mask in [('occlusion',pos),('visible',c['rigid'])]:
            if kind=='visible' and risk is not None:mask=mask&(risk>.002)
            ys,xs=torch.where(mask)
            gen=torch.Generator().manual_seed(9181+i+(kind=='visible'))
            order=(risk[ys,xs].argsort(descending=True)[:max(256,rays_per_kind*8)] if kind=='visible' and risk is not None
                   else torch.randperm(len(ys),generator=gen)[:max(256,rays_per_kind*8)].to(ys.device))
            ys,xs=ys[order],xs[order]
            if kind=='visible' and len(ys):
                z=c['depth'][ys,xs]
                rays=torch.stack(((xs-v.cx)/v.focal_x,(ys-v.cy)/v.focal_y,torch.ones_like(z)),1)
                camera=rays*z[:,None]; transform=v.world_view_transform
                world=(camera-transform[3,:3])@transform[:3,:3].T
                witnesses=torch.zeros(len(ys),device='cuda',dtype=torch.int32)
                for j in neighbors[i]:
                    nv=views[j]; nc=cache[j]; p=world@nv.world_view_transform[:3,:3]+nv.world_view_transform[3,:3]
                    u=(nv.focal_x*p[:,0]/p[:,2]+nv.cx).round().long(); w=(nv.focal_y*p[:,1]/p[:,2]+nv.cy).round().long()
                    valid=(p[:,2]>.2)&(u>=0)&(u<nv.image_width)&(w>=0)&(w<nv.image_height)
                    ids=valid.nonzero().flatten(); uu=u[ids].cpu(); ww=w[ids].cpu()
                    agreement=nc['rigid'][ww,uu].cuda()&((nc['depth'][ww,uu].cuda()-p[ids,2]).abs()<.02*p[ids,2]+.01)
                    witnesses[ids]+=agreement.int()
                keep=witnesses>=2;ys,xs=ys[keep],xs[keep]
            for y,x in zip(ys[:rays_per_kind].tolist(),xs[:rays_per_kind].tolist()): chosen.append((kind,y,x))
        chosen=list(dict.fromkeys(chosen+[(r['kind'],r['y'],r['x']) for r in prior_rays if r['view']==i]))
        if not chosen: continue
        uv=torch.tensor([[x,y] for _,y,x in chosen],device='cuda',dtype=torch.float32)
        z=torch.stack([c['depth'][y,x] for _,y,x in chosen])
        projected=native._C.diagnostic_volume_support(xyz.contiguous(),scales.contiguous(),quat.contiguous(),
            v.world_view_transform.contiguous(),v.full_proj_transform.contiguous(),v.image_width,v.image_height,
            math.tan(v.FoVx*.5),math.tan(v.FoVy*.5))
        K,omitted=sparse_ray_support(projected,uv,z,v.image_width,v.image_height)
        gate=static_detail_forward_visibility_gate(base,int(v.colmap_id),include_pending_exact=False)
        gate=torch.cat((gate,torch.ones(len(candidate.xyz),device='cuda'))).cpu().numpy()
        if not np.isin(gate,[0.,1.]).all(): raise ValueError('Nonbinary ownership requires a different optical model')
        K=K.multiply(gate).tocsr();kernels.append(K);discarded.extend(omitted.tolist())
        for kind,y,x in chosen:
            lower.append(float(L[y,x]) if kind=='occlusion' else 0.)
            # Visibility tolerance is conditional on multi-view wall agreement;
            # it is not inferred solely from the presence of a wall behind trees.
            upper.append(float(-torch.log1p(-(.03/c['rgb'][:,y,x].max().clamp_min(.03)).clamp_max(.95))) if kind=='visible' else np.inf)
            metadata.append(dict(view=i,x=x,y=y,kind=kind,wall_depth=float(c['depth'][y,x]),
                                 wall_rgb=c['rgb'][:,y,x].cpu().tolist(),target=c['target'][:,y,x].cpu().tolist(),
                                 evidence='eroded_mask_and_two_neighbor_wall_depth_RGB_agreement' if kind=='visible' else 'eroded_tree_RGB_necessary_lower_bound'))
        print(json.dumps(dict(joint_support_view=i,rays=len(chosen),nonzeros=K.nnz)),flush=True)
    if not kernels or not any(r['kind']=='visible' for r in metadata):
        raise RuntimeError('Insufficient independently corroborated visible-wall constraints')
    K=sparse.vstack(kernels,format='csr');lo=np.array(lower);hi=np.array(upper)
    fixed=K[:,~eligible]@tau0[~eligible]; variable_ids=np.flatnonzero(eligible)
    Kv=K[:,eligible].tocsr();used=np.unique(Kv.indices); variable_ids=variable_ids[used];Kv=Kv[:,used]
    fixed_excess=np.maximum(fixed-hi,0)
    # An already-invalid frozen contribution is reported separately, not hidden
    # by forcing inconsistent intervals into the solver.
    lo=np.maximum(lo-fixed,0);hi=np.maximum(hi-fixed,0)
    caps=np.full(len(variable_ids),-math.log(1e-6));caps[variable_ids<len(base)]=-math.log(.005)
    result=solve_optical_intervals(Kv,lo,hi,caps)
    sparse.save_npz(output/'joint_kernel.npz',Kv)
    np.savez(output/'joint_solution.npz',primitive_ids=variable_ids,tau=result['tau'],lower=lo,upper=hi,
             current_tau=tau0[variable_ids],fixed_optical_depth=fixed)
    report={k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
    report.update(rays=metadata,variables=len(variable_ids),nonzeros=Kv.nnz,
        snapshot_sha256=snapshot_sha256,source_checkpoint_sha256=manifest['source_checkpoint_sha256'],
        native_binary_sha256=sha256_file(Path(native._C.__file__)),
        rays_per_kind=rays_per_kind,violated_only=violated_only,
        risk_solution=str(risk_solution) if risk_solution is not None else None,
        all_training_views=all_training_views,
        frozen_constraint_excess=float(fixed_excess.sum()),max_omitted_kernel_sum=max(discarded),
        lower_slack=result['lower_slack'].tolist(),upper_slack=result['upper_slack'].tolist(),
        training_views=selected,excluded_views_untouched=True,
        limitations=['Conditional single-background-depth exponential submodel; native mixed revalidation REQUIRED',
                     'Visible-wall labels depend on existing geometry, masks and multi-view agreement, not ground truth',
                     'A sparse ray subset cannot establish full-scene feasibility; no parameter changes or promotion'])
    (output/'joint_feasibility.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['rays','lower_slack','upper_slack']}),flush=True)
