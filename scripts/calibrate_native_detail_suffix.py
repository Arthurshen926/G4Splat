"""Paired raw-layer versus area-depth surface proposals; source map frozen."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import render_hybrid,static_detail_forward_visibility_gate
from outdoor.native_detail_suffix import NativeDetailSuffix
from outdoor.surface_geometry_delta import SurfaceGeometryDelta
from outdoor.scoped_detail_loss import backward_detail_objective
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.evaluate_hybrid_teacher import _tree_boundary_masks


def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=1920,white_background=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--proposals',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--geometry',choices=('native','area'),required=True)
    p.add_argument('--steps',type=int,default=8)
    p.add_argument('--eval-count',type=int,default=4)
    p.add_argument('--surface-geometry',action='store_true')
    p.add_argument('--maximum-surface-displacement',type=float,default=.5)
    p.add_argument('--surface-geometry-lr',type=float,default=.001)
    p.add_argument('--eval-every',type=int,default=0)
    p.add_argument('--refined-canopy',type=Path)
    p.add_argument('--refine-proposals',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.65)
    torch.manual_seed(504);np.random.seed(504)
    m=json.loads((a.run/'manifest.json').read_text());report=json.loads((a.proposals.parent/'audit.json').read_text())
    train=m['training_views'];excluded=m['excluded_views']
    if report['training_views']!=train or set(train)&set(excluded):raise ValueError('Training contract mismatch')
    if Path(a.source_path).resolve()!=Path(m['args']['source_path']).resolve():raise ValueError('Dataset mismatch')
    if a.steps<1 or a.eval_count not in (4,48):raise ValueError('Positive steps and 4/48 evaluation views required')
    with np.load(a.proposals) as packet: data={k:packet[k].copy() for k in packet.files}
    keep=data['normal_valid'].astype(bool)
    data={k:v[keep] for k,v in data.items()}
    if not len(data['xyz']):raise ValueError('No normal-supported surface proposals')
    teacher=load_hybrid_teacher(m['args']['checkpoint'],sh_degree=a.sh_degree)
    canopy=teacher.foliage; canopy_candidate=None; canopy_identity=None
    if a.refined_canopy:
        from scripts.load_refined_canopy import load_refined_canopy
        canopy,canopy_candidate,canopy_identity=load_refined_canopy(a.refined_canopy,teacher,m)
    base=teacher.surface
    properties=('get_xyz','get_scaling','get_rotation','get_opacity','get_features')
    frozen={name:getattr(base,name).detach().cpu().clone() for name in properties}
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
    geometry=SurfaceGeometryDelta(base,enabled=a.surface_geometry,maximum_displacement=a.maximum_surface_displacement)
    suffix=NativeDetailSuffix(geometry,tensor(data['xyz' if a.geometry=='native' else 'area_xyz']),
                              tensor(data['normal']),tensor(data['rgb']),tensor(data['native_pixel_sigma']),
                              train_base_geometry=a.surface_geometry,refine_proposals=a.refine_proposals)
    dataset=model.extract(a);dataset.model_path=str(a.output)
    views=LazyScene(dataset,GaussianModel(a.sh_degree),image_cache_size=8).getTrainCameras()
    camera_contract=json.loads((a.run/'cameras.json').read_text())
    for i in train+excluded:
        if views[i].image_name!=camera_contract[i]['img_name']:
            raise ValueError('Source camera identity/order changed')
    masks=CambridgeMaskLookup(Path(a.source_path),Path(m['args']['masks']))
    manifest=dict(scope='unverified_local_surface_optics__fixed_geometry__not_production',geometry=a.geometry,
        steps=a.steps,training_views=train,excluded_views=excluded,proposal_count=len(data['xyz']),
        proposal_sha256=hashlib.sha256(a.proposals.read_bytes()).hexdigest(),source_checkpoint=m['args']['checkpoint'],
        initial_opacity=.01,position_scale_rotation_frozen=not a.surface_geometry,source_geometry_frozen=not a.surface_geometry,
        surface_geometry=a.surface_geometry,maximum_surface_displacement=a.maximum_surface_displacement,
        source_opacity_and_appearance_frozen=True,surface_geometry_lr=a.surface_geometry_lr,
        local_rgb_gradient_scope='appended_proposal_optics_only',global_rgb_scope='valid_object_and_distance_masks',
        eval_every=a.eval_every)
    manifest['scope']='existing_surface_geometry_and_local_optics__experimental_not_production'
    manifest['refined_canopy']=canopy_identity
    manifest['refine_proposals']=a.refine_proposals
    manifest['proposal_reprojection_weight']=.001 if a.refine_proposals else 0.
    manifest['canopy_loader_sha256']=hashlib.sha256((ROOT/'scripts/load_refined_canopy.py').read_bytes()).hexdigest()
    manifest['geometry_helper_sha256']=hashlib.sha256((ROOT/'outdoor/surface_geometry_delta.py').read_bytes()).hexdigest()
    manifest['loss_helper_sha256']=hashlib.sha256((ROOT/'outdoor/scoped_detail_loss.py').read_bytes()).hexdigest()
    manifest['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['suffix_helper_sha256']=hashlib.sha256((ROOT/'outdoor/native_detail_suffix.py').read_bytes()).hexdigest()
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    common=dict(background=torch.zeros(3,device='cuda'),include_dynamic=False,optical_replacement_policy='disabled')
    def prediction(v,surface,grad=False):
        gate=static_detail_forward_visibility_gate(teacher.foliage,int(v.colmap_id),include_pending_exact=False)
        if canopy_candidate is not None:
            gate=torch.cat((gate,gate.new_ones(len(canopy_candidate.xyz))))
        out=render_hybrid(v,surface,canopy,volume_gate=gate,
                          structural_trainable_start=(0 if a.surface_geometry else suffix.start) if grad else None,**common)
        return out.render+(1-out.alpha)*teacher.sky(v)
    eval_ids=[660,738,674,657] if a.eval_count==4 else excluded
    if not set(eval_ids)<=set(excluded):raise ValueError('Evaluation outside regression contract')
    @torch.no_grad()
    def evaluate(step,indices=None):
        rows=[]
        for i in (eval_ids if indices is None else indices):
            v=views[i];gt=v.original_image.cuda();old=prediction(v,base).clamp(0,1);new=prediction(v,suffix).clamp(0,1)
            obj,sky,dist,nt=masks.get_index_masks(v.image_name,(0,1,2,3),(v.image_height,v.image_width),torch.device('cuda'))
            tree=obj&sky&dist&~nt;rigid=obj&sky&dist&nt;_,edge,_=_tree_boundary_masks(~tree)
            regions=dict(tree=tree,rigid=rigid,hard=rigid&edge)
            if i==660:
                rail=torch.zeros_like(rigid);rail[855:1065,60:1170]=True
                regions['railing_rectangle']=rail
            metric=lambda rgb,mask:float(-10*torch.log10(((rgb-gt).square()[:,mask]).mean().clamp_min(1e-12))) if mask.any() else None
            rows.append(dict(index=i,source={k:metric(old,vv) for k,vv in regions.items()},
                             updated={k:metric(new,vv) for k,vv in regions.items()}))
            if step>0:
                panel=torch.cat((gt,old,new),2).permute(1,2,0).cpu().numpy()
                name=f'view_{i}.png' if step==a.steps else f'step_{step:06d}_view_{i}.png'
                Image.fromarray((panel.clip(0,1)*255).round().astype('uint8')).save(a.output/name)
        (a.output/f'eval_{step:06d}.json').write_text(json.dumps(rows,indent=2))
    evaluate(0)
    optimizer=torch.optim.Adam([{'params':[suffix.dc],'lr':.01},{'params':[suffix.logit],'lr':.03}]+geometry.optimization_groups(a.surface_geometry_lr)+suffix.geometry_groups(),eps=1e-15)
    # Prioritize only actual source/support observations; all pixels still have
    # RGB loss and all proposals render in all cameras, without view gating.
    def evidence_mask(i,h,w):
        mask=torch.zeros((h,w),device='cuda')
        source=data['source_view']==i
        xy=[data['native_xy'][source]]
        for slot in range(data['support_views'].shape[1]):
            rows=data['support_views'][:,slot]==i;xy.append(data['support_native_xy'][rows,slot])
        xy=np.concatenate(xy);xy=xy[(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)]
        if len(xy):
            indices=np.rint(xy).astype(np.int64)
            indices=indices[(indices[:,0]<w)&(indices[:,1]<h)]
            mask[torch.as_tensor(indices[:,1],device='cuda'),torch.as_tensor(indices[:,0],device='cuda')]=1
        return F.max_pool2d(mask[None,None],5,1,2)[0,0].bool()
    order=np.random.default_rng(504).permutation(train)
    def save_state(step):
        target=a.output/('suffix.pth' if step==a.steps else f'checkpoint_{step:06d}.pth')
        temporary=target.with_suffix('.tmp.pth')
        torch.save(dict(step=step,dc=suffix.dc.detach().cpu(),logit=suffix.logit.detach().cpu(),
            optimizer=optimizer.state_dict(),manifest=manifest,training_order=order.tolist(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            surface_position_code=geometry.position_code.detach().cpu(),surface_scale_code=geometry.scale_code.detach().cpu(),
            surface_rotation_code=geometry.rotation_code.detach().cpu(),
            proposal_position_delta=suffix.position_delta.detach().cpu(),
            proposal_scale_delta=suffix.scale_delta.detach().cpu()),temporary)
        temporary.replace(target)
    with (a.output/'trace.jsonl').open('x') as log:
        for step in range(1,a.steps+1):
            i=int(order[(step-1)%len(order)]);v=views[i];target=v.original_image.cuda()
            local=evidence_mask(i,v.image_height,v.image_width)
            obj,_,dist,_=masks.get_index_masks(v.image_name,(0,1,2,3),(v.image_height,v.image_width),torch.device('cuda'))
            valid_rgb=obj&dist;local &= valid_rgb
            if not valid_rgb.any():raise ValueError('Training view has no valid static RGB observations')
            rgb=prediction(v,suffix,True)
            error=(rgb-target).abs();global_loss=error[:,valid_rgb].mean()
            reprojection=global_loss.new_zeros(())
            if a.refine_proposals:
                selected=[];pixels=[]
                rows=np.flatnonzero(data['source_view']==i)
                selected.extend(rows.tolist());pixels.extend(data['native_xy'][rows].tolist())
                for slot in range(data['support_views'].shape[1]):
                    rows=np.flatnonzero(data['support_views'][:,slot]==i)
                    selected.extend(rows.tolist());pixels.extend(data['support_native_xy'][rows,slot].tolist())
                if selected:
                    c=camera_contract[i]
                    points=suffix.proposal_xyz[selected]
                    q=(points-tensor(c['position']))@tensor(c['rotation'])
                    if (q[:,2]<=.2).any():raise ValueError('Refined detail moved behind its support camera')
                    fx=c['fx']*v.image_width/c['width'];fy=c['fy']*v.image_height/c['height']
                    cx=c['cx']*v.image_width/c['width']-.5;cy=c['cy']*v.image_height/c['height']-.5
                    uv=torch.stack((q[:,0]/q[:,2]*fx+cx,q[:,1]/q[:,2]*fy+cy),-1)
                    reprojection=F.smooth_l1_loss(uv,tensor(pixels))
            local_loss=error[:,local].mean() if local.any() else None
            backward_detail_objective(global_loss+.001*reprojection,local_loss,[suffix.dc,suffix.logit])
            loss=global_loss+.001*reprojection+(local_loss if local_loss is not None else 0)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            row=dict(step=step,index=i,loss=float(loss.detach()),supported_pixels=int(local.sum()),
                     global_rgb_loss=float(global_loss.detach()),local_proposal_loss=float(local_loss.detach()) if local_loss is not None else 0.,
                     reprojection_loss=float(reprojection.detach()),
                     opacity_mean=float(suffix.logit.sigmoid().mean().detach()))
            log.write(json.dumps(row)+'\n');log.flush()
            if step==1 or step%8==0:print(json.dumps(row),flush=True)
            if a.eval_every>0 and step%a.eval_every==0 and step<a.steps:
                save_state(step)
                evaluate(step,[660,738,674,657])
    unchanged=all(torch.equal(frozen[name],getattr(base,name).detach().cpu()) for name in properties)
    (a.output/'frozen_parameter_audit.json').write_text(json.dumps(dict(unchanged=unchanged)))
    if not unchanged:raise RuntimeError('Frozen source surface changed')
    (a.output/'geometry_movement.json').write_text(json.dumps(geometry.movement_audit(),indent=2))
    save_state(a.steps)
    evaluate(a.steps)
    print(json.dumps(dict(completed=True,steps=a.steps,reference_surface_unchanged=unchanged,
                         optimized_geometry=geometry.movement_audit())),flush=True)


if __name__=='__main__':main()
