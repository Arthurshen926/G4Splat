"""Controlled static versus temporally interpolated tree-center refinement.

No evaluation image is fitted. Temporal renders are separate diagnostic
outputs, never substitutes for the static canonical reconstruction metric.
"""
import argparse
import json
import math
import re
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import torch
import torch.nn.functional as F
from PIL import Image
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import render_hybrid,static_detail_forward_visibility_gate,persistent_static_evidence_mask
from outdoor.canopy_deformation_diagnostic import BoundedCanopyCenterField,enable_shared_canopy_optics,project_shared_canopy_optics_
from outdoor.canopy_detail_loss import masked_canopy_ssim_loss,native_rigid_foliage_extinction_loss
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.evaluate_canopy_validation import FIXED,ADDITIONAL,tensor_digest
from scripts.evaluate_hybrid_teacher import _tree_boundary_masks
from scripts.train_unified_outdoor_teacher import _file_sha256


def frame_time(name):
    match=re.search(r'frame(\d+)',name)
    if not match:raise ValueError('Explicit sequential frame identity required')
    return int(match.group(1))


def rigid_extinction_increase_loss(alpha,baseline,rigid):
    """Refinement guard: penalize added spill, not removal of existing canopy.

    This is relative preservation of an initial model, NOT a statement that
    its existing rigid-pixel leaf contribution is physically correct.
    """
    if alpha.numel()!=rigid.numel() or baseline.shape!=rigid.shape:
        raise ValueError('Aligned fixed-reference alpha required')
    core=F.avg_pool2d(rigid.to(alpha)[None,None],3,1,1)[0,0]>=1-1.e-6
    value=-torch.log1p(-alpha.reshape_as(rigid).clamp(0,1-1.e-5))
    reference=-torch.log1p(-baseline.detach().to(alpha).clamp(0,1-1.e-5))
    return (F.relu(value-reference)*core).sum()/core.sum().clamp_min(1)


def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=640,white_background=True)
    for key in ('checkpoint','cohort','masks','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--rank',type=int,choices=(0,8),default=0)
    p.add_argument('--steps',type=int,default=800)
    p.add_argument('--eval-every',type=int,default=400)
    p.add_argument('--spacing',type=float,default=2.)
    p.add_argument('--radius',type=float,default=.1)
    p.add_argument('--canonical-reference-frame',type=int,
        help='Anchor temporal displacement to an actual training frame; None retains legacy mean-code reference')
    p.add_argument('--regularization-reference-radius',type=float,default=.1)
    p.add_argument('--regularization-reference-spacing',type=float,default=2.)
    p.add_argument('--lr',type=float,default=.02)
    p.add_argument('--joint-shared-optics',action='store_true')
    p.add_argument('--shared-color-lr',type=float,default=.0025)
    p.add_argument('--shared-dc-lr-multiplier',type=float,default=1.,
                   help='Scale actual Adam DC updates only; higher SH and opacity schedules unchanged')
    p.add_argument('--shared-opacity-lr',type=float,default=.02)
    p.add_argument('--shared-opacity-projection',choices=('initial_upper','absolute'),default='initial_upper',
        help='Preserve the initial upper envelope; absolute reproduces historical diagnostic clipping')
    p.add_argument('--native-rigid-alpha-weight',type=float,default=0.)
    p.add_argument('--gpu-memory-fraction',type=float,default=.40)
    p.add_argument('--mixed-bound-cache',type=Path)
    p.add_argument('--mixed-bound-weight',type=float,default=0.)
    p.add_argument('--dense-cycle-guide',type=Path)
    p.add_argument('--dense-cycle-weight',type=float,default=0.)
    p.add_argument('--rigid-preservation-mode',choices=('absolute','initial_model'),default='absolute')
    p.add_argument('--observed-anchor-associations',type=Path)
    p.add_argument('--observed-anchor-tracks',type=Path)
    p.add_argument('--observed-anchor-thin-audit',type=Path)
    p.add_argument('--observed-anchor-weight',type=float,default=0.)
    a=p.parse_args()
    if (not math.isfinite(a.dense_cycle_weight) or a.dense_cycle_weight<0
        or (a.dense_cycle_weight and (a.dense_cycle_guide is None or a.rank))):
        raise ValueError('Static training-only reciprocal depth guide required')
    if a.steps<1 or a.eval_every<1 or not 0<a.gpu_memory_fraction<=1 or not 0<a.lr<1:
        raise ValueError('Positive bounded diagnostic schedule required')
    if not math.isfinite(a.shared_dc_lr_multiplier) or not 0<a.shared_dc_lr_multiplier<=20:
        raise ValueError('Finite bounded DC learning-rate multiplier required')
    if a.shared_dc_lr_multiplier!=1 and not a.joint_shared_optics:
        raise ValueError('DC learning-rate ablation requires shared leaf optics')
    if any(not math.isfinite(value) or value<=0 for value in
           (a.regularization_reference_radius,a.regularization_reference_spacing,a.shared_color_lr,a.shared_opacity_lr)):
        raise ValueError('Positive physical regularization reference required')
    if not math.isfinite(a.native_rigid_alpha_weight) or a.native_rigid_alpha_weight<0:
        raise ValueError('Finite nonnegative rigid protection required')
    if not math.isfinite(a.mixed_bound_weight) or a.mixed_bound_weight<0:
        raise ValueError('Finite nonnegative diagnostic bound weight required')
    if a.mixed_bound_weight and (not a.joint_shared_optics or a.mixed_bound_cache is None):
        raise ValueError('Bound experiments require a cache and joint leaf optics')
    anchor_paths=(a.observed_anchor_associations,a.observed_anchor_tracks,a.observed_anchor_thin_audit)
    if any(anchor_paths) and not all(anchor_paths):raise ValueError('Complete independent anchor provenance required')
    if not math.isfinite(a.observed_anchor_weight) or a.observed_anchor_weight<0:
        raise ValueError('Finite nonnegative regional guide weight required')
    if a.observed_anchor_weight and (not all(anchor_paths) or a.rank):
        raise ValueError('This geometry-guided experiment supports only an explicit static field')
    cohort=json.loads(a.cohort.read_text())
    holdout=FIXED+ADDITIONAL
    train=cohort['calibrated_training_views']
    if set(train)&set(holdout) or set(cohort['excluded_views'])!=set(holdout):
        raise ValueError('All 48 validation views must be excluded from fitting')
    if (Path(cohort['source_checkpoint']).resolve()!=a.checkpoint.resolve()
        or cohort['source_checkpoint_sha256']!=_file_sha256(a.checkpoint)):
        raise ValueError('Training cohort must match the source checkpoint')
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(a.gpu_memory_fraction)
    dataset=model.extract(a);dataset.model_path=str(a.output)
    teacher=load_hybrid_teacher(a.checkpoint,sh_degree=dataset.sh_degree)
    foliage=teacher.foliage
    immutable={f'leaf.{name}':parameter for name,parameter in foliage.named_parameters()}
    immutable.update({f'rigid.{name}':getattr(teacher.surface,name)
                      for name in ('_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest')})
    for name,module in [('sky',teacher.sky),('appearance',teacher.appearance)]:
        if hasattr(module,'named_parameters'):
            immutable.update({f'{name}.{key}':value for key,value in module.named_parameters()})
    for parameter in immutable.values():parameter.requires_grad_(False)
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=8).getTrainCameras()
    canonical=teacher.state['training_contract']['static_scene_canonical_rgb']['canonical_sequence']
    if any(not views[i].image_name.startswith(canonical+'__') for i in train+holdout):
        raise ValueError('One canonical sequence required for temporal interpolation')
    train=sorted(train,key=lambda i:frame_time(views[i].image_name))
    times=[frame_time(views[i].image_name) for i in train]
    eligible=persistent_static_evidence_mask(foliage)&foliage.static_leaf_mask
    initial_optical_upper=(foliage.opacity_logits.detach().clone()
                           if a.joint_shared_optics and a.shared_opacity_projection=='initial_upper' else None)
    field=BoundedCanopyCenterField(foliage.xyz.detach(),eligible,times,
                                   spacing=a.spacing,radius=a.radius,rank=a.rank,
                                   reference_time=a.canonical_reference_frame)
    observed_guidance=None
    if all(anchor_paths):
        from scripts.canopy_observed_anchor_guidance import load_guidance
        observed_guidance=load_guidance(*anchor_paths,a.checkpoint,foliage,field,views,train,holdout)
        print(json.dumps(dict(observed_regional_groups=len(observed_guidance['targets']),
                              observed_regional_guide_weight=a.observed_anchor_weight)),flush=True)
    groups=[dict(params=list(field.parameters()),lr=a.lr,initial_lr=a.lr)]
    if a.joint_shared_optics:
        immutable.pop('leaf.features');immutable.pop('leaf.opacity_logits')
        optical_hooks=enable_shared_canopy_optics(foliage,eligible)
        groups.extend([dict(params=[foliage.features],lr=a.shared_color_lr,initial_lr=a.shared_color_lr),
                       dict(params=[foliage.opacity_logits],lr=a.shared_opacity_lr,initial_lr=a.shared_opacity_lr)])
    optimizer=torch.optim.Adam(groups,eps=1.e-15)
    def frozen_fingerprints():
        result={name:tensor_digest(value) for name,value in immutable.items()}
        if a.joint_shared_optics:
            result.update({f'ineligible.{name}':tensor_digest(getattr(foliage,name)[~eligible])
                           for name in ('features','opacity_logits')})
        return result
    before=frozen_fingerprints()
    masks=CambridgeMaskLookup(Path(dataset.source_path),a.masks)
    bound_manifest=None
    if a.mixed_bound_cache is not None:
        from scripts.build_canopy_mixed_order_bound import CONTRACT
        bound_manifest=json.loads((a.mixed_bound_cache/'audit.json').read_text())
        if (bound_manifest['contract']!=CONTRACT
            or not bound_manifest['all_canopy_training_views_covered']
            or Path(bound_manifest['source_checkpoint']).resolve()!=a.checkpoint.resolve()
            or bound_manifest['source_checkpoint_sha256']!=cohort['source_checkpoint_sha256']
            or bound_manifest['masks_sha256']!=_file_sha256(a.masks)
            or set(bound_manifest['calibrated_training_views'])!=set(train)
            or set(bound_manifest['excluded_views'])!=set(holdout)
            or bound_manifest['surface_fingerprints']!={name:tensor_digest(getattr(teacher.surface,name))
                for name in ('_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest')}):
            raise ValueError('Bound cache does not match this frozen-background experiment')
        expected=[]
        for i in train:
            v=views[i]
            obj,sky,dist,tree=masks.get_index_masks(v.image_name,(0,1,2,3),
                (v.image_height,v.image_width),torch.device('cpu'))
            if int((obj&sky&dist&~tree).sum())>=128:expected.append(i)
        if set(expected)!=set(bound_manifest['selected_training_views']):
            raise ValueError('Bound cache misses eligible training views')
        bound_records={row['index']:row for row in bound_manifest['records']}
        if len(bound_records)!=len(expected) or set(bound_records)!=set(expected):
            raise ValueError('Invalid bound record identities')
    dense_guide=[]
    if a.dense_cycle_guide is not None:
        from scripts.canopy_dense_depth_guidance import load_dense_guide,interval_log_depth_loss
        dense_guide=load_dense_guide(a.dense_cycle_guide,views,train,holdout)
    order=[train[i] for i in torch.randperm(len(train),generator=torch.Generator().manual_seed(1701)).tolist()]
    results=[]
    saw_field_gradient=False

    def regions(view):
        obj,sky,dist,tree=masks.get_index_masks(view.image_name,(0,1,2,3),
            (view.image_height,view.image_width),torch.device('cuda'))
        known=obj&sky&dist;canopy=known&~tree;rigid=known&tree
        inside,outside,_=_tree_boundary_masks(~canopy)
        return dict(tree=canopy,tree_interior=canopy&~inside,tree_boundary=canopy&inside,
                    rigid=rigid,hard=rigid&outside,sky=obj&dist&~sky)

    def render(view,time):
        shifted=foliage.xyz.detach()+field.offsets(time)
        output=render_hybrid(view,teacher.surface,foliage,
            background=torch.zeros(3,device='cuda'),include_dynamic=False,
            optical_replacement_policy='disabled',structural_trainable_start=None,
            volume_gate=static_detail_forward_visibility_gate(foliage,int(view.colmap_id),include_pending_exact=False),
            volume_means_override=shifted)
        return output.render+(1-output.alpha)*teacher.sky(view),output.volume_alpha

    baseline_rigid={}
    if a.rigid_preservation_mode=='initial_model':
        with torch.no_grad():
            for ordinal,index in enumerate(train):
                view=views[index];rgb,alpha=render(view,None)
                baseline_rigid[index]=((rgb-view.original_image.cuda()).abs().mean(0).cpu(),
                                       alpha.reshape(view.image_height,view.image_width).cpu())
                if ordinal%50==0 or ordinal+1==len(train):
                    print(json.dumps(dict(initial_rigid_reference_views=ordinal+1,total_views=len(train))),flush=True)

    @torch.no_grad()
    def evaluate(step):
        rows=[]
        for index in holdout:
            view=views[index];m=regions(view);target=view.original_image.cuda()
            predictions={'static_canonical':render(view,None)}
            if a.rank:predictions['interpolated_deformed']=render(view,frame_time(view.image_name))
            entry={'index':index,'modes':{}}
            for mode,(prediction,alpha) in predictions.items():
                # Deployment clips the composed RGB; training keeps its raw
                # gradients, as in the matched native refinement diagnostic.
                prediction=prediction.clamp(0,1)
                if step==0:
                    error=float((prediction-teacher.render(view,task=None,conditioned=False)['rgb']).abs().max())
                    if error>1.e-6:raise RuntimeError(f'Zero field differs from native static rendering: {error}')
                entry['modes'][mode]={key:float(-10*((prediction[:,mask]-target[:,mask]).square().mean().clamp_min(1.e-12)).log10())
                                           if mask.any() else None for key,mask in m.items() if key!='sky'}
                entry['modes'][mode]['tree_alpha']=float(alpha.reshape_as(m['tree'])[m['tree']].mean())
                pair=torch.cat((target,prediction),2).clamp(0,1)
                Image.fromarray((pair.permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(
                    a.output/f'view_{index}_{mode}_{step:04d}.png')
            rows.append(entry)
        summary={mode:{key:sum(row['modes'][mode][key] for row in rows if row['modes'][mode][key] is not None)
                       /sum(row['modes'][mode][key] is not None for row in rows)
                       for key in ('tree','tree_interior','tree_boundary','rigid','hard','tree_alpha')}
                 for mode in rows[0]['modes']}
        result={'step':step,'summary':summary,'per_view':rows};results.append(result)
        (a.output/'metrics.json').write_text(json.dumps({'args':vars(a),'training_views':train,
            'scope':('shared_persistent_leaf_optics_and_center_field__rigid_covariance_sky_frozen'
                     if a.joint_shared_optics else 'tree_center_field_only__all_base_parameters_frozen')
                     +'__static_and_temporal_metrics_separate',
            'parameter_count':sum(x.numel() for x in field.parameters()),'node_count':len(field.keys),
            'shared_optics_eligible_parameters':int(eligible.sum())*(foliage.features[0].numel()+1) if a.joint_shared_optics else 0,
            'eligible_rows':int(eligible.sum()),'results':results},default=str,indent=2))
        if step:
            torch.save({'diagnostic_only':True,'source_checkpoint':str(a.checkpoint.resolve()),
                'step':step,'args':vars(a),'training_views':train,'field':field.state_dict(),
                'shared_leaf_optics':({name:getattr(foliage,name).detach().cpu() for name in ('features','opacity_logits')}
                                      if a.joint_shared_optics else None)},a.output/f'field_{step:04d}.pth')
        print(json.dumps({'evaluation':step,'summary':summary}),flush=True)

    evaluate(0)
    for step in range(1,a.steps+1):
        index=order[(step-1)%len(order)];view=views[index];m=regions(view)
        time=frame_time(view.image_name) if a.rank else None
        optimizer.zero_grad(set_to_none=True)
        logging=step==1 or step%25==0 or step==a.steps
        previous_logits=foliage.opacity_logits.detach().clone() if logging and a.joint_shared_optics else None
        prediction,alpha=render(view,time)
        target=view.original_image.cuda();error=(prediction-target).abs().mean(0)
        tree_photo=(error*m['tree']).sum()/m['tree'].sum().clamp_min(1)
        rigid_error=error
        if baseline_rigid:
            rigid_error=F.relu(error-baseline_rigid[index][0].to(error))
        photo=tree_photo+(rigid_error*m['rigid']).sum()/m['rigid'].sum().clamp_min(1)
        ssim=masked_canopy_ssim_loss(prediction,target,m['tree'])
        color_loss=(error*m['tree']).sum()/m['tree'].sum().clamp_min(1)+.2*ssim
        color_gradient=(torch.autograd.grad(color_loss,foliage.features,retain_graph=True)[0]
                        if a.joint_shared_optics else None)
        sky=(torch.log1p(alpha.reshape_as(m['sky']).clamp_min(0)/.005)*m['sky']).sum()/m['sky'].sum().clamp_min(1)
        magnitude,spatial,temporal=field.regularization(time)
        # Keep the physical displacement prior fixed when testing a larger
        # admissible radius; increasing the bound must not silently weaken it.
        regularization_scale=(a.radius/a.regularization_reference_radius)**2
        spatial_scale=(a.regularization_reference_spacing/a.spacing)**2
        loss=photo+.2*ssim+.1*sky+regularization_scale*(.01*magnitude+.05*spatial_scale*spatial+.01*temporal)
        rigid_extinction=native_rigid_foliage_extinction_loss(alpha,m['rigid'])
        if baseline_rigid:
            rigid_extinction=rigid_extinction_increase_loss(alpha,baseline_rigid[index][1],m['rigid'])
        loss=loss+a.native_rigid_alpha_weight*rigid_extinction
        bound_loss=alpha.sum()*0
        bound_pixels=0
        if bound_manifest is not None and index in bound_records:
            # Targets are necessary background-model bounds, not dense opacity
            # labels. Never fit validation views or alter the native renderer.
            evidence=torch.load(a.mixed_bound_cache/f'bound_{index}.pth',map_location='cpu')
            if (evidence['contract']!=bound_manifest['contract']
                or evidence['image_name']!=view.image_name
                or evidence['target_rgb_sha256']!=tensor_digest(view.original_image)
                or evidence['target_rgb_sha256']!=bound_records[index]['target_rgb_sha256']):
                raise ValueError('Bound pixel evidence has a different RGB identity')
            required=evidence['required_alpha'].to(alpha).detach()
            applicable=evidence['applicable'].to(device=alpha.device)
            if (required.shape!=m['tree'].shape or applicable.shape!=required.shape
                or applicable.dtype!=torch.bool or not torch.isfinite(required).all()
                or (required<0).any() or (required>1).any() or (applicable&~m['tree']).any()):
                raise ValueError('Invalid diagnostic bound raster')
            selected=applicable&(required>0)
            bound_pixels=int(selected.sum())
            if bound_pixels:
                actual=alpha.reshape_as(required)
                # A one-sided log deficit keeps pressure on transparent leaves
                # without rewarding opacity beyond this weak necessary bound.
                deficit=((required[selected]+1.e-6).log()
                         -(actual[selected]+1.e-6).log()).clamp_min(0)
                bound_loss=deficit.mean()
            loss=loss+a.mixed_bound_weight*bound_loss
        observed_loss=alpha.sum()*0
        if observed_guidance is not None:
            from scripts.canopy_observed_anchor_guidance import centroid_guidance_loss
            observed_loss=centroid_guidance_loss(field,observed_guidance)
            loss=loss+a.observed_anchor_weight*observed_loss
        dense_loss=loss*0;dense_pixels=0
        if a.dense_cycle_weight and step%4==0:
            evidence=dense_guide[(step//4-1)%len(dense_guide)]
            guide_view=views[evidence['index']];xy=evidence['uv'];x,y=xy.unbind(1)
            guide_output=render_hybrid(guide_view,teacher.surface,foliage,
                background=torch.zeros(3,device='cuda'),include_dynamic=False,
                optical_replacement_policy='disabled',structural_trainable_start=None,
                volume_gate=static_detail_forward_visibility_gate(foliage,int(guide_view.colmap_id),include_pending_exact=False),
                volume_means_override=foliage.xyz.detach()+field.offsets(None),
                volume_opacity_gradient_gate=torch.zeros_like(foliage.opacity_logits.reshape(-1)),
                volume_appearance_gradient_gate=torch.zeros_like(foliage.opacity_logits.reshape(-1)))
            dense_loss,dense_pixels=interval_log_depth_loss(guide_output.volume_depth[0,y,x],evidence['depth'],
                guide_output.volume_alpha[0,y,x],regions(guide_view)['tree'][y,x])
            if step==4:
                parameters=list(field.parameters())
                optical_parameters=[p for p in (foliage.opacity_logits,foliage.features) if p.requires_grad]
                gradients=torch.autograd.grad(dense_loss,parameters+optical_parameters,
                                              retain_graph=True,allow_unused=True)
                geometry_gradient=sum(float(g.abs().sum()) for g in gradients[:len(parameters)] if g is not None)
                optical_gradient=sum(float(g.abs().sum()) for g in gradients[len(parameters):] if g is not None)
                if dense_pixels==0 or not math.isfinite(geometry_gradient) or geometry_gradient<=0 or optical_gradient!=0:
                    raise RuntimeError('Reciprocal guide must move geometry without optical gradients')
                print(json.dumps(dict(dense_guide_gradient_audit=True,pixels=dense_pixels,
                    geometry_gradient_l1=geometry_gradient,optical_gradient_l1=optical_gradient)),flush=True)
            loss=loss+a.dense_cycle_weight*dense_loss
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite deformation objective')
        loss.backward()
        if color_gradient is not None:foliage.features.grad=color_gradient
        if a.joint_shared_optics and any(not torch.isfinite(p.grad).all() for p in
                                       (foliage.features,foliage.opacity_logits)):
            raise RuntimeError('Nonfinite shared optical gradient')
        gradient=sum(float(x.grad.abs().sum()) for x in field.parameters() if x.grad is not None)
        if not math.isfinite(gradient):raise RuntimeError('Nonfinite deformation gradient')
        saw_field_gradient |= gradient>0
        if any(x.grad is not None for x in immutable.values()):
            raise RuntimeError('Expected only explicitly permitted native CUDA gradients')
        if step==1 and not baseline_rigid and not gradient>0:
            raise RuntimeError('Expected a nonzero native field gradient in the absolute-control smoke')
        for group in optimizer.param_groups:group['lr']=group['initial_lr']*.1**((step-1)/max(a.steps-1,1))
        before_dc=foliage.features[:,0].detach().clone() if a.joint_shared_optics and a.shared_dc_lr_multiplier!=1 else None
        optimizer.step()
        if before_dc is not None:
            from outdoor.canopy_deformation_diagnostic import apply_dc_learning_rate_multiplier_
            apply_dc_learning_rate_multiplier_(foliage,eligible,before_dc,a.shared_dc_lr_multiplier)
        if a.joint_shared_optics:
            project_shared_canopy_optics_(foliage,eligible,initial_opacity_logits=initial_optical_upper)
        if logging:
            with torch.no_grad():
                displacement=field.offsets(time).norm(dim=1)
                maximum=float(displacement.max())
                if not math.isfinite(maximum) or maximum>a.radius+1.e-6:
                    raise RuntimeError('Tree-center displacement exceeded the declared bound')
            row={'step':step,'view':index,'loss':float(loss),'photo':float(photo),'ssim':float(ssim),
                 'field_gradient_l1':gradient,'maximum_displacement':maximum,
                 'mean_eligible_displacement':float(displacement[eligible].mean()),
                 'physical_regularization_scale':regularization_scale,
                 'physical_spatial_scale':spatial_scale,
                 'native_rigid_extinction':float(rigid_extinction),
                 'mixed_bound_loss':float(bound_loss),'mixed_bound_pixels':bound_pixels,
                 'observed_regional_centroid_loss':float(observed_loss),
                 'dense_cycle_depth_loss':float(dense_loss),'dense_cycle_pixels':dense_pixels,
                 'shared_dc_lr_multiplier':a.shared_dc_lr_multiplier,
                 'magnitude':float(magnitude),'spatial':float(spatial),'temporal':float(temporal)}
            if a.joint_shared_optics:
                delta=foliage.opacity_logits.detach()-previous_logits
                row.update(shared_color_gradient_l1=float(foliage.features.grad.abs().sum()),
                    shared_opacity_gradient_l1=float(foliage.opacity_logits.grad.abs().sum()),
                    opacity_growth_rows=int((delta>0).sum()),opacity_retirement_rows=int((delta<0).sum()))
            with (a.output/'trace.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if step%a.eval_every==0 or step==a.steps:evaluate(step)
    after=frozen_fingerprints()
    if not saw_field_gradient:raise RuntimeError('No native field update was exercised')
    if after!=before:raise RuntimeError('A frozen base parameter changed')
    (a.output/'frozen_parameter_audit.json').write_text(json.dumps({'unchanged':True,'fingerprints':before},indent=2))


if __name__=='__main__':main()
