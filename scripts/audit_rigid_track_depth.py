"""Read-only native rigid depth versus real multi-view descriptor tracks.

This is a contradiction screen, not permission to remove surfels. A median
depth discrepancy does not identify the offending primitive, and a learned
descriptor cycle is evidence rather than ground truth.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import render_hybrid
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.evaluate_canopy_validation import FIXED,ADDITIONAL,tensor_digest
from scripts.train_unified_outdoor_teacher import _file_sha256


def independent_track_mask(offsets, camera_indices, allowed_cameras,
                           reprojection, angle, cycle_rank, *, minimum_cameras=3, require_cycle=True):
    offsets=np.asarray(offsets);camera_indices=np.asarray(camera_indices)
    reprojection=np.asarray(reprojection);angle=np.asarray(angle);cycle_rank=np.asarray(cycle_rank)
    n=len(offsets)-1
    if n<0 or offsets[0]!=0 or offsets[-1]!=len(camera_indices) or np.any(np.diff(offsets)<0):
        raise ValueError('Invalid ragged observations')
    if any(np.shape(x)!=(n,) for x in (reprojection,angle,cycle_rank)):
        raise ValueError('Misaligned track quality arrays')
    counts=np.diff(offsets)
    rows=np.repeat(np.arange(n),counts)
    forbidden=np.bincount(rows,weights=~np.isin(camera_indices,allowed_cameras),minlength=n)>0
    # Count distinct cameras, not observations; duplicates do not establish
    # independent triangulation support.
    pairs=np.unique(np.stack((rows,camera_indices),axis=1),axis=0)
    distinct=np.bincount(pairs[:,0],minlength=n) if len(pairs) else np.zeros(n,dtype=int)
    return (~forbidden)&(distinct>=minimum_cameras)&np.isfinite(reprojection)&(reprojection<1)&np.isfinite(angle)&(angle>=1)&((cycle_rank>=1) if require_cycle else True)


def cycle_archive(directory,views,*,pairs_only=False):
    """Reuse independently closed RGB descriptor cycles, never projected matches.

    Covariance here is the local fixed-camera reprojection information inverse
    under one-pixel isotropic noise; it is not calibrated learned-depth noise.
    """
    records=[];sources={}
    count=2 if pairs_only else 3
    for path in sorted(directory.glob('matches_*_*.npz' if pairs_only else 'tracks_*_*_*.npz')):
        indices=[int(x) for x in path.stem.split('_')[1:]]
        if len(indices)!=count or set(indices)&set(FIXED+ADDITIONAL):
            raise ValueError('Independent non-validation descriptor cameras required')
        sources[str(path)]=_file_sha256(path)
        with np.load(path,allow_pickle=False) as a:
            xyz=a['points'];pixels=np.stack([a[k] for k in (('left','right') if pairs_only else ('left','middle','right'))],axis=1)
            base=a['geometric']&(a['labels']>0)&np.isfinite(xyz).all(1)
            if not pairs_only:base &= a['third_error']<1
            xyz=xyz[base];pixels=pixels[base]
        if not len(xyz):continue
        info=np.zeros((len(xyz),3,3));errors=[];rays=[];positive=np.ones(len(xyz),dtype=bool)
        for local,index in enumerate(indices):
            v=views[index];transform=v.world_view_transform.cpu().numpy().astype(np.float64)
            camera=xyz@transform[:3,:3]+transform[3,:3];x,y,z=camera.T
            positive &= z>0
            projected=np.stack((v.focal_x*x/z+v.cx,v.focal_y*y/z+v.cy),1)
            errors.append(np.linalg.norm(projected-pixels[:,local],axis=1))
            jac=np.zeros((len(xyz),2,3));jac[:,0,0]=v.focal_x/z;jac[:,1,1]=v.focal_y/z
            jac[:,0,2]=-v.focal_x*x/z**2;jac[:,1,2]=-v.focal_y*y/z**2
            jac=jac@transform[:3,:3].T
            info+=np.swapaxes(jac,1,2)@jac
            ray=xyz-v.camera_center.cpu().numpy();rays.append(ray/np.linalg.norm(ray,axis=1,keepdims=True))
        angle=np.stack([np.degrees(np.arccos(np.clip((rays[i]*rays[j]).sum(1),-1,1)))
                        for i in range(count) for j in range(i+1,count)],1)
        error=np.max(errors,axis=0);angle=np.quantile(angle,.1,axis=1)
        good=positive&(error<1)&(angle>=1)&np.isfinite(info).all((1,2))
        good &= np.linalg.eigvalsh(np.nan_to_num(info))[:,0]>1.e-8
        covariance=np.linalg.inv(info[good])
        for n,row in enumerate(np.flatnonzero(good)):
            records.append((xyz[row],pixels[row],indices,error[row],angle[row],np.diag(covariance[n])))
    if not records:raise ValueError('No independently closed usable cycles')
    camera_ids=sorted({i for row in records for i in row[2]});lookup={i:j for j,i in enumerate(camera_ids)}
    return dict(correspondence_builder=np.array('mast3r_reciprocal_pair_hypotheses' if pairs_only else 'mast3r_reciprocal_three_edge_cycles'),
        xyz=np.array([r[0] for r in records],dtype=np.float32),
        observation_pixels=np.concatenate([r[1] for r in records]).astype(np.float32),
        observation_offsets=np.arange(0,count*len(records)+1,count),
        observation_camera_indices=np.array([lookup[i] for r in records for i in r[2]]),
        reprojection_error=np.array([r[3] for r in records]),triangulation_angle_p10=np.array([r[4] for r in records]),
        descriptor_cycle_rank=np.full(len(records),0 if pairs_only else 1,dtype=int),
        position_covariance_diag=np.array([r[5] for r in records],dtype=np.float32),
        camera_names=np.array([views[i].image_name for i in camera_ids]),
        camera_image_sizes=np.array([[views[i].image_width,views[i].image_height] for i in camera_ids])),sources


def nvm_archive(directory,views):
    """Import actual historical observations without inventing descriptor cycles."""
    payload=json.loads((directory/'audit.json').read_text())
    if payload['scope']!='original_observed_nvm_canopy_tracks__read_only_not_coverage_prior':
        raise ValueError('Expected audited original observations')
    if set(payload['excluded_views'])!=set(FIXED+ADDITIONAL):
        raise ValueError('Original track audit has different validation exclusions')
    records=payload['records']
    if not records:raise ValueError('No qualifying original canopy tracks')
    camera_ids=sorted({o['index'] for r in records for o in r['observations']})
    lookup={index:i for i,index in enumerate(camera_ids)}
    observations=[o for r in records for o in r['observations']]
    if any(o['image_name']!=views[o['index']].image_name for o in observations):
        raise ValueError('Original observation camera identity changed')
    counts=[len(r['observations']) for r in records]
    return dict(correspondence_builder=np.array('original_nvm_observed_multiview_tracks'),
        xyz=np.array([r['xyz'] for r in records],dtype=np.float32),
        observation_pixels=np.array([o['uv'] for o in observations],dtype=np.float32),
        observation_offsets=np.r_[0,np.cumsum(counts)],
        observation_camera_indices=np.array([lookup[o['index']] for o in observations]),
        reprojection_error=np.array([max(o['reprojection'] for o in r['observations']) for r in records]),
        triangulation_angle_p10=np.array([r['angle_p10'] for r in records]),
        descriptor_cycle_rank=np.zeros(len(records),dtype=int),
        position_covariance_diag=np.array([np.diag(r['covariance']) for r in records],dtype=np.float32),
        camera_names=np.array([views[i].image_name for i in camera_ids]),
        camera_image_sizes=np.array([[views[i].image_width,views[i].image_height] for i in camera_ids]))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=640,white_background=True)
    for key in ('checkpoint','masks','output'):p.add_argument('--'+key,type=Path,required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--tracks',type=Path)
    source.add_argument('--cycle-directory',type=Path)
    source.add_argument('--pair-directory',type=Path,help='Two-view hypotheses only; never equivalent to cycle-verified evidence')
    source.add_argument('--nvm-track-audit',type=Path,help='Original observed tracks, not projected SfM coverage or descriptor cycles')
    p.add_argument('--maximum-views',type=int,default=0)
    a=p.parse_args()
    if a.maximum_views<0:raise ValueError('Negative scan limit')
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
    sha={k:_file_sha256(getattr(a,k)) for k in ('checkpoint','masks')}
    if a.tracks is not None:sha['tracks']=_file_sha256(a.tracks)
    teacher=load_hybrid_teacher(a.checkpoint,sh_degree=a.sh_degree)
    fields=('_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest')
    before={k:tensor_digest(getattr(teacher.surface,k)) for k in fields}
    dataset=model.extract(a);dataset.model_path=str(a.output)
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=4).getTrainCameras()
    masks=CambridgeMaskLookup(Path(dataset.source_path),a.masks)
    excluded=set(FIXED+ADDITIONAL)
    by_name={v.image_name:i for i,v in enumerate(views)}
    if a.tracks is not None:
        with np.load(a.tracks,allow_pickle=False) as archive:
            data={k:archive[k] for k in archive.files}
    elif a.nvm_track_audit is not None:
        data=nvm_archive(a.nvm_track_audit,views)
        sha['original_track_audit']=_file_sha256(a.nvm_track_audit/'audit.json')
    else:
        data,sha['descriptor_files']=cycle_archive(a.pair_directory or a.cycle_directory,views,
                                                   pairs_only=a.pair_directory is not None)
    if str(data['correspondence_builder'].item()) not in (
            'mast3r_reciprocal_descriptor_union_find','mast3r_reciprocal_three_edge_cycles',
            'mast3r_reciprocal_pair_hypotheses','original_nvm_observed_multiview_tracks'):
        raise ValueError('Real reciprocal descriptor observations required')
    names=[Path(str(x)).stem for x in data['camera_names']]
    allowed=[i for i,name in enumerate(names) if name in by_name and by_name[name] not in excluded]
    offsets=data['observation_offsets'];cam=data['observation_camera_indices']
    keep=independent_track_mask(offsets,cam,allowed,data['reprojection_error'],
        data['triangulation_angle_p10'],data['descriptor_cycle_rank'],
        minimum_cameras=2 if a.pair_directory else 3,
        require_cycle=a.pair_directory is None and a.nvm_track_audit is None)
    track=np.repeat(np.arange(len(keep)),np.diff(offsets))
    observed_keep=keep[track]
    selected=[i for i in allowed if np.any(observed_keep&(cam==i))]
    if a.maximum_views:selected=selected[:a.maximum_views]
    rows=[]
    for ordinal,ci in enumerate(selected):
        index=by_name[names[ci]];view=views[index]
        obs=np.flatnonzero(observed_keep&(cam==ci));ids=track[obs]
        xyz=torch.as_tensor(data['xyz'][ids],device='cuda')
        transform=view.world_view_transform
        point=xyz@transform[:3,:3]+transform[3,:3]
        z=point[:,2]
        projected=torch.stack((point[:,0]/z*view.focal_x+view.cx,
                               point[:,1]/z*view.focal_y+view.cy),1)
        # Pixel centers scale with the raster, not corner coordinates.
        size=torch.tensor([view.image_width,view.image_height],device='cuda')
        native_size=torch.as_tensor(data['camera_image_sizes'][ci],device='cuda')
        pixels=(torch.as_tensor(data['observation_pixels'][obs],device='cuda')+.5)*size/native_size-.5
        residual=(projected-pixels).norm(dim=1)
        xy=pixels.round().long();x,y=xy.unbind(1)
        valid=torch.isfinite(point).all(1)&(z>0)&(residual<1)&(x>=0)&(x<view.image_width)&(y>=0)&(y<view.image_height)
        x=x.clamp(0,view.image_width-1);y=y.clamp(0,view.image_height-1)
        obj,sky,dist,tree=masks.get_index_masks(view.image_name,(0,1,2,3),
            (view.image_height,view.image_width),torch.device('cuda'))
        known=obj&sky&dist
        output=render_hybrid(view,teacher.surface,teacher.foliage,
            background=torch.zeros(3,device='cuda'),include_dynamic=False,
            volume_gate=torch.zeros_like(teacher.foliage.opacity_logits.reshape(-1)),
            optical_replacement_policy='disabled',structural_trainable_start=None)
        rigid_z=output.median_depth[0,y,x];alpha=output.surface_alpha[0,y,x]
        # Diagonal covariance is only a screen; do not call this a full
        # calibrated uncertainty interval or use it to authorize correction.
        variance=torch.as_tensor(data['position_covariance_diag'][ids],device='cuda')
        sigma=(variance*transform[:3,2].square()).sum(1).clamp_min(0).sqrt()
        margin=torch.maximum(torch.full_like(z,.10),torch.maximum(.05*z,3*sigma))
        valid &= known[y,x]&(alpha>=.98)&torch.isfinite(sigma)&(sigma<.1*z)&(rigid_z>0)
        gap=z-rigid_z
        region={}
        for label,mask in (('canopy',~tree[y,x]),('rigid',tree[y,x])):
            subset=valid&mask;v=gap[subset]
            region[label]=dict(observations=int(subset.sum()),
                rigid_shallower=int((subset&(gap>margin)).sum()),
                rigid_deeper=int((subset&(gap< -margin)).sum()),
                median_track_minus_rigid_depth=float(v.median()) if len(v) else None)
        row=dict(index=index,image_name=view.image_name,regions=region,
                 input_observations=len(obs),native_reprojection_valid=int((residual<1).sum()))
        rows.append(row)
        np.savez_compressed(a.output/f'observations_{index}.npz',track_ids=ids,
            pixels=pixels.cpu().numpy(),track_depth=z.cpu().numpy(),rigid_depth=rigid_z.cpu().numpy(),
            margin=margin.cpu().numpy(),valid=valid.cpu().numpy(),canopy=(~tree[y,x]).cpu().numpy())
        print(json.dumps(dict(completed_views=ordinal+1,total_views=len(selected),**row)),flush=True)
    if before!={k:tensor_digest(getattr(teacher.surface,k)) for k in fields}:
        raise RuntimeError('Read-only audit changed rigid parameters')
    result=dict(scope='independent_of_moge__descriptor_track_discrepancy_screen__not_primitive_removal_authority',
        evidence_kind=str(data['correspondence_builder'].item()),
        source_checkpoint=str(a.checkpoint),input_sha256=sha,surface_fingerprints=before,
        excluded_views=sorted(excluded),eligible_tracks=int(keep.sum()),records=rows,
        complete_scan=not a.maximum_views,limitations=['Learned correspondences can be wrong',
        ('Existing tracks may have participated in rigid fitting; not held-out geometry' if a.tracks or a.nvm_track_audit
         else 'Descriptor cycles are new diagnostics; covariance assumes one-pixel independent noise'),
        'Median depth and diagonal covariance do not identify a removable primitive'])
    (a.output/'audit.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
