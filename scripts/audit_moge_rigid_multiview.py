"""CPU training-only native MoGe depth/normal consistency, not geometry truth."""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from matcha.cambridge_masks import CambridgeMaskLookup


def camera_points_to_world(points,rotation,position):
    return points@np.asarray(rotation).T+np.asarray(position)


def world_points_to_camera(points,rotation,position):
    return (points-np.asarray(position))@np.asarray(rotation)


def nearest_other_cameras(centers,count=6):
    centers=np.asarray(centers,dtype=float)
    distance=((centers[:,None]-centers[None])**2).sum(-1)
    # Coincident camera centers must not let the source view count as support.
    np.fill_diagonal(distance,np.inf)
    return np.argsort(distance,axis=1,kind='stable')[:,:min(count,max(0,len(centers)-1))]


def baseline_angle(points,first,second):
    a=points-np.asarray(first);b=points-np.asarray(second)
    denom=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
    return np.rad2deg(np.arccos(np.clip((a*b).sum(-1)/np.maximum(denom,1e-12),-1,1)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--depth-directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--rays-per-view',type=int,default=256)
    p.add_argument('--export-supported-rays',action='store_true',help='Export conditional measurements, never geometry truth')
    p.add_argument('--neighbor-count',type=int,default=6)
    p.add_argument('--minimum-baseline-deg',type=float,default=1.)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    if a.neighbor_count<2 or not 0<a.minimum_baseline_deg<90:
        raise ValueError('At least two neighbors and a baseline angle in (0,90) required')
    manifest=json.loads((a.run/'manifest.json').read_text());train=manifest['training_views']
    assert not set(train)&set(manifest['excluded_views'])
    cameras=json.loads((a.run/'cameras.json').read_text())
    masks=CambridgeMaskLookup(Path(manifest['args']['source_path']),Path(manifest['args']['masks']))
    scale=manifest['calibration']['metric_scale']
    centers=np.array([cameras[i]['position'] for i in train])
    neighbors=nearest_other_cameras(centers,a.neighbor_count)
    cache=OrderedDict();cache_capacity=max(16,2*a.neighbor_count)
    def load(index):
        if index in cache:
            value=cache.pop(index);cache[index]=value;return value
        c=cameras[index]
        with np.load(a.depth_directory/(c['img_name']+'.npz')) as z:
            d=z['depth_m'].astype(np.float32)*scale;n=z['normal_direct_camera'].astype(np.float32)
            valid=z['valid_mask'].astype(bool)&np.isfinite(d)&(d>.2)&np.isfinite(n).all(-1)
        obj,sky,dist,non_tree=masks.get_index_masks(c['img_name'],(0,1,2,3),d.shape,torch.device('cpu'))
        valid&=(obj&sky&dist&non_tree).numpy()
        value=(d,n,valid);cache[index]=value
        while len(cache)>cache_capacity:cache.popitem(last=False)
        return value
    totals=dict(rays=0,in_frame_pairs=0,rigid_valid_pairs=0,depth_agree2percent=0,
                depth_agree5percent=0,depth_agree10percent=0,depth5_normal30=0,
                rays_with_two_depth5_neighbors=0,rays_with_two_depth5_normal30_neighbors=0)
    rows=[];export=[];diverse_total=0
    for ordinal,index in enumerate(train):
        c=cameras[index];d,n,valid=load(index);h,w=d.shape
        yy,xx=np.nonzero(valid)
        if not len(xx):continue
        pick=np.random.default_rng(1701+index).choice(len(xx),min(a.rays_per_view,len(xx)),replace=False)
        x=xx[pick];y=yy[pick];z=d[y,x];sx=w/c['width'];sy=h/c['height']
        local=np.stack(((x+.5-c['cx']*sx)/(c['fx']*sx)*z,(y+.5-c['cy']*sy)/(c['fy']*sy)*z,z),axis=-1)
        world=camera_points_to_world(local,c['rotation'],c['position'])
        normal=n[y,x]@np.asarray(c['rotation']).T
        normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-8)
        depth_support=np.zeros(len(x),dtype=np.int32);joint_support=np.zeros(len(x),dtype=np.int32)
        diverse_support=np.zeros(len(x),dtype=np.int32)
        totals['rays']+=len(x)
        for neighbor in neighbors[ordinal]:
            j=train[neighbor];cj=cameras[j];dj,nj,vj=load(j);hj,wj=dj.shape
            q=world_points_to_camera(world,cj['rotation'],cj['position'])
            px=np.rint((q[:,0]/np.maximum(q[:,2],1e-8)*cj['fx']+cj['cx'])*(wj/cj['width'])-.5).astype(np.int64)
            py=np.rint((q[:,1]/np.maximum(q[:,2],1e-8)*cj['fy']+cj['cy'])*(hj/cj['height'])-.5).astype(np.int64)
            ids=np.flatnonzero((q[:,2]>.2)&(px>=0)&(px<wj)&(py>=0)&(py<hj))
            totals['in_frame_pairs']+=len(ids)
            ids=ids[vj[py[ids],px[ids]]];totals['rigid_valid_pairs']+=len(ids)
            target=dj[py[ids],px[ids]];error=np.abs(q[ids,2]/target-1)
            for threshold,key in [(.02,'depth_agree2percent'),(.05,'depth_agree5percent'),(.1,'depth_agree10percent')]:
                totals[key]+=int((error<threshold).sum())
            depth_support[ids]+=(error<.05)
            target_normal=nj[py[ids],px[ids]]@np.asarray(cj['rotation']).T
            target_normal/=np.maximum(np.linalg.norm(target_normal,axis=-1,keepdims=True),1e-8)
            joint=(error<.05)&(np.abs((target_normal*normal[ids]).sum(-1))>np.cos(np.deg2rad(30)))
            totals['depth5_normal30']+=int(joint.sum());joint_support[ids]+=joint
            diverse_support[ids]+=joint&(baseline_angle(world[ids],c['position'],cj['position'])>=a.minimum_baseline_deg)
        totals['rays_with_two_depth5_neighbors']+=int((depth_support>=2).sum())
        totals['rays_with_two_depth5_normal30_neighbors']+=int((joint_support>=2).sum())
        qualified=diverse_support>=2;diverse_total+=int(qualified.sum())
        if a.export_supported_rays and qualified.any():
            export.append(np.column_stack((np.full(int(qualified.sum()),index),x[qualified],y[qualified],
                z[qualified],world[qualified],normal[qualified],diverse_support[qualified])))
        rows.append(dict(index=index,rays=len(x),two_depth5=int((depth_support>=2).sum()),two_depth5_normal30=int((joint_support>=2).sum())))
        if ordinal%20==0:print(json.dumps(dict(completed=ordinal+1,total=len(train))),flush=True)
    report=dict(scope='training_only_native_MoGe_rigid_consistency__not_ground_truth',totals=totals,per_view=rows,
        global_scale=scale,canopy_profiles_used=False,training_views=train,excluded_views=manifest['excluded_views'],
        limitations=['Uniform rigid rays, not a thin-structure-specific sample',f'Nearest {a.neighbor_count} camera centers; baseline filter does not establish independent correspondence',
                     'Consistency is not correct correspondence, absolute accuracy, or source-model improvement'])
    report['diverse_support']=dict(count=diverse_total,minimum_baseline_deg=a.minimum_baseline_deg,
                                  queried_neighbors=a.neighbor_count,required_neighbors=2,
                                  relative_depth_tolerance=.05,normal_angle_deg=30)
    report['cpu_cache_capacity']=cache_capacity
    if a.minimum_baseline_deg==1.:
        report['two_depth5_normal30_neighbors_each_baseline_at_least_1deg']=diverse_total
    if a.export_supported_rays:
        columns=['source_index','native_x','native_y','camera_z','world_x','world_y','world_z',
                 'normal_x','normal_y','normal_z','consistent_diverse_neighbors']
        measurements=np.concatenate(export) if export else np.empty((0,len(columns)))
        np.savez_compressed(a.output/'conditional_rigid_measurements.npz',measurements=measurements)
        report['conditional_measurements']=dict(file='conditional_rigid_measurements.npz',columns=columns,
            count=len(measurements),depth_directory=str(a.depth_directory),source_run=str(a.run),
            warning='Sampled single-depth hypotheses, not fused points or verified source-model corrections')
    (a.output/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(totals),flush=True)


if __name__=='__main__':main()
