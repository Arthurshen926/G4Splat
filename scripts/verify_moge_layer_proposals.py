"""Associate native layer samples across TRAIN cameras before geometry birth.

Depth/RGB/baseline agreement is conditional evidence, not correspondence truth.
No old-model depth agreement or normal validity is required for depth proposals.
"""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.audit_moge_rigid_multiview import camera_points_to_world, world_points_to_camera, nearest_other_cameras, baseline_angle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--packets',type=Path,required=True)
    p.add_argument('--depth-directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-views',type=int,default=24)
    p.add_argument('--rays-per-view',type=int,default=512)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    m=json.loads((a.run/'manifest.json').read_text()); cams=json.loads((a.run/'cameras.json').read_text())
    audit=json.loads((a.packets/'audit.json').read_text())
    train=m['training_views'];excluded=m['excluded_views']
    if set(train)&set(excluded) or audit['training_views']!=train:raise ValueError('Train identity mismatch')
    neighbors=nearest_other_cameras(np.array([cams[i]['position'] for i in train]),12)
    # Selection uses only training-view layer statistics, not regression residuals.
    ranked=sorted(audit['per_view'],key=lambda r:r['regions']['rigid']['exactly_two_mode_cells'],reverse=True)
    selected=[r['index'] for r in ranked[:a.source_views]]
    masks=CambridgeMaskLookup(Path(m['args']['source_path']),Path(m['args']['masks']))
    scale=float(m['calibration']['metric_scale']);cache=OrderedDict()
    def load(i):
        if i in cache:cache.move_to_end(i);return cache[i]
        c=cams[i]
        with np.load(a.depth_directory/(c['img_name']+'.npz')) as z:
            depth=z['depth_m'].astype(np.float32)*scale
            valid=z['valid_mask'].astype(bool)&z['refinement_valid_mask'].astype(bool)&np.isfinite(depth)&(depth>0)
            normal=z['normal_direct_camera'].astype(np.float32)
        rgb=np.asarray(Image.open(Path(m['args']['source_path'])/'images'/(c['img_name']+'.png')).convert('RGB'),dtype=np.float32)/255
        if rgb.shape[:2]!=depth.shape:raise ValueError('Native image identity/shape mismatch')
        obj,sky,dist,nt=masks.get_index_masks(c['img_name'],(0,1,2,3),depth.shape,torch.device('cpu'))
        rigid=(obj&sky&dist&nt).numpy()
        value=depth,valid,normal,rgb,rigid;cache[i]=value
        while len(cache)>16:cache.popitem(last=False)
        return value
    packets=[];records=[]
    for ordinal,i in enumerate(selected):
        c=cams[i];d,valid,n,rgb,rigid=load(i);h,w=d.shape
        with np.load(a.packets/(c['img_name']+'.npz')) as z:
            # Start with nearer native samples from mixed rigid cells. Background
            # modes stay in the evidence packet; they are not promoted as thin rods.
            x=z['front_native_x'];y=z['front_native_y'];eligible=z['rigid']&(x>=0)&(y>=0)
            ids=np.flatnonzero(eligible)
            ids=np.random.default_rng(502+i).choice(ids,min(len(ids),a.rays_per_view),replace=False)
            x=x[ids];y=y[ids]
            area_z=z['raw_area_depth'][ids]*scale
        z=d[y,x];sx=w/c['width'];sy=h/c['height']
        local=np.stack(((x+.5-c['cx']*sx)/(c['fx']*sx)*z,(y+.5-c['cy']*sy)/(c['fy']*sy)*z,z),-1)
        world=camera_points_to_world(local,c['rotation'],c['position']);color=rgb[y,x]
        area_world=camera_points_to_world(local*(area_z/z)[:,None],c['rotation'],c['position'])
        support=np.full((len(x),12),-1,np.int32);support_xy=np.full((len(x),12,2),-1,np.int32)
        depth_count=np.zeros(len(x),int);rgb_count=np.zeros(len(x),int)
        for slot,nidx in enumerate(neighbors[train.index(i)]):
            j=train[nidx];cj=cams[j];dj,vj,nj,rj,mj=load(j);hj,wj=dj.shape
            q=world_points_to_camera(world,cj['rotation'],cj['position'])
            px=np.rint((q[:,0]/np.maximum(q[:,2],1e-8)*cj['fx']+cj['cx'])*wj/cj['width']-.5).astype(int)
            py=np.rint((q[:,1]/np.maximum(q[:,2],1e-8)*cj['fy']+cj['cy'])*hj/cj['height']-.5).astype(int)
            ids=np.flatnonzero((q[:,2]>0)&(px>=0)&(px<wj)&(py>=0)&(py<hj))
            ids=ids[vj[py[ids],px[ids]]&mj[py[ids],px[ids]]]
            good=(np.abs(q[ids,2]/dj[py[ids],px[ids]]-1)<.03)&(baseline_angle(world[ids],c['position'],cj['position'])>=1.)
            ids=ids[good];depth_count[ids]+=1
            ids=ids[np.abs(color[ids]-rj[py[ids],px[ids]]).mean(-1)<.12]
            rgb_count[ids]+=1;support[ids,slot]=j;support_xy[ids,slot]=np.stack((px[ids],py[ids]),-1)
        good=(rgb_count>=2)&valid[y,x]&rigid[y,x]
        normal=n[y,x]@np.asarray(c['rotation']).T
        normal_valid=np.isfinite(normal).all(-1)&(np.linalg.norm(normal,axis=-1)>1e-5)
        normal=np.nan_to_num(normal);normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-8)
        packets.append(dict(source_view=np.full(good.sum(),i,np.int32),native_xy=np.stack((x,y),-1)[good],
            xyz=world[good],area_xyz=area_world[good],rgb=color[good],normal=normal[good],normal_valid=normal_valid[good],
            native_pixel_sigma=(z/(c['fx']*sx)*.7)[good],
            support_views=support[good],support_native_xy=support_xy[good],depth_camera_z=z[good]))
        records.append(dict(index=i,proposed=len(x),depth_three_views=int((depth_count>=2).sum()),
                            depth_rgb_three_views=int(good.sum()),normal_missing=int((good&~normal_valid).sum())))
        print(json.dumps(records[-1]),flush=True)
    output={key:np.concatenate([r[key] for r in packets]) for key in packets[0]}
    np.savez_compressed(a.output/'conditional_native_layer_proposals.npz',**output)
    measurements=np.column_stack((output['source_view'],output['native_xy'],output['depth_camera_z'],
                                  output['xyz'],output['normal'],(output['support_views']>=0).sum(-1)))
    np.savez_compressed(a.output/'conditional_rigid_measurements.npz',measurements=measurements)
    report=dict(scope='training_only_conditional_layer_association__not_verified_geometry',training_views=train,
                excluded_views=excluded,selected_source_views=selected,records=records,
                accepted_samples=len(output['xyz']),scale=scale,
                conditional_measurements=dict(source_run=str(a.run.resolve())),
                limitations=['Predicted-depth and RGB consistency are not independent triangulation',
                             'Nearest-neighbor native lookup; repetitive texture and occlusion remain possible',
                             'No source model replacement or production geometry insertion'])
    with (a.output/'audit.json').open('x') as f:json.dump(report,f,indent=2)


if __name__=='__main__':main()
