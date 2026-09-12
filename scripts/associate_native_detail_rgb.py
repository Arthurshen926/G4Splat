"""Reassociate conditional native proposals with RGB matches before triangulation."""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from outdoor.native_observation_matching import match_patch,triangulate_rays,triangulate_consensus,epipolar_pixel_line,rebind_angular_sigma


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--proposals',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--consensus',action='store_true')
    p.add_argument('--epipolar',action='store_true')
    p.add_argument('--subpixel',action='store_true')
    p.add_argument('--expand-support',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    m=json.loads((a.run/'manifest.json').read_text());cams=json.loads((a.run/'cameras.json').read_text())
    prior=json.loads((a.proposals.parent/'audit.json').read_text())
    if prior['training_views']!=m['training_views'] or set(m['training_views'])&set(m['excluded_views']):raise ValueError('Training identity mismatch')
    with np.load(a.proposals) as z:data={k:z[k].copy() for k in z.files}
    cache=OrderedDict()
    def image(i):
        if i not in m['training_views']:raise ValueError('Non-training image requested')
        if i in cache:cache.move_to_end(i);return cache[i]
        x=np.asarray(Image.open(Path(m['args']['source_path'])/'images'/(cams[i]['img_name']+'.png')).convert('RGB'),dtype=np.float32)/255
        cache[i]=x
        while len(cache)>16:cache.popitem(last=False)
        return x
    def ray(i,xy,shape):
        c=cams[i];h,w=shape[:2]
        d=np.array([(xy[0]+.5-c['cx']*w/c['width'])/(c['fx']*w/c['width']),
                    (xy[1]+.5-c['cy']*h/c['height'])/(c['fy']*h/c['height']),1.])
        return d@np.asarray(c['rotation']).T
    kept=[];positions=[];matched_rows=[];counts=dict(proposals=len(data['xyz']),forward=0,roundtrip=0,three_views=0,reprojected=0)
    def epipolar(i,xy,shape,j,target_shape):
        direction=ray(i,xy,shape)
        c=cams[j];h,w=target_shape[:2]
        fx=c['fx']*w/c['width'];fy=c['fy']*h/c['height']
        cx=c['cx']*w/c['width']-.5;cy=c['cy']*h/c['height']-.5
        return epipolar_pixel_line(cams[i]['position'],direction,c['position'],c['rotation'],fx,fy,cx,cy)
    for row in range(len(data['xyz'])):
        source=int(data['source_view'][row]);xy=data['native_xy'][row];src=image(source)
        observations=[(source,xy,src.shape)]
        support=list(enumerate(data['support_views'][row]))
        if a.expand_support:
            center=np.asarray(cams[source]['position'])
            nearest=sorted((j for j in m['training_views'] if j!=source),
                           key=lambda j:np.linalg.norm(np.asarray(cams[j]['position'])-center))[:12]
            support=[(-1,j) for j in nearest]
        for slot,j in support:
            if j<0:continue
            j=int(j);dst=image(j)
            if a.expand_support:
                c=cams[j];q=(data['xyz'][row]-np.asarray(c['position']))@np.asarray(c['rotation']);h,w=dst.shape[:2]
                if q[2]<=0:continue
                guess=np.array([(q[0]/q[2]*c['fx']+c['cx'])*w/c['width']-.5,
                                (q[1]/q[2]*c['fy']+c['cy'])*h/c['height']-.5])
                if not (0<=guess[0]<w and 0<=guess[1]<h):continue
            else:guess=data['support_native_xy'][row,slot]
            forward=match_patch(src,xy,dst,guess,epipolar_line=epipolar(source,xy,src.shape,j,dst.shape) if a.epipolar else None,subpixel=a.subpixel)
            if forward is None:continue
            counts['forward']+=1
            backward=match_patch(dst,forward[0],src,xy,epipolar_line=epipolar(j,forward[0],dst.shape,source,src.shape) if a.epipolar else None,subpixel=a.subpixel)
            if backward is None or np.linalg.norm(backward[0]-xy)>1:continue
            counts['roundtrip']+=1;observations.append((j,forward[0],dst.shape))
        if len(observations)<3:continue
        counts['three_views']+=1
        centers=[cams[i]['position'] for i,_,_ in observations]
        directions=[ray(i,uv,shape) for i,uv,shape in observations]
        def reprojection_errors(xyz):
            result=[]
            for i,uv,shape in observations:
                c=cams[i];q=(xyz-np.asarray(c['position']))@np.asarray(c['rotation']);h,w=shape[:2]
                if q[2]<=0:result.append(float('inf'));continue
                projected=np.array([(q[0]/q[2]*c['fx']+c['cx'])*w/c['width']-.5,
                                    (q[1]/q[2]*c['fy']+c['cy'])*h/c['height']-.5])
                result.append(np.linalg.norm(projected-uv))
            return np.asarray(result)
        if a.consensus:
            fit=triangulate_consensus(centers,directions,reprojection_errors)
            if fit is None:continue
            xyz,inliers,_=fit
            observations=[o for o,k in zip(observations,inliers) if k]
        else:
            xyz=triangulate_rays(centers,directions)
        if xyz is None:continue
        errors=[]
        for i,uv,shape in observations:
            c=cams[i];q=(xyz-np.asarray(c['position']))@np.asarray(c['rotation']);h,w=shape[:2]
            projected=np.array([(q[0]/q[2]*c['fx']+c['cx'])*w/c['width']-.5,
                                (q[1]/q[2]*c['fy']+c['cy'])*h/c['height']-.5])
            errors.append(np.linalg.norm(projected-uv))
        if max(errors)>1.5:continue
        counts['reprojected']+=1;kept.append(row);positions.append(xyz)
        matched_rows.append(dict(row=row,observations=[dict(view=i,xy=uv.tolist()) for i,uv,_ in observations],
                                 max_reprojection_error=max(errors)))
        if row%100==0:print(json.dumps(dict(row=row,counts=counts)),flush=True)
    ids=np.asarray(kept,dtype=np.int64)
    payload={k:v[ids] for k,v in data.items()}
    payload['prior_xyz']=payload['xyz'].copy();payload['xyz']=np.asarray(positions,dtype=np.float64).reshape(-1,3)
    payload['prior_support_views']=payload['support_views'].copy()
    payload['prior_support_native_xy']=payload['support_native_xy'].copy()
    payload['prior_depth_camera_z']=payload['depth_camera_z'].copy()
    payload['prior_native_pixel_sigma']=payload['native_pixel_sigma'].copy()
    payload['support_native_xy']=payload['support_native_xy'].astype(np.float64)
    payload['support_views'].fill(-1);payload['support_native_xy'].fill(-1)
    for row,record in enumerate(matched_rows):
        for slot,obs in enumerate(record['observations'][1:]):
            payload['support_views'][row,slot]=obs['view']
            payload['support_native_xy'][row,slot]=obs['xy']
        source=int(payload['source_view'][row]);c=cams[source]
        q=(payload['xyz'][row]-np.asarray(c['position']))@np.asarray(c['rotation'])
        payload['depth_camera_z'][row]=q[2]
    payload['native_pixel_sigma']=rebind_angular_sigma(payload['prior_native_pixel_sigma'],
        payload['prior_depth_camera_z'],payload['depth_camera_z'])
    np.savez_compressed(a.output/'rgb_triangulated_proposals.npz',**payload)
    report=dict(scope='training_only_roundtrip_RGB_correspondence_and_triangulation__not_production',counts=counts,
        triangulation_policy='source_anchored_triplet_consensus' if a.consensus else 'all_observations',
        epipolar_pixel_gate=1.5 if a.epipolar else None,
        subpixel_step=.25 if a.subpixel else None,
        support_search='nearest12_training_RGB_without_depth_acceptance' if a.expand_support else 'prior_depth_accepted_views',
        scale_policy='preserve_source_angular_footprint_after_retriangulation__not_verified_shape',
        observations=matched_rows,training_views=m['training_views'],excluded_views=m['excluded_views'],
        limitations=['Local search initialized by predicted geometry may retain correspondence bias',
                     'Repeated texture can still match consistently; independent visibility audit required',
                     'Support pixels and source depth are rebound to the new geometry; prior arrays are provenance only',
                     'No geometry inserted or source model changed'])
    (a.output/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(counts),flush=True)


if __name__=='__main__':main()
