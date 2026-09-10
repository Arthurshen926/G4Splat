"""Depth-independent static-scene consistency audit on neighboring training RGB.

Forward/backward-consistent, textured optical-flow matches are compared to the
calibrated epipolar geometry. Residuals can indicate motion, correspondence error,
or camera error; they are NOT proof of wind and do not change training authority.
"""
import argparse,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
from matcha.cambridge_masks import CambridgeMaskLookup


def fundamental_from_cameras(source,target):
    def intrinsics(c):return np.array([[c['fx'],0,c['cx']],[0,c['fy'],c['cy']],[0,0,1.]])
    a=np.asarray(source['rotation']);b=np.asarray(target['rotation'])
    rotation=b.T@a;translation=b.T@(np.asarray(source['position'])-np.asarray(target['position']))
    x,y,z=translation;skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    matrix=np.linalg.inv(intrinsics(target)).T@skew@rotation@np.linalg.inv(intrinsics(source))
    norm=np.linalg.norm(matrix)
    if norm<1.e-12:raise ValueError('No usable epipolar baseline')
    return matrix/norm


def sampson_pixel_distance(source,target,matrix):
    left=np.concatenate((source,np.ones((*source.shape[:-1],1))),axis=-1)
    right=np.concatenate((target,np.ones((*target.shape[:-1],1))),axis=-1)
    l2=left@matrix.T;l1=right@matrix
    error=np.abs((right*l2).sum(axis=-1))
    return error/np.sqrt((l1[...,:2]**2).sum(-1)+(l2[...,:2]**2).sum(-1)).clip(1.e-12)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cameras','dataset','masks','training-metrics','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);cv2.setNumThreads(4)
    cameras=json.loads(a.cameras.read_text());training=json.loads(a.training_metrics.read_text())
    allowed=set(training.get('sampled_view_indices',training.get('sampled_indices',[])))
    # Calibrator records its complete order in the process log, not metrics;
    # use the initializer's explicit source list when an audit is supplied.
    if not allowed:allowed=set(training.get('source_view_indices',[]))
    if not allowed:raise ValueError('Explicit training-only camera cohort required')
    masks=CambridgeMaskLookup(a.dataset,a.masks)
    def regions(i):
        c=cameras[i];obj,sky,dist,tree=masks.get_index_masks(c['img_name'],(0,1,2,3),(c['height'],c['width']),torch.device('cpu'))
        known=(obj&sky&dist).numpy();return known&~tree.numpy(),known&tree.numpy()
    pairs=[]
    for i in sorted(allowed):
        if i+1 not in allowed:continue
        if cameras[i]['img_name'].split('__')[0]!=cameras[i+1]['img_name'].split('__')[0]:continue
        if regions(i)[0].sum()>=128:pairs.append((i,i+1))
    pairs=[pairs[k] for k in np.linspace(0,len(pairs)-1,min(12,len(pairs))).round().astype(int)]
    records=[]
    for i,j in pairs:
        c=cameras[i];w,h=c['width'],c['height']
        def read(k):
            path=a.dataset/'images'/(cameras[k]['img_name']+'.png')
            image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            if image is None:raise FileNotFoundError(path)
            return cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA)
        left=read(i);right=read(j)
        flow=cv2.calcOpticalFlowFarneback(left,right,None,.5,5,21,5,7,1.5,0)
        reverse=cv2.calcOpticalFlowFarneback(right,left,None,.5,5,21,5,7,1.5,0)
        yy,xx=np.mgrid[:h,:w].astype(np.float32);grid=np.stack((xx,yy),axis=-1);destination=grid+flow
        u,v=destination[...,0],destination[...,1]
        back=cv2.remap(reverse,u,v,cv2.INTER_LINEAR);warped=cv2.remap(right.astype(np.float32)/255,u,v,cv2.INTER_LINEAR)
        reference=left.astype(np.float32)/255
        blur=lambda x:cv2.boxFilter(x,-1,(5,5))
        mx,my=blur(reference),blur(warped)
        vx=(blur(reference**2)-mx**2).clip(0);vy=(blur(warped**2)-my**2).clip(0)
        correlation=(blur(reference*warped)-mx*my)/np.sqrt(vx*vy).clip(1.e-8)
        reliable=((np.linalg.norm(flow+back,axis=-1)<.5)&(correlation>.8)&(vx>.02**2)&(vy>.02**2)&(u>=3)&(v>=3)&(u<w-3)&(v<h-3))
        distance=sampson_pixel_distance(grid,destination,fundamental_from_cameras(c,cameras[j]))
        row={'source':i,'target':j}
        for label,mask in zip(('canopy','rigid'),regions(i)):
            target_mask=regions(j)[0 if label=='canopy' else 1].astype(np.uint8)
            same=cv2.remap(target_mask,u,v,cv2.INTER_NEAREST)>0
            keep=reliable&mask&same;values=distance[keep]
            row[label]={'reliable_matches':len(values),'median_px':float(np.median(values)) if len(values) else None,
                'p90_px':float(np.quantile(values,.90)) if len(values) else None,
                'over_2px_fraction':float((values>2).mean()) if len(values) else None}
        records.append(row);print(json.dumps(row),flush=True)
    (a.output/'audit.json').write_text(json.dumps({'scope':'training_only__depth_independent_epipolar_consistency__not_wind_ground_truth',
        'camera_sha256':hashlib.sha256(a.cameras.read_bytes()).hexdigest(),'records':records},indent=2))


if __name__=='__main__':main()
