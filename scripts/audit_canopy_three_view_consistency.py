"""Read-only three-view RGB consistency; not a motion oracle or training gate."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.audit_canopy_epipolar_consistency import fundamental_from_cameras,sampson_pixel_distance


def flow_correspondence(left,right):
    flow=cv2.calcOpticalFlowFarneback(left,right,None,.5,5,21,5,7,1.5,0)
    reverse=cv2.calcOpticalFlowFarneback(right,left,None,.5,5,21,5,7,1.5,0)
    h,w=left.shape;yy,xx=np.mgrid[:h,:w].astype(np.float32)
    grid=np.stack((xx,yy),-1);destination=grid+flow;u,v=destination[...,0],destination[...,1]
    back=cv2.remap(reverse,u,v,cv2.INTER_LINEAR)
    reference=left.astype(np.float32)/255
    warped=cv2.remap(right.astype(np.float32)/255,u,v,cv2.INTER_LINEAR)
    blur=lambda x:cv2.boxFilter(x,-1,(11,11))
    mx,my=blur(reference),blur(warped)
    vx=(blur(reference**2)-mx**2).clip(0);vy=(blur(warped**2)-my**2).clip(0)
    ncc=(blur(reference*warped)-mx*my)/np.sqrt(vx*vy).clip(1.e-8)
    reliable=(np.linalg.norm(flow+back,axis=-1)<.3)&(ncc>.9)&(vx>.03**2)&(vy>.03**2)&(u>=6)&(v>=6)&(u<w-6)&(v<h-6)
    return grid,destination,flow,reliable


def three_view_cycle_error(ab,bc,ac):
    if ab.shape!=bc.shape or ab.shape!=ac.shape or ab.ndim!=3 or ab.shape[-1]!=2:
        raise ValueError('Aligned HxWx2 flow arrays required')
    yy,xx=np.mgrid[:ab.shape[0],:ab.shape[1]].astype(np.float32)
    continuation=cv2.remap(bc,xx+ab[...,0],yy+ab[...,1],cv2.INTER_LINEAR)
    return np.linalg.norm(ab+continuation-ac,axis=-1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cameras','dataset','masks','training-metrics','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);cv2.setNumThreads(4)
    cameras=json.loads(a.cameras.read_text());training=json.loads(a.training_metrics.read_text())
    allowed=set(training.get('sampled_view_indices',training.get('source_view_indices',[])))
    if not allowed:raise ValueError('Explicit training-only cohort required')
    masks=CambridgeMaskLookup(a.dataset,a.masks)
    def regions(i):
        c=cameras[i];obj,sky,dist,tree=masks.get_index_masks(c['img_name'],(0,1,2,3),(c['height'],c['width']),torch.device('cpu'))
        known=(obj&sky&dist).numpy();return known&~tree.numpy(),known&tree.numpy()
    triples=[(i,i+1,i+2) for i in sorted(allowed) if {i+1,i+2}<=allowed
        and len({cameras[k]['img_name'].split('__')[0] for k in (i,i+1,i+2)})==1 and regions(i)[0].sum()>=128]
    if not triples:raise ValueError('No training-only adjacent triples')
    triples=[triples[k] for k in np.linspace(0,len(triples)-1,min(12,len(triples))).round().astype(int)]
    records=[]
    for i,j,k in triples:
        c=cameras[i];w,h=c['width'],c['height']
        def read(index):
            path=a.dataset/'images'/(cameras[index]['img_name']+'.png');image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            if image is None:raise FileNotFoundError(path)
            return cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA)
        left,middle,right=read(i),read(j),read(k)
        grid,b,ab,valid_ab=flow_correspondence(left,middle)
        _,dest,ac,valid_ac=flow_correspondence(left,right)
        _,_,bc,valid_bc=flow_correspondence(middle,right)
        cycle=three_view_cycle_error(ab,bc,ac)
        reliable=valid_ab&valid_ac&(cycle<.3)&(cv2.remap(valid_bc.astype(np.uint8),b[...,0],b[...,1],cv2.INTER_NEAREST)>0)
        errors=[sampson_pixel_distance(grid,d,fundamental_from_cameras(c,cameras[t])) for d,t in [(b,j),(dest,k)]]
        row={'views':[i,j,k]}
        for channel,label in enumerate(('canopy','rigid')):
            core=cv2.erode(regions(i)[channel].astype(np.uint8),np.ones((11,11),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
            keep=reliable&core
            for d,t in [(b,j),(dest,k)]:
                mask=cv2.erode(regions(t)[channel].astype(np.uint8),np.ones((11,11),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
                keep&=cv2.remap(mask,d[...,0],d[...,1],cv2.INTER_NEAREST)>0
            values=np.maximum(errors[0],errors[1])[keep]
            row[label]={'matches':len(values),'median_max_pair_error_px':float(np.median(values)) if len(values) else None,
                'p90_max_pair_error_px':float(np.quantile(values,.9)) if len(values) else None,
                'both_pairs_over_1px_fraction':float(((errors[0][keep]>1)&(errors[1][keep]>1)).mean()) if len(values) else None}
        records.append(row);print(json.dumps(row),flush=True)
    (a.output/'audit.json').write_text(json.dumps({'scope':'training_only_three_view_cycle_and_patch_consistency__not_ground_truth_motion',
        'reliability':{'forward_backward_px':.3,'three_view_cycle_px':.3,'patch_ncc':.9,'patch_width':11,'minimum_std':.03},
        'records':records},indent=2))


if __name__=='__main__':main()
