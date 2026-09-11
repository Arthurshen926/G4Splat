"""CPU audit: fixed MoGe target rays after candidate motion."""
import argparse
import json
from pathlib import Path
import torch
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    state=torch.load(a.run/'candidates_0800.pth',map_location='cpu',weights_only=False)
    cameras=json.loads((a.run/'cameras.json').read_text())
    observations=torch.load(a.run/'moge_position_observations.pth',map_location='cpu',weights_only=False)
    runtime=Path(state['manifest']['args']['runtime_cache'])
    cache=json.loads(runtime.read_text())
    depths=np.load(runtime.parent/'depth_m.npy',mmap_mode='r')
    s=state['candidate'];initial=s['cloud.xyz']
    final=initial+state['manifest']['args']['position_radius_sigmas']*s['cloud.scales']*s['position_code'].tanh()
    counts=dict(observations=0,over_half_pixel=0,over_one_pixel=0,over_three_pixels=0)
    shifts=[]
    depth_counts=dict(valid_depth_pairs=0,new_depth_differs_over3percent=0,new_depth_differs_over10percent=0)
    for index,(ids,lo,hi) in observations.items():
        if not len(ids):continue
        c=cameras[index];r=torch.tensor(c['rotation']);t=torch.tensor(c['position'])
        first=(initial[ids]-t)@r;last=(final[ids]-t)@r
        # Audit angular motion in runtime pixel units; no invented visibility.
        focal=torch.tensor([c['fx'],c['fy']]) if 'fx' in c else torch.tensor([c['focal_x'],c['focal_y']])
        displacement=((last[:,:2]/last[:,2,None]-first[:,:2]/first[:,2,None])*focal).abs().amax(-1)
        counts['observations']+=len(ids)
        for key,limit in [('over_half_pixel',.5),('over_one_pixel',1.),('over_three_pixels',3.)]:
            counts[key]+=int((displacement>limit).sum())
        shifts.append(displacement)
        center=torch.tensor([c['cx'],c['cy']])
        pixels=[(q[:,:2]/q[:,2,None]*focal+center-.5).round().long() for q in (first,last)]
        valid=torch.ones(len(ids),dtype=torch.bool)
        for q in pixels:valid&=(q[:,0]>=0)&(q[:,0]<c['width'])&(q[:,1]>=0)&(q[:,1]<c['height'])
        d=torch.from_numpy(np.array(depths[cache['camera_order'].index(c['img_name'])],copy=True))
        before,after=[d[q[valid,1],q[valid,0]] for q in pixels]
        ok=torch.isfinite(before)&torch.isfinite(after)&(before>0)&(after>0)
        gap=(after[ok]/before[ok]-1).abs()
        depth_counts['valid_depth_pairs']+=int(ok.sum())
        depth_counts['new_depth_differs_over3percent']+=int((gap>.03).sum())
        depth_counts['new_depth_differs_over10percent']+=int((gap>.1).sum())
    all_shifts=torch.cat(shifts)
    result=dict(run=str(a.run),**counts,**depth_counts,median_pixels=float(all_shifts.median()),
                p95_pixels=float(torch.quantile(all_shifts,.95)),
                limitation='Projection displacement only; not visibility or exact target pixel reassociation')
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
