"""CPU re-association audit on saved candidate geometry; no fitting."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import numpy as np
import torch
import torch.nn.functional as F
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.seed_bound_moge_diagnostic import configure_seed_bound_moge
from scripts.evaluate_hybrid_teacher import _tree_boundary_masks
from scripts.canopy_refresh_moge_observations import refresh_observations


class Calibration:
    def configure_moge3_metric_scale(self,value):self.scale=value
    def configure_moge3_canopy_depth_scales(self,value):self.profiles=value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    state=torch.load(a.run/'candidates_0800.pth',map_location='cpu',weights_only=False)
    manifest=state['manifest'];args=manifest['args'];s=state['candidate']
    xyz=s['cloud.xyz']+args['position_radius_sigmas']*s['cloud.scales']*s['position_code'].tanh()
    records=torch.load(a.run/'moge_position_observations.pth',map_location='cpu',weights_only=False)
    views=[]
    for c in json.loads((a.run/'cameras.json').read_text()):
        transform=torch.eye(4);transform[:3,:3]=torch.tensor(c['rotation'])
        transform[3,:3]=-torch.tensor(c['position'])@transform[:3,:3]
        views.append(SimpleNamespace(image_name=c['img_name'],world_view_transform=transform,
            camera_center=torch.tensor(c['position']),image_width=c['width'],image_height=c['height'],
            focal_x=c['fx'],focal_y=c['fy'],cx=c['cx'],cy=c['cy']))
    cache=json.loads(Path(args['runtime_cache']).read_text())
    raw=np.load(Path(args['runtime_cache']).parent/cache['arrays']['depth_m']['path'],mmap_mode='r')
    cal=Calibration();configure_seed_bound_moge(cal,args['initialization'],manifest['calibration'],manifest['calibration']['metric_scale'])
    masks=CambridgeMaskLookup(Path(args['source_path']),Path(args['masks']))
    def query(index,v):
        d=torch.from_numpy(np.array(raw[cache['camera_order'].index(v.image_name)],copy=True))*cal.scale*cal.profiles[v.image_name]
        obj,sky,dist,tree=masks.get_index_masks(v.image_name,(0,1,2,3),(v.image_height,v.image_width),torch.device('cpu'))
        canopy=obj&sky&dist&~tree;inside,_,_=_tree_boundary_masks(~canopy)
        valid=torch.isfinite(d)&(d>.2);clean=torch.where(valid,d,torch.zeros_like(d))
        hi=F.max_pool2d(clean[None,None],3,1,1)[0,0]
        lo=-F.max_pool2d(-torch.where(valid,d,torch.full_like(d,float('inf')))[None,None],3,1,1)[0,0]
        good=valid&(F.avg_pool2d(valid.float()[None,None],3,1,1)[0,0]>=1)&(hi/lo.clamp_min(.2)<1.05)&canopy&~inside
        return d,good
    _,report=refresh_observations(xyz,views,records,query,excluded=manifest['excluded_views'],
        initial_xyz=(s['cloud.xyz'] if args.get('moge_position_reachable_only') else None),
        motion_extent=(args['position_radius_sigmas']*s['cloud.scales'] if args.get('moge_position_reachable_only') else None))
    report['run']=str(a.run)
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
