"""Training RGB correspondence test of three conditional depth hypotheses."""
import argparse
from collections import OrderedDict
import io,json
from pathlib import Path
from types import SimpleNamespace
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.extract_canopy_render_state import DeferredUnpickler
from scripts.canopy_photometric_depth_search import photometric_depth_cost
from scripts.audit_moge_rigid_multiview import nearest_other_cameras


def matched_neighbor_cost(costs):
    """J x N x K: every hypothesis uses exactly the same valid neighbors."""
    common=torch.isfinite(costs).all(-1)
    count=common.sum(0)
    mean=torch.where(common[...,None],costs,torch.zeros_like(costs)).sum(0)/count.clamp_min(1)[:,None]
    return mean.masked_fill((count<2)[:,None],float('nan')),count


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','geometry-audit','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    m=json.loads((a.run/'manifest.json').read_text());c=json.loads((a.run/'cameras.json').read_text())
    train=m['training_views'];assert not set(train)&set(m['excluded_views'])
    init=json.loads((Path(m['args']['initialization'])/'initialization_manifest.json').read_text())
    reader=torch._C.PyTorchFileReader(init['foliage_seed'])
    profiles=DeferredUnpickler(io.BytesIO(reader.get_record('data.pkl'))).load()['audit']['fresh_canonical_leaves']['depth_profiles_by_image']
    data=np.load(a.geometry_audit/'depth_comparison.npz')['samples']
    masks=CambridgeMaskLookup(Path(m['args']['source_path']),Path(m['args']['masks']))
    nearest=nearest_other_cameras(np.array([c[i]['position'] for i in train]),12)
    cache=OrderedDict()
    def load(index):
        if index in cache:
            value=cache.pop(index);cache[index]=value;return value
        row=c[index];matrix=torch.eye(4);matrix[:3,:3]=torch.tensor(row['rotation'])
        view=SimpleNamespace(cx=row['cx']-.5,cy=row['cy']-.5,focal_x=row['fx'],focal_y=row['fy'],
            camera_center=torch.tensor(row['position']),world_view_transform=matrix)
        path=Path(m['args']['source_path'])/'images'/(row['img_name']+'.png')
        rgb=torch.tensor(np.array(Image.open(path).convert('RGB').resize((640,360),Image.Resampling.BILINEAR)),dtype=torch.float32).permute(2,0,1)/255
        fields=masks.get_index_masks(row['img_name'],(0,1,2,3),(360,640),torch.device('cpu'))
        valid=fields[0]&fields[1]&fields[2]&fields[3]
        value=(view,rgb,valid);cache[index]=value
        while len(cache)>32:cache.popitem(last=False)
        return value
    rows=[];ray_records=[]
    for index in np.unique(data[:,0].astype(int)):
        if index not in train:raise ValueError('Evaluation camera in photometric test')
        points=data[(data[:,0]==index)&(data[:,5]>.9)&(data[:,4]>0)]
        if not len(points):continue
        source,rgb,_=load(index)
        uv=torch.tensor((points[:,1:3]+.5)/3-.5,dtype=torch.float32)
        depths=torch.tensor(np.column_stack((points[:,3],points[:,3]*profiles[c[index]['img_name']],points[:,4])),dtype=torch.float32)
        scores=[];neighbors=[train[j] for j in nearest[train.index(index)]]
        for j in neighbors:
            neighbor=load(j)
            # Reuse patch projection for ONE neighbor; its duplicate is not
            # counted as independent support. Count unique neighbors below.
            cost,_=photometric_depth_cost(source,uv,depths,rgb,[neighbor,neighbor])
            scores.append(cost)
        cost,count=matched_neighbor_cost(torch.stack(scores));valid=count>=2
        ray_records.append(np.column_stack((np.full(int(valid.sum()),index),points[valid.numpy(),1:3],
            depths[valid].numpy(),cost[valid].numpy(),count[valid].numpy())))
        row=dict(index=int(index),rays=len(points),common_supported=int(valid.sum()),
            mean_cost=cost[valid].mean(0).tolist() if valid.any() else None,
            wins=torch.bincount(cost[valid].argmin(1),minlength=3).tolist() if valid.any() else [0,0,0])
        rows.append(row);print(json.dumps(row),flush=True)
    report=dict(scope='training_only_patch_reprojection__not_geometry_truth',hypotheses=['raw_global_moge','rigid_anchor_calibrated_moge','source_surface'],
        per_view=rows,warning='Shared valid neighbors; plane patches, repeated textures and occlusion can still mislead; calibrated depths use source anchors')
    np.savez_compressed(a.output/'matched_ray_costs.npz',rays=np.concatenate(ray_records))
    report['ray_columns']=['index','native_x','native_y','raw_z','calibrated_z','source_z',
                           'raw_cost','calibrated_cost','source_cost','common_neighbors']
    (a.output/'audit.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
