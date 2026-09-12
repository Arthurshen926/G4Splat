"""Full-frame render, fixed training-observation local metrics (not holdout)."""
import argparse,json,sys
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import numpy as np
import torch
from PIL import Image
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import render_hybrid,static_detail_forward_visibility_gate
from outdoor.native_detail_suffix import NativeDetailSuffix


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=1920,white_background=True)
    for key in ['experiment','control','association','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.5)
    states=[torch.load(r/'suffix.pth',map_location='cpu',weights_only=False) for r in [a.experiment,a.control]]
    if [s['manifest']['refine_proposals'] for s in states]!=[True,False]:raise ValueError('Wrong arms')
    for s in states:
        if s['manifest']['surface_geometry'] or s['manifest']['refined_canopy'] is not None:raise ValueError('Unsupported source changes')
    m=states[0]['manifest'];teacher=load_hybrid_teacher(m['source_checkpoint'],sh_degree=a.sh_degree)
    data=np.load(a.association/'rgb_triangulated_proposals.npz')
    keep=data['normal_valid'].astype(bool);tensor=lambda v:torch.as_tensor(v,dtype=torch.float32,device='cuda')
    suffixes=[]
    for s in states:
        if s['manifest']['proposal_sha256']!=m['proposal_sha256'] or s['step']!=states[0]['step']:raise ValueError('Unmatched proposals or step')
        suffix=NativeDetailSuffix(teacher.surface,tensor(data['xyz'][keep]),tensor(data['normal'][keep]),
            tensor(data['rgb'][keep]),tensor(data['native_pixel_sigma'][keep]),refine_proposals=s['manifest']['refine_proposals'])
        for key in ['dc','logit']:getattr(suffix,key).copy_(s[key].cuda())
        suffix.position_delta.copy_(s['proposal_position_delta'].cuda());suffix.scale_delta.copy_(s['proposal_scale_delta'].cuda())
        suffixes.append(suffix)
    observations=defaultdict(list)
    for r,valid in zip(json.loads((a.association/'audit.json').read_text())['observations'],keep):
        if valid:
            for o in r['observations']:observations[o['view']].append(o['xy'])
    selected=sorted(observations,key=lambda i:(-len(observations[i]),i))[:8]
    dataset=model.extract(a);dataset.model_path=str(a.output)
    views=LazyScene(dataset,GaussianModel(a.sh_degree),image_cache_size=4).getTrainCameras();records=[]
    for i in selected:
        if i not in m['training_views'] or i in m['excluded_views']:raise ValueError('Training scope mismatch')
        v=views[i];target=v.original_image.cuda();h,w=v.image_height,v.image_width
        mask=torch.zeros(h,w,device='cuda',dtype=torch.bool)
        for uv in observations[i]:
            x,y=np.rint(uv).astype(int);mask[max(0,y-4):min(h,y+5),max(0,x-4):min(w,x+5)]=True
        gate=static_detail_forward_visibility_gate(teacher.foliage,int(v.colmap_id),include_pending_exact=False)
        images=[];metrics=[]
        for surface in [teacher.surface]+suffixes:
            out=render_hybrid(v,surface,teacher.foliage,volume_gate=gate,background=torch.zeros(3,device='cuda'),
                include_dynamic=False,optical_replacement_policy='disabled',structural_trainable_start=None)
            rgb=(out.render+(1-out.alpha)*teacher.sky(v)).clamp(0,1)
            metrics.append(float(-10*torch.log10((rgb[:,mask]-target[:,mask]).square().mean().clamp_min(1e-12))))
            images.append(rgb)
        record=dict(view=i,pixels=int(mask.sum()),source=metrics[0],experiment=metrics[1],control=metrics[2],delta=metrics[1]-metrics[2])
        records.append(record);print(json.dumps(record),flush=True)
        panel=torch.cat([target]+images,2).permute(1,2,0).cpu().numpy()
        Image.fromarray((panel*255).round().astype('uint8')).save(a.output/f'view_{i}.png')
    report=dict(scope='eight_train_views_selected_by_observation_count__local9x9_RGB__not_generalization',records=records,
        mean_delta=float(np.mean([r['delta'] for r in records])))
    (a.output/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
