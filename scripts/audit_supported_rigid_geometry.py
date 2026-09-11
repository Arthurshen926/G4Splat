"""Read-only native surface depth against conditional multiview measurements."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import numpy as np
import torch
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import render_hybrid


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=1920,white_background=True)
    for name in ('run','measurements','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--max-views',type=int,default=16)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
    manifest=json.loads((a.run/'manifest.json').read_text())
    measurement_report=json.loads((a.measurements.parent/'audit.json').read_text())
    if Path(measurement_report['conditional_measurements']['source_run']).resolve()!=a.run.resolve():
        raise ValueError('Changed measurement source')
    if Path(a.source_path).resolve()!=Path(manifest['args']['source_path']).resolve():
        raise ValueError('Changed dataset')
    with np.load(a.measurements) as data:measurements=data['measurements']
    indices=np.unique(measurements[:,0].astype(int))
    if not set(indices)<=set(manifest['training_views']) or set(indices)&set(manifest['excluded_views']):
        raise ValueError('Training-only measurements required')
    if a.max_views<=0:raise ValueError('Positive maximum views required')
    indices=indices[np.linspace(0,len(indices)-1,min(a.max_views,len(indices))).round().astype(int)]
    dataset=model.extract(a);dataset.model_path=str(a.output)
    teacher=load_hybrid_teacher(manifest['args']['checkpoint'],sh_degree=dataset.sh_degree)
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=1).getTrainCameras()
    zero=torch.zeros_like(teacher.foliage.opacity_logits.reshape(-1))
    rows=[];samples=[]
    for index in indices:
        v=views[index];points=measurements[measurements[:,0]==index]
        x=points[:,1].astype(int);y=points[:,2].astype(int)
        if v.image_width!=1920 or v.image_height!=1080:raise ValueError('Native measurement raster mismatch')
        result=render_hybrid(v,teacher.surface,teacher.foliage,background=torch.zeros(3,device='cuda'),
            include_dynamic=False,optical_replacement_policy='disabled',structural_trainable_start=None,
            volume_gate=zero)
        depth=result.median_depth[0].cpu().numpy()[y,x]
        alpha=result.surface_alpha[0].cpu().numpy()[y,x]
        valid=np.isfinite(depth)&(depth>0)&(alpha>.9)
        error=np.abs(depth[valid]/points[valid,3]-1)
        rows.append(dict(index=int(index),measurements=len(points),opaque_surface_samples=int(valid.sum()),
            median_relative_difference=float(np.median(error)) if len(error) else None,
            fraction_difference_above_10percent=float(np.mean(error>.1)) if len(error) else None))
        samples.append(np.column_stack((points[:,0:4],depth,alpha)))
        print(json.dumps(rows[-1]),flush=True)
    np.savez_compressed(a.output/'depth_comparison.npz',samples=np.concatenate(samples))
    report=dict(scope='conditional_depth_disagreement__not_absolute_geometry_error',per_view=rows,
        columns=['source_index','native_x','native_y','moge_camera_z','surface_median_z','surface_alpha'],
        source_checkpoint=manifest['args']['checkpoint'],measurements=str(a.measurements),
        limitation='Uniformly selected training cameras/rays; no geometry updates, no rail-specific accuracy claim')
    (a.output/'audit.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
