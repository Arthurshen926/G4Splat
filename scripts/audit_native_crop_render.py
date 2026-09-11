"""Read-only full-frame versus crop renderer identity audit; no model updates."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import numpy as np
from PIL import Image
import torch
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from scripts.native_resolution_crop_camera import NativeCropCamera


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=640,white_background=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--view',type=int,default=660)
    p.add_argument('--backward-smoke',action='store_true')
    p.add_argument('--box',type=int,nargs=4,default=[900,600,1540,960])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.42)
    dataset=model.extract(a);dataset.model_path=str(a.output)
    teacher=load_hybrid_teacher(a.checkpoint,sh_degree=dataset.sh_degree)
    view=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=1).getTrainCameras()[a.view]
    path=Path(a.source_path)/'images'/(view.image_name+'.png')
    rgb=torch.from_numpy(np.array(Image.open(path).convert('RGB'),copy=True)).permute(2,0,1).float()/255
    h,w=rgb.shape[-2:]
    full=NativeCropCamera(view,rgb,(0,0,w,h))
    crop=NativeCropCamera(view,rgb,tuple(a.box))
    full_rgb=teacher.render(full,task=None,conditioned=False)['rgb'].cpu()
    crop_rgb=teacher.render(crop,task=None,conditioned=False)['rgb'].cpu()
    l,t,r,b=a.box;expected=full_rgb[:,t:b,l:r];error=(expected-crop_rgb).abs()
    report=dict(view=a.view,box=a.box,max_abs=float(error.max()),mean_abs=float(error.mean()),
                interior_max_abs=float(error[:,16:-16,16:-16].max()),
                limitation='Source checkpoint only; no optimization or candidate quality claim')
    if a.backward_smoke:
        for parameter in teacher.foliage.parameters():parameter.requires_grad_(False)
        for name in ('features','opacity_logits'):
            getattr(teacher.foliage,name).requires_grad_(True)
        torch.cuda.reset_peak_memory_stats()
        with torch.enable_grad():
            prediction=teacher.render(full,task=None,conditioned=False)['rgb']
            loss=(prediction-rgb.to(prediction.device)).abs().mean()
            loss.backward()
        report['backward_smoke']=dict(loss=float(loss),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            gradients={name:dict(finite=bool(torch.isfinite(getattr(teacher.foliage,name).grad).all()),
                                l1=float(getattr(teacher.foliage,name).grad.abs().sum()))
                       for name in ('features','opacity_logits')},optimizer_step=False)
    (a.output/'audit.json').write_text(json.dumps(report,indent=2))
    panel=torch.cat([rgb[:,t:b,l:r],expected,crop_rgb],2).permute(1,2,0).clamp(0,1).numpy()
    Image.fromarray((panel*255).round().astype('uint8')).save(a.output/'comparison.png')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
