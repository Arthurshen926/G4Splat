"""Read-only CUDA equivalence probe; no training or model changes."""
import argparse,json,time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import torch
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import render_hybrid,static_detail_forward_visibility_gate
from scripts.canopy_frozen_reference_cache import FrozenReferenceCache
from scripts.canopy_reference_bundle_cache import FrozenReferenceBundleCache


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=640,white_background=True)
    for name in ('run','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
    m=json.loads((a.run/'manifest.json').read_text())
    if Path(a.source_path).resolve()!=Path(m['args']['source_path']).resolve():raise ValueError('Changed dataset')
    dataset=model.extract(a);dataset.model_path=str(a.output)
    teacher=load_hybrid_teacher(m['args']['checkpoint'],sh_degree=dataset.sh_degree)
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=1).getTrainCameras()
    ids=m['training_views'][:2];sequence=ids+[ids[0]]
    assert not set(ids)&set(m['excluded_views'])
    calls=0
    def render(index):
        nonlocal calls
        calls+=1;v=views[index]
        gate=static_detail_forward_visibility_gate(teacher.foliage,int(v.colmap_id),include_pending_exact=False)
        out=render_hybrid(v,teacher.surface,teacher.foliage,volume_gate=gate,
            background=torch.zeros(3,device='cuda'),include_dynamic=False,
            optical_replacement_policy='disabled',structural_trainable_start=None)
        return out.render+(1-out.alpha)*teacher.sky(v),out.surface_alpha.reshape(1,v.image_height,v.image_width)
    render(ids[0]);torch.cuda.synchronize();calls=0
    rgb_cache=FrozenReferenceCache(2);alpha_cache=FrozenReferenceCache(2,channels=1)
    old=[];start=time.perf_counter()
    for index in sequence:
        old.append((rgb_cache.get(index,lambda:render(index)[0],'cuda'),
                    alpha_cache.get(index,lambda:render(index)[1],'cuda')))
    torch.cuda.synchronize();old_seconds=time.perf_counter()-start;old_calls=calls;calls=0
    cache=FrozenReferenceBundleCache(2);errors=[];start=time.perf_counter()
    for index,expected in zip(sequence,old):
        actual=cache.get(index,lambda:render(index),'cuda')
        errors.append([float((x-y).abs().max()) for x,y in zip(actual,expected)])
        for x,y in zip(actual,expected):torch.testing.assert_close(x,y,rtol=0,atol=0)
    torch.cuda.synchronize()
    report=dict(scope='reference_outputs_only__not_optimizer_or_quality_validation',views=sequence,
        old_render_calls=old_calls,new_render_calls=calls,max_abs_errors=errors,
        old_seconds=old_seconds,new_seconds=time.perf_counter()-start,
        limitation='Tiny warm probe; elapsed times are not an end-to-end training speedup estimate')
    (a.output/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
