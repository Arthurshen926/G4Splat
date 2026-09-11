"""Batch pure-rigid native prefixes for previously selected TRAINING rays.

No candidate/model update, no new leaf target and no inference masking.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import torch
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import render_hybrid
from outdoor.moge3_evidence import sha256_file
from scripts.canopy_native_surface_prefix import surface_api,native_prefix,prefix_rgb


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__);model=ModelParams(parser)
    parser.set_defaults(data_device='cpu',resolution=640,white_background=True)
    for key in ('run','constraints','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--reference-prefix',type=Path)
    parser.add_argument('--only-reference-rays',action='store_true')
    args=parser.parse_args()
    if args.only_reference_rays and args.reference_prefix is None:raise ValueError('Explicit probe reference required')
    manifest=json.loads((args.run/'manifest.json').read_text());source=manifest['args']['checkpoint']
    constraints=json.loads(args.constraints.read_text())
    if (constraints['source_checkpoint_sha256']!=manifest['source_checkpoint_sha256']
            or sha256_file(source)!=manifest['source_checkpoint_sha256']
            or set(constraints['training_views'])!=set(manifest['training_views'])
            or set(manifest['training_views'])&set(manifest['excluded_views'])
            or Path(args.source_path).resolve()!=Path(manifest['args']['source_path']).resolve()):
        raise ValueError('Same immutable source scene and excluded-view contract required')
    references={}
    if args.reference_prefix is not None:
        reference=json.loads(args.reference_prefix.read_text())
        for row in reference['records']:
            ray=row['ray'];references[(ray['view'],ray['x'],ray['y'])]=row
    rays=constraints['rays']
    if args.only_reference_rays:
        rays=[r for r in rays if (r['view'],r['x'],r['y']) in references]
        if len(rays)!=len(references):raise ValueError('Reference rays do not match source constraints')
    if not rays or any(r['view'] not in manifest['training_views'] for r in rays):
        raise ValueError('Nonempty training-only rays required')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
    api=surface_api();teacher=load_hybrid_teacher(source,sh_degree=args.sh_degree)
    surface=teacher.surface;base=teacher.foliage
    if getattr(surface,'_chart_geometry_provider',None) is not None:
        raise ValueError('Chart overrides require an explicit diagnostic geometry contract')
    dataset=model.extract(args);dataset.model_path=str(args.output)
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=4).getTrainCameras()
    xyz=surface.get_xyz.contiguous();scales=surface.get_scaling.contiguous()
    quat=surface.get_rotation.contiguous();opacity=surface.get_opacity.reshape(-1).contiguous()
    from utils.sh_utils import eval_sh
    common=dict(background=torch.zeros(3,device='cuda'),include_dynamic=False,
                optical_replacement_policy='disabled',structural_trainable_start=None,
                volume_gate=torch.zeros(len(base),device='cuda'))
    groups={}
    for ray in rays:groups.setdefault(ray['view'],[]).append(ray)
    records=[];prefixes=[];maximum_rgb_error=maximum_probe_error=0.
    for ordinal,(index,selected) in enumerate(groups.items()):
        v=views[index];h,w=v.image_height,v.image_width
        projected=api.diagnostic_surface_support(xyz,scales,quat,opacity,
            v.world_view_transform.contiguous(),v.full_proj_transform.contiguous(),w,h)
        direction=torch.nn.functional.normalize(xyz-v.camera_center.reshape(1,3),dim=-1)
        degree=min(int(surface.active_sh_degree),int(base.sh_degree))
        colors=(eval_sh(degree,surface.get_features.transpose(1,2),direction)+.5).clamp_min(0)
        for start in range(0,len(selected),8):
            batch=selected[start:start+8];fields=torch.zeros(len(batch),h,w,device='cuda')
            for j,ray in enumerate(batch):
                if not (0<=ray['x']<w and 0<=ray['y']<h):raise ValueError('Ray outside original image')
                fields[j,ray['y'],ray['x']]=1
            package=render_hybrid(v,surface,base,audit_fields=fields,**common)
            for j,ray in enumerate(batch):
                x,y=ray['x'],ray['y'];key=(index,x,y)
                p=native_prefix(projected,opacity,colors,package.responsibility[:len(xyz),j+1],(x,y))
                total=prefix_rgb(p,1e10);expected=package.render[:,y,x]
                error=float((total-expected).abs().max());maximum_rgb_error=max(maximum_rgb_error,error)
                torch.testing.assert_close(total,expected,atol=3e-5,rtol=3e-5)
                torch.testing.assert_close(expected,torch.tensor(ray['wall_rgb'],device='cuda'),atol=3e-5,rtol=3e-5)
                target=v.original_image[:,y,x].cuda()
                crossing=(p['cumulative_rgb']>target[None]+.03).any(1).nonzero().flatten()
                frontier=float(p['depths'][crossing[0]]) if len(crossing) else None
                probe_error=None
                if key in references:
                    samples=references[key]['samples']
                    query=torch.tensor([s['depth'] for s in samples],device='cuda')
                    observed=torch.tensor([s['surface_rgb'] for s in samples],device='cuda')
                    probe_error=float((prefix_rgb(p,query)-observed).abs().max())
                    maximum_probe_error=max(maximum_probe_error,probe_error)
                    if probe_error>3e-4:raise ValueError('Prefix disagrees with original native opaque-screen probe')
                record=dict(ray=ray,contributors=len(p['ids']),frontier_depth=frontier,
                    frontier_to_median=frontier/ray['wall_depth'] if frontier is not None else None,
                    native_rgb_reconstruction_error=error,opaque_screen_reference_error=probe_error)
                records.append(record)
                prefixes.append(dict(ray=ray,**{k:value.cpu() for k,value in p.items()}))
            del package
        del projected
        if ordinal%16==0:print(json.dumps(dict(prefix_views=ordinal+1,prefix_rays=len(records))),flush=True)
    artifact=args.output/'surface_prefixes.pth'
    temporary=args.output/'surface_prefixes.pth.tmp'
    torch.save(dict(prefixes=prefixes,source_checkpoint_sha256=manifest['source_checkpoint_sha256']),temporary)
    temporary.replace(artifact)
    report=dict(scope='native_pure_rigid_radiance_prefix__training_only__not_leaf_depth_truth',
        source_checkpoint_sha256=manifest['source_checkpoint_sha256'],constraints_sha256=sha256_file(args.constraints),
        diagnostic_binary_sha256=sha256_file(api.__file__),prefixes_sha256=sha256_file(artifact),
        helper_sha256=sha256_file(Path(__file__).with_name('canopy_native_surface_prefix.py')),
        script_sha256=sha256_file(Path(__file__)),training_views=manifest['training_views'],
        excluded_views=manifest['excluded_views'],maximum_native_rgb_error=maximum_rgb_error,
        maximum_opaque_screen_reference_error=maximum_probe_error,records=records,
        limitations=['Fixed rigid appearance/geometry; does not decide whether foreground rigid proxies are correct',
                    'Exact native contributing depths and weights, with reported floating-point summation tolerance',
                    'Rays selected by prior training diagnostics; not a full image or blind holdout certificate'])
    (args.output/'surface_prefix_batch.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','training_views','excluded_views')}),flush=True)


if __name__=='__main__':main()
