"""Native source surface contribution in front of reassociated detail tracks."""
import argparse,json,sys
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import numpy as np
import torch
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import render_hybrid
from scripts.canopy_native_surface_prefix import surface_api,native_prefix,prefix_rgb
from utils.sh_utils import eval_sh


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=1920,white_background=True)
    for key in ['run','association','heldout','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.5)
    m=json.loads((a.run/'manifest.json').read_text());audit=json.loads((a.association/'audit.json').read_text())
    if audit['training_views']!=m['training_views'] or set(m['training_views'])&set(m['excluded_views']):raise ValueError('Camera contract')
    checks=defaultdict(list)
    for x in json.loads(a.heldout.read_text())['rows']:checks[x['row']].append(x['heldout_error'])
    accepted={row for row,v in checks.items() if len(v)>=4 and all(e is not None and e<=1.5 for e in v)}
    data=np.load(a.association/'rgb_triangulated_proposals.npz');groups=defaultdict(list)
    for i,record in enumerate(audit['observations']):
        if record['row'] not in accepted:continue
        for o in record['observations']:
            if o['view'] not in m['training_views']:raise ValueError('Non-training observation')
            groups[o['view']].append((record['row'],data['xyz'][i],o['xy']))
    teacher=load_hybrid_teacher(m['args']['checkpoint'],sh_degree=a.sh_degree)
    surface=teacher.surface;base=teacher.foliage;api=surface_api()
    dataset=model.extract(a);dataset.model_path=str(a.output)
    views=LazyScene(dataset,GaussianModel(a.sh_degree),image_cache_size=4).getTrainCameras()
    xyz=surface.get_xyz.contiguous();scales=surface.get_scaling.contiguous();quat=surface.get_rotation.contiguous()
    opacity=surface.get_opacity.flatten().contiguous();records=[]
    common=dict(background=torch.zeros(3,device='cuda'),include_dynamic=False,optical_replacement_policy='disabled',
                structural_trainable_start=None,volume_gate=torch.zeros(len(base),device='cuda'))
    for index,observations in groups.items():
        v=views[index];h,w=v.image_height,v.image_width
        projected=api.diagnostic_surface_support(xyz,scales,quat,opacity,v.world_view_transform.contiguous(),v.full_proj_transform.contiguous(),w,h)
        direction=torch.nn.functional.normalize(xyz-v.camera_center[None],dim=-1)
        colors=(eval_sh(min(surface.active_sh_degree,base.sh_degree),surface.get_features.transpose(1,2),direction)+.5).clamp_min(0)
        for start in range(0,len(observations),4):
            batch=observations[start:start+4];fields=torch.zeros(len(batch),h,w,device='cuda')
            for j,(_,_,uv) in enumerate(batch):
                x,y=np.rint(uv).astype(int);fields[j,y,x]=1
            package=render_hybrid(v,surface,base,audit_fields=fields,**common)
            for j,(row,point,uv) in enumerate(batch):
                x,y=np.rint(uv).astype(int)
                prefix=native_prefix(projected,opacity,colors,package.responsibility[:len(xyz),j+1],(x,y))
                torch.testing.assert_close(prefix_rgb(prefix,1e10),package.render[:,y,x],atol=3e-5,rtol=3e-5)
                world=torch.as_tensor(point,device='cuda',dtype=xyz.dtype)
                z=(world@v.world_view_transform[:3,:3]+v.world_view_transform[3,:3])[2]
                front=prefix['depths']<=z
                front_rgb=prefix_rgb(prefix,z);target=v.original_image[:,y,x].cuda()
                records.append(dict(row=row,view=index,xy=[int(x),int(y)],depth=float(z),
                    foreground_surface_weight=float(prefix['weights'][front].sum()),
                    foreground_rgb=front_rgb.cpu().tolist(),target_rgb=target.cpu().tolist(),
                    overbright_prefix=bool((front_rgb>target+.03).any()),
                    foreground_ids=prefix['ids'][front].cpu().tolist()))
            del package
        print(json.dumps(dict(view=index,rays=len(records))),flush=True)
    report=dict(scope='pure_surface_prefix_at_consensus_detail__not_proof_surface_is_wrong',
                tracks=len(accepted),rays=len(records),overbright=sum(r['overbright_prefix'] for r in records),records=records)
    (a.output/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
