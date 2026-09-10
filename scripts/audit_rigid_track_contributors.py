"""Identify native rigid contributors at actual contradictory observations.

This is attribution only. A historical feature can lie behind a legitimate
foreground branch; no surface is removed or exported by this diagnostic.
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
from scripts.evaluate_canopy_validation import FIXED,ADDITIONAL,tensor_digest
from scripts.audit_rigid_free_space import surfel_depth_extent


def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=640,white_background=True)
    for key in ('checkpoint','tracks','support','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--track-id',type=int,required=True)
    p.add_argument('--observation-views',type=int,nargs='+',
                   help='Optional independently photometrically checked observation subset')
    a=p.parse_args();torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
    data=json.loads(a.tracks.read_text())
    track=next(r for r in data['records'] if r['track_id']==a.track_id)
    if a.observation_views is not None:
        chosen=set(a.observation_views)
        track={**track,'observations':[o for o in track['observations'] if o['index'] in chosen]}
        if {o['index'] for o in track['observations']}!=chosen:raise ValueError('Unknown observation view')
    if len(track['observations'])<3 or any(o['index'] in set(FIXED+ADDITIONAL) for o in track['observations']):
        raise ValueError('Independent training observations required')
    a.output.mkdir(parents=True,exist_ok=False)
    teacher=load_hybrid_teacher(a.checkpoint,sh_degree=a.sh_degree)
    surface=teacher.surface;names=('_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest')
    before={name:tensor_digest(getattr(surface,name)) for name in names}
    support=torch.load(a.support,map_location='cpu')
    if not support['complete_scan'] or support['surface_fingerprints']!=before:
        raise ValueError('Whole-view support of this exact surface required')
    dataset=model.extract(a);dataset.model_path=str(a.output)
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=4).getTrainCameras()
    for name in names:getattr(surface,name).requires_grad_(False)
    for param in teacher.foliage.parameters():param.requires_grad_(False)
    dc=surface._features_dc.detach().clone();rest=surface._features_rest.detach().clone();degree=surface.active_sh_degree
    surface._features_dc.requires_grad_(True)
    counts=torch.zeros(len(surface.get_xyz),dtype=torch.int16,device='cuda')
    mass=torch.zeros(len(counts),device='cuda');records=[]
    transforms=surface.get_covariance().detach()
    try:
        with torch.no_grad():
            surface._features_dc.zero_();surface._features_rest.zero_();surface.active_sh_degree=0
        for obs in track['observations']:
            if not obs['canopy']:continue
            view=views[obs['index']]
            if view.image_name!=obs['image_name']:raise ValueError('Camera identity changed')
            x,y=torch.tensor(obs['uv']).round().long().tolist()
            out=render_hybrid(view,surface,teacher.foliage,background=torch.zeros(3,device='cuda'),
                volume_gate=torch.zeros_like(teacher.foliage.opacity_logits.reshape(-1)),
                include_dynamic=False,optical_replacement_policy='disabled',structural_trainable_start=0)
            if float((out.render.detach()-.5*out.surface_alpha.detach()).abs().max())>1.e-5:
                raise RuntimeError('Constant-radiance identity failed')
            g=torch.autograd.grad(out.render[0,y,x],surface._features_dc)[0][:,0,0]/.28209479177387814
            if not torch.isfinite(g).all() or g.min() < -1.e-5:raise RuntimeError('Invalid native responsibility')
            if not torch.allclose(g.sum(),out.surface_alpha.detach()[0,y,x],atol=1.e-5,rtol=2.e-5):
                raise RuntimeError('Responsibility does not conserve pixel alpha')
            _,lower,upper=surfel_depth_extent(transforms,view.world_view_transform)
            cov=torch.tensor(track['covariance'],device='cuda',dtype=torch.float32)
            axis=view.world_view_transform[:3,2]
            sigma=(axis@cov@axis).clamp_min(0).sqrt()
            margin=max(.10,.05*obs['depth'],3*float(sigma))
            contributing=g>1.e-6
            contradiction=contributing&(upper<obs['depth']-margin)
            counts+=contradiction.to(counts.dtype);mass+=g.detach().clamp_min(0)
            ids=torch.nonzero(contributing).flatten()
            records.append(dict(index=obs['index'],track_depth=obs['depth'],margin=margin,
                native_rigid_depth=float(out.median_depth[0,y,x]),visible_rows=int(contributing.sum()),
                wholly_shallower_rows=int(contradiction.sum()),
                wholly_shallower_mass=float(g[contradiction].sum()),
                contributors=[dict(row=int(i),mass=float(g[i]),depth_lower=float(lower[i]),depth_upper=float(upper[i])) for i in ids]))
            print(json.dumps({k:v for k,v in records[-1].items() if k!='contributors'}),flush=True)
            del out,g
    finally:
        with torch.no_grad():
            surface._features_dc.copy_(dc);surface._features_rest.copy_(rest);surface.active_sh_degree=degree
        surface._features_dc.requires_grad_(False)
    if before!={name:tensor_digest(getattr(surface,name)) for name in names}:
        raise RuntimeError('Attribution changed rigid parameters')
    ids=torch.nonzero(counts>=2).flatten().cpu()
    candidates=[dict(row=int(i),contradictory_views=int(counts[i]),total_observation_mass=float(mass[i]),
        known_rigid_views=int(support['rigid_views'][i]),known_rigid_mass=float(support['rigid_mass'][i]),
        canopy_views=int(support['canopy_views'][i]),canopy_mass=float(support['canopy_mass'][i])) for i in ids]
    report=dict(scope='local_native_contributor_attribution__not_removal_authority',track_id=a.track_id,
        checkpoint_sha256=sha256_file(a.checkpoint),tracks_sha256=sha256_file(a.tracks),
        surface_fingerprints=before,records=records,candidates=candidates,
        limitations=['Visible feature does not prove foreground is empty',
                    'Whole-surfel six-sigma depth is conservative, not correspondence truth',
                    'Known building support remains protected; no model parameters changed'])
    (a.output/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(candidate_count=len(candidates),candidates=candidates)),flush=True)


if __name__=='__main__':main()
