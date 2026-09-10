"""Image-only local plane sweeps around observed versus monocular depths.

Read-only diagnostic: sparse feature depth nominates a search center, never
authorizes material or rigid edits. All source/target views are training views.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / '2d-gaussian-splatting')]
import torch
import torch.nn.functional as F
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.canopy_patch_stereo import candidate_patch_scores, unambiguous_depth_matches
from outdoor.moge3_evidence import sha256_file
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.evaluate_canopy_validation import FIXED, ADDITIONAL


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(p)
    p.set_defaults(data_device='cpu', resolution=640, white_background=True)
    for key in ('tracks', 'prior-audit', 'masks', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--source', type=int, default=665)
    p.add_argument('--targets', type=int, nargs='+', default=[664, 666])
    a = p.parse_args()
    if len(set(a.targets)) < 2 or a.source in a.targets:
        raise ValueError('Two distinct non-source target cameras required')
    if set([a.source, *a.targets]) & set(FIXED + ADDITIONAL):
        raise ValueError('Validation cameras cannot supply matching evidence')
    tracks = json.loads(a.tracks.read_text())
    prior = json.loads(a.prior_audit.read_text())
    if prior['track_audit_sha256'] != sha256_file(a.tracks):
        raise ValueError('Prior and actual observations differ')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.15)
    a.output.mkdir(parents=True, exist_ok=False)
    dataset = model.extract(a)
    dataset.model_path = str(a.output)
    views = LazyScene(dataset, GaussianModel(dataset.sh_degree), image_cache_size=4).getTrainCameras()
    selected = [views[i] for i in [a.source, *a.targets]]
    if any(not v.image_name.startswith('seq2__') for v in selected):
        raise ValueError('Canonical training cameras required')
    masks = CambridgeMaskLookup(Path(dataset.source_path), a.masks)
    def core(view):
        obj, sky, dist, tree = masks.get_index_masks(view.image_name, (0,1,2,3),
            (view.image_height, view.image_width), torch.device('cuda'))
        return F.max_pool2d((~(obj & sky & dist & ~tree)).float()[None,None],5,1,2)[0,0] == 0
    source = selected[0]
    source_core = core(source)
    targets_core = [core(v) for v in selected[1:]]
    observations = {(r['track_id'], r['index']): r for r in prior['records']}
    records = []
    offsets = torch.cartesian_prod(torch.arange(-8,9,device='cuda'), torch.arange(-8,9,device='cuda'))
    for track in tracks['records']:
        obs = {o['index']: o for o in track['observations'] if o['canopy']}
        if not set([a.source,*a.targets]) <= set(obs):
            continue
        row = observations.get((track['track_id'], a.source))
        if row is None:
            continue
        uv = torch.tensor(obs[a.source]['uv'],device='cuda') + offsets
        xy = uv.round().long()
        inside = (xy[:,0]>=2)&(xy[:,0]<source.image_width-2)&(xy[:,1]>=2)&(xy[:,1]<source.image_height-2)
        uv = uv[inside]; xy = xy[inside]
        uv = uv[source_core[xy[:,1],xy[:,0]]]
        if not len(uv):
            continue
        for radius in (.15, .4):
            shifts = torch.linspace(-radius,radius,61,device='cuda')
            separation = max(2, round(.075/(2*radius/60)))
            for label, center in [('observed',row['track_depth']),('moge',row['calibrated_moge'])]:
                depths = torch.full((len(uv),),center,device='cuda')
                matches = []
                scores_all = []
                z = depths[:,None]*shifts.exp()[None]
                rays = torch.stack(((uv[:,0]-source.cx)/source.focal_x,
                                    (uv[:,1]-source.cy)/source.focal_y,torch.ones_like(uv[:,0])),dim=1)
                world = (rays[:,None]*z[:,:,None])@source.world_view_transform[:3,:3].T + source.camera_center
                for target, target_core in zip(selected[1:],targets_core):
                    scores = candidate_patch_scores(source,target,uv,depths,shifts)
                    xyz = world@target.world_view_transform[:3,:3]+target.world_view_transform[3,:3]
                    safe = xyz[...,2].clamp_min(1e-6)
                    x = (xyz[...,0]/safe*target.focal_x+target.cx).round().long()
                    y = (xyz[...,1]/safe*target.focal_y+target.cy).round().long()
                    valid = (xyz[...,2]>.2)&(x>=0)&(x<target.image_width)&(y>=0)&(y<target.image_height)
                    valid &= target_core[y.clamp(0,target.image_height-1),x.clamp(0,target.image_width-1)]
                    # Uniqueness considers ALL photometric alternatives, even
                    # non-tree ones. Semantics must not hide competing peaks.
                    matches.append(unambiguous_depth_matches(scores,separation_bins=separation)&valid)
                    scores_all.append(scores)
                common = torch.stack(matches).all(dim=0)
                accepted = common.any(dim=1)
                score = torch.stack(scores_all).amin(dim=0)
                best = torch.where(common,score,torch.full_like(score,-2.)).argmax(dim=1)
                chosen = z[torch.arange(len(uv),device='cuda'),best]
                records.append(dict(track_id=track['track_id'],center_kind=label,center_depth=center,
                    log_radius=radius,source_pixels=len(uv),joint_matches=int(accepted.sum()),
                    individual_matches=[int(m.any(dim=1).sum()) for m in matches],
                    median_depth=float(chosen[accepted].median()) if accepted.any() else None,
                    median_min_ncc=float(score[torch.arange(len(uv),device='cuda'),best][accepted].median()) if accepted.any() else None))
                print(json.dumps(records[-1]),flush=True)
    report=dict(scope='read_only_patch_sweep_not_material_or_rigid_correction',source=a.source,
        targets=a.targets,records=records,track_sha256=sha256_file(a.tracks),prior_sha256=sha256_file(a.prior_audit),
        limitations=['Local tangent planes may fail on thin nonplanar moving foliage',
                    'Sparse neighborhood center depth is a hypothesis, not dense truth',
                    'Repeated texture and occlusion can invalidate photometric correspondences'])
    (a.output/'audit.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
