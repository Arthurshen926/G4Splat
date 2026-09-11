"""Read-only native surface-radiance prefix diagnostic, NOT a geometry target.

A temporary nearly opaque black probe measures which unchanged surface light
is already in front of a depth along one training ray. Other foliage is gated
off. No scene parameter is edited and the virtual probes are never exported.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from outdoor.hybrid_gaussian_renderer import render_hybrid
from scripts.canopy_candidate_diagnostic import CandidateCloud, NativeCandidateView, ray_candidates


@torch.no_grad()
def probe_surface_prefix(teacher, base, views, manifest, constraints_dir, replay_path, output, *, snapshot_sha256):
    constraints = json.loads((Path(constraints_dir)/'joint_feasibility.json').read_text())
    replay = json.loads(Path(replay_path).read_text())
    info=json.loads((Path(replay['solution'])/'report.json').read_text())
    if (constraints.get('snapshot_sha256') != snapshot_sha256
            or constraints['source_checkpoint_sha256'] != manifest['source_checkpoint_sha256']
            or Path(info['source']).resolve() != Path(constraints_dir).resolve()):
        raise ValueError('Native prefix probe requires the exact prior scene/constraint state')
    from scipy import sparse
    solution = np.load(Path(replay['solution'])/'joint_solution.npz')
    K = sparse.load_npz(Path(constraints_dir)/'joint_kernel.npz')
    optical = K @ solution['tau'] + solution['fixed_optical_depth']
    native = {(r['view'], r['x'], r['y'], r['kind']):r for r in replay['rays'] if r['mode']=='lp_counterfactual'}
    ranked = []
    for ordinal, ray in enumerate(constraints['rays']):
        if ray['kind'] != 'occlusion': continue
        actual = native[(ray['view'],ray['x'],ray['y'],ray['kind'])]
        error = float(np.abs(np.asarray(ray['wall_rgb'])*np.exp(-optical[ordinal])-actual['surface_rgb']).max())
        ranked.append((error,ordinal))
    chosen = [ordinal for _,ordinal in sorted(ranked,reverse=True)[:8]]
    if set(constraints['training_views'])-set(manifest['training_views']):
        raise ValueError('Prefix diagnosis must use training rays only')
    common = dict(background=torch.zeros(3,device='cuda'),include_dynamic=False,
                  optical_replacement_policy='disabled',structural_trainable_start=None)
    records = []
    for ordinal in chosen:
        ray = constraints['rays'][ordinal]; v=views[ray['view']]; x,y=ray['x'],ray['y']
        target=v.original_image[:,y,x].cuda()
        wall=render_hybrid(v,teacher.surface,base,volume_gate=torch.zeros(len(base),device='cuda'),**common)
        wall_rgb=wall.render[:,y,x].clone()
        torch.testing.assert_close(wall_rgb,torch.tensor(ray['wall_rgb'],device='cuda'),atol=2e-5,rtol=2e-5)
        del wall
        samples=[]
        def prefix(depth):
            uv=torch.tensor([[float(x),float(y)]],device='cuda');z=torch.tensor([depth],device='cuda')
            xyz,scale,color=ray_candidates(v,uv,z,torch.zeros(1,3,device='cuda'),[1.,1.,1.],pixel_sigma=16.)
            cloud=CandidateCloud(xyz,scale,color)
            cloud.logits.fill_(math.log((1-1e-6)/1e-6))
            probe=NativeCandidateView(base,cloud)
            result=render_hybrid(v,teacher.surface,probe,
                volume_gate=torch.cat((torch.zeros(len(base),device='cuda'),torch.ones(3,device='cuda'))),**common)
            rgb=result.render[:,y,x].clone();samples.append(dict(depth=float(depth),surface_rgb=rgb.cpu().tolist()))
            return rgb
        median=ray['wall_depth'];lo=.21;hi=max(lo+.1,median*1.25)
        near=prefix(lo);at_median=prefix(median);far=prefix(hi)
        if (near>target+.03+1e-4).any():
            feasible_depth=None;status='even_near_probe_not_feasible'
        elif not (far>target+.03).any():
            feasible_depth=hi;status='upper_search_bound_still_feasible'
        else:
            for _ in range(12):
                mid=(lo+hi)/2
                if (prefix(mid)>target+.03).any():hi=mid
                else:lo=mid
            feasible_depth=lo;status='bounded_prefix_transition'
        ordered=sorted(samples,key=lambda s:s['depth'])
        monotonic_violation=max((float(np.max(np.asarray(a['surface_rgb'])-b['surface_rgb'])) for a,b in zip(ordered,ordered[1:])),default=0.)
        record=dict(ray=ray,median_prefix_rgb=at_median.cpu().tolist(),target=target.cpu().tolist(),
                    feasible_depth_upper_search=feasible_depth,status=status,samples=samples,
                    maximum_nonmonotonic_rgb=monotonic_violation,
                    depth_to_median_ratio=feasible_depth/median if feasible_depth is not None else None)
        records.append(record)
        print(json.dumps({k:v for k,v in record.items() if k!='samples'}),flush=True)
    report=dict(scope='temporary_black_probe__native_surface_prefix__training_only__not_depth_truth_or_model_change',
                input_replay=str(replay_path),records=records,
                limitations=['Freezes the current rigid model; does not establish whether its foreground surface contributions are geometrically correct',
                             'Three finite-alpha virtual splats approximate an opaque screen; tolerance and monotonicity reported',
                             'Search feasibility is radiometric and conditional, not a new leaf depth supervision target'])
    (output/'surface_prefix.json').write_text(json.dumps(report,indent=2))
