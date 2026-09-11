"""Geometry-only audit of saved conditional depth intervals and allowed motion."""
import argparse
import json
from pathlib import Path
import torch


def depth_box_interval(initial,scales,axis,translation,radius):
    center=initial@axis+translation
    extent=radius*(scales*axis.abs()).sum(-1)
    return center-extent,center+extent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);reports=[]
    for run in a.runs:
        state=torch.load(run/'candidates_0800.pth',map_location='cpu',weights_only=False)
        manifest=state['manifest'];s=state['candidate'];radius=manifest['args']['position_radius_sigmas']
        obs=torch.load(run/'moge_position_observations.pth',map_location='cpu',weights_only=False)
        cameras=json.loads((run/'cameras.json').read_text());initial=s['cloud.xyz'];scales=s['cloud.scales']
        updated=initial+radius*scales*s['position_code'].tanh()
        stats=dict(observations=0,individually_unreachable=0,initial_outside=0,final_outside=0,
                   initial_relative_violation_sum=0.,final_relative_violation_sum=0.,minimum_relative_violation_sum=0.)
        for i,(ids,lo,hi) in obs.items():
            c=cameras[i];assert c['id']==i and i not in manifest['excluded_views']
            axis=torch.tensor(c['rotation'])[:,2];translation=-torch.tensor(c['position'])@axis
            z0=initial[ids]@axis+translation;z1=updated[ids]@axis+translation
            zl,zh=depth_box_interval(initial[ids],scales[ids],axis,translation,radius)
            gap0=(lo-z0).clamp_min(0)+(z0-hi).clamp_min(0)
            gap1=(lo-z1).clamp_min(0)+(z1-hi).clamp_min(0)
            minimum=(lo-zh).clamp_min(0)+(zl-hi).clamp_min(0);den=(lo+hi)*.5
            stats['observations']+=len(ids)
            stats['individually_unreachable']+=int((minimum>1e-6).sum())
            stats['initial_outside']+=int((gap0>1e-6).sum());stats['final_outside']+=int((gap1>1e-6).sum())
            for key,gap in [('initial',gap0),('final',gap1),('minimum',minimum)]:
                stats[key+'_relative_violation_sum']+=float((gap/den).double().sum())
        reports.append(dict(run=str(run),**stats))
    result=dict(scope='conditional_interval_reachability__not_depth_accuracy',records=reports,
                limitations=['Per-observation box reachability does not establish joint multi-view feasibility',
                             'Lower MoGe residual is agreement with a prior, not independent geometry accuracy'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
