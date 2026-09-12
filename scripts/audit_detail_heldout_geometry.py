"""Leave-one-observation-out geometry checks; not blind correspondence validation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from outdoor.native_observation_matching import triangulate_rays

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--association',type=Path,required=True)
p.add_argument('--cameras',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
records=json.loads((a.association/'audit.json').read_text())['observations']
cams=json.loads(a.cameras.read_text())
with np.load(a.association/'rgb_triangulated_proposals.npz') as z:prior=z['prior_xyz'].copy()
# Packet matching used native1920x1080 images; verify camera aspect contract.
def intrinsics(c):
    if abs(c['width']/c['height']-1920/1080)>1e-6:raise ValueError('Unexpected aspect')
    return c['fx']*1920/c['width'],c['fy']*1080/c['height'],c['cx']*1920/c['width']-.5,c['cy']*1080/c['height']-.5
def ray(o):
    c=cams[o['view']];fx,fy,cx,cy=intrinsics(c);u,v=o['xy']
    return np.array([(u-cx)/fx,(v-cy)/fy,1.])@np.asarray(c['rotation']).T
def error(point,o):
    c=cams[o['view']];q=(point-np.asarray(c['position']))@np.asarray(c['rotation'])
    if q[2]<=0:return None
    fx,fy,cx,cy=intrinsics(c)
    return float(np.linalg.norm(np.array([q[0]/q[2]*fx+cx,q[1]/q[2]*fy+cy])-o['xy']))
rows=[]
for index,r in enumerate(records):
    obs=r['observations']
    if len(obs)<4:continue
    for held in range(len(obs)):
        fit=[o for k,o in enumerate(obs) if k!=held]
        point=triangulate_rays([cams[o['view']]['position'] for o in fit],[ray(o) for o in fit])
        rows.append(dict(row=r['row'],held_view=obs[held]['view'],
            prior_error=error(prior[index],obs[held]),heldout_error=error(point,obs[held]) if point is not None else None))
report=dict(scope='leave_one_ray_out__correspondence_selection_still_shared__not_blind_validation',
    total_tracks=len(records),testable_tracks=sum(len(r['observations'])>=4 for r in records),
    rows=rows,passed=sum(r['heldout_error'] is not None and r['heldout_error']<=1.5 for r in rows),
    limitations=['Three-observation tracks cannot be tested by a three-ray fit with held-out evidence',
                 'Observation selection used all matches, so this is not independent matching confirmation',
                 'No model insertion or reconstruction claim'])
with a.output.open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report))
