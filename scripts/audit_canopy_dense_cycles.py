"""Reciprocal geometry check of independently searched training depth maps.

Matches remain geometric hypotheses, not automatic leaf material or authority
to delete foreground buildings. No reference or training model is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def project(points, camera):
    local=(points-np.asarray(camera['position']))@np.asarray(camera['rotation'])
    z=local[:,2]
    uv=local[:,:2]/np.maximum(z[:,None],1e-12)
    uv=uv*np.array([camera['fx'],camera['fy']])+np.array([camera['cx'],camera['cy']])
    return uv,z


def pair_matches(source, target, camera, pixel_tolerance=2., log_depth_tolerance=.02):
    if not len(target['uv']):return np.full(len(source['uv']),-1,dtype=np.int64)
    uv,z=project(source['xyz'],camera)
    distance,index=cKDTree(target['uv']).query(uv)
    valid=(z>0)&(distance<=pixel_tolerance)
    valid &= np.abs(np.log(np.maximum(z,1e-12)/target['depth'][index]))<=log_depth_tolerance
    return np.where(valid,index,-1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--maps',type=Path,nargs=3,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    reports=[json.loads((d/'audit.json').read_text()) for d in a.maps]
    sources=[r['source'] for r in reports]
    if len(set(sources))!=3:raise ValueError('Three independently searched sources required')
    for r in reports:
        if set(r['targets'])!=set(sources)-{r['source']}:raise ValueError('Different camera triplets')
        for key in ['runtime_sha256','depth_array_sha256','masks_sha256','camera_metadata_sha256']:
            if r[key]!=reports[0][key]:raise ValueError(f'Mismatched input {key}')
        if r['image_tensor_sha256']!=reports[0]['image_tensor_sha256']:
            raise ValueError('Different image content')
    cameras=json.loads((a.maps[0]/'cameras.json').read_text())
    maps=[dict(np.load(d/'hypotheses_radius_0.4.npz',allow_pickle=False)) for d in a.maps]
    summaries=[]
    for pixel_tolerance,log_depth_tolerance in [(2.,.02),(2.5,.05)]:
        pairs={(i,j):pair_matches(maps[i],maps[j],cameras[sources[j]],pixel_tolerance,log_depth_tolerance)
               for i in range(3) for j in range(3) if i!=j}
        accepted=[]
        for row in range(len(maps[0]['uv'])):
            b=pairs[0,1][row];c=pairs[0,2][row]
            if b<0 or c<0:continue
            if (pairs[1,0][b]==row and pairs[2,0][c]==row and
                pairs[1,2][b]==c and pairs[2,1][c]==b):accepted.append([row,int(b),int(c)])
        summaries.append(dict(pixel_tolerance=pixel_tolerance,log_depth_tolerance=log_depth_tolerance,
            directed_pair_matches={f'{sources[i]}->{sources[j]}':int((v>=0).sum()) for (i,j),v in pairs.items()},
            full_reciprocal_triplets=len(accepted),indices=accepted))
    payload=dict(scope='independent_dense_reciprocal_geometry__not_verified_leaf_material',sources=sources,
        input_counts=[len(m['uv']) for m in maps],summaries=summaries,
        input_sha256={str(d/name):hashlib.sha256((d/name).read_bytes()).hexdigest()
                      for d in a.maps for name in ['audit.json','hypotheses_radius_0.4.npz']},
        limitations=['Sparse grid proximity is not an exact feature identity',
                    'Shared images, MoGe search centers and camera poses are not independent sensors',
                    'Visible background gaps must not be converted into leaf opacity'])
    (a.output/'audit.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload),flush=True)


if __name__=='__main__':main()
