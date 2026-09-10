"""Read-only original observed tracks: never synthetic projected coverage.

Only tracks observed entirely in allowed canonical training cameras qualify.
Raw NVM image observations must reproject under the current exact pinhole
cameras; unsupported distortion conventions are rejected, not silently fitted.
These historical tracks are not held-out geometry or independent camera poses.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from matcha.cambridge_masks import CambridgeMaskLookup
from outdoor.moge3_evidence import sha256_file


def triangulate_training_observations(cameras,pixels):
    """Refit from allowed measured pixels only; no original NVM xyz input."""
    if len(cameras)<3 or len(cameras)!=len(pixels):raise ValueError('Three observed cameras required')
    rotations=np.array([c['rotation'] for c in cameras],dtype=float)
    centers=np.array([c['position'] for c in cameras],dtype=float)
    fx=np.array([c['fx'] for c in cameras]);fy=np.array([c['fy'] for c in cameras])
    cx=np.array([c['cx'] for c in cameras]);cy=np.array([c['cy'] for c in cameras])
    uv=np.asarray(pixels,dtype=float)
    rays=np.stack(((uv[:,0]-cx)/fx,(uv[:,1]-cy)/fy,np.ones(len(cameras))),1)
    rays=np.einsum('nij,nj->ni',rotations,rays)
    rays/=np.linalg.norm(rays,axis=1,keepdims=True)
    projector=np.eye(3)[None]-rays[:,:,None]*rays[:,None,:]
    info=projector.sum(0)
    if np.linalg.eigvalsh(info)[0]<1.e-8:return None
    xyz=np.linalg.solve(info,np.einsum('nij,nj->i',projector,centers))
    for _ in range(8):
        point=np.einsum('ni,nij->nj',xyz-centers,rotations);x,y,z=point.T
        if not np.isfinite(point).all() or (z<=0).any():return None
        residual=np.stack((x/z*fx+cx,y/z*fy+cy),1)-uv
        jac=np.zeros((len(cameras),2,3));jac[:,0,0]=fx/z;jac[:,1,1]=fy/z
        jac[:,0,2]=-fx*x/z**2;jac[:,1,2]=-fy*y/z**2
        jac=jac@np.swapaxes(rotations,1,2);j=jac.reshape(-1,3)
        info=j.T@j
        if np.linalg.eigvalsh(info)[0]<1.e-8:return None
        delta=np.linalg.solve(info,j.T@residual.reshape(-1))
        xyz-=delta
        if np.linalg.norm(delta)<1.e-8:break
    return xyz if np.isfinite(xyz).all() else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('nvm','camera-audit','dataset','masks','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--retriangulate-training-subset',action='store_true',
        help='Discard non-training observations BEFORE fresh triangulation; never reuse original xyz')
    a=p.parse_args();torch.set_num_threads(4)
    audit=json.loads((a.camera_audit/'audit.json').read_text())
    excluded=set(audit['excluded_views'])
    cameras=json.loads((a.camera_audit/'cameras.json').read_text())
    lookup={c['img_name']:(i,c) for i,c in enumerate(cameras)
            if c['img_name'].startswith('seq2__') and i not in excluded}
    masks=CambridgeMaskLookup(a.dataset,a.masks)
    a.output.mkdir(parents=True,exist_ok=False)
    cached={};records=[];counts=dict(input_points=0,all_allowed_three_view=0,
                                   pinhole_reprojection_valid=0,angular_valid=0,canopy_tracks=0)
    def next_nonempty(stream):
        while True:
            line=stream.readline()
            if not line:raise ValueError('Truncated NVM')
            if line.strip():return line.strip()
    with a.nvm.open() as stream:
        if next_nonempty(stream)!='NVM_V3':raise ValueError('Expected original NVM_V3 observations')
        count=int(next_nonempty(stream));allowed={}
        for ci in range(count):
            values=next_nonempty(stream).split()
            name=str(Path(values[0]).with_suffix('')).replace('/','__')
            if name not in lookup:continue
            index,camera=lookup[name]
            center=np.array(values[6:9],dtype=float)
            if not np.allclose(center,camera['position'],atol=1.e-4,rtol=0):
                raise ValueError('Original track and runtime camera gauges differ')
            allowed[ci]=(index,camera,float(values[1]))
        count=int(next_nonempty(stream))
        for track_id in range(count):
            values=next_nonempty(stream).split();counts['input_points']+=1
            n=int(values[6])
            if len(values)!=7+4*n:raise ValueError('Malformed NVM observations')
            measurements=[(j,int(values[7+4*j])) for j in range(n)]
            if a.retriangulate_training_subset:
                measurements=[(j,ci) for j,ci in measurements if ci in allowed]
            ids=[ci for _,ci in measurements]
            if len(set(ids))<3 or len(set(ids))!=len(ids) or any(ci not in allowed for ci in ids):continue
            counts['all_allowed_three_view']+=1
            if a.retriangulate_training_subset:
                fit_cameras=[];fit_pixels=[]
                for j,ci in measurements:
                    _,camera,focal=allowed[ci]
                    fit_cameras.append(camera)
                    fit_pixels.append([float(values[9+4*j])*camera['fx']/focal+camera['cx'],
                                       float(values[10+4*j])*camera['fy']/focal+camera['cy']])
                xyz=triangulate_training_observations(fit_cameras,fit_pixels)
                if xyz is None:continue
            else:xyz=np.array(values[:3],dtype=float)
            observations=[];rays=[];info=np.zeros((3,3))
            good=True
            for j,ci in measurements:
                index,camera,focal=allowed[ci]
                rotation=np.array(camera['rotation']);center=np.array(camera['position'])
                point=(xyz-center)@rotation;x,y,z=point
                if not np.isfinite(point).all() or z<=0:good=False;break
                uv=np.array([float(values[9+4*j])*camera['fx']/focal+camera['cx'],
                             float(values[10+4*j])*camera['fy']/focal+camera['cy']])
                projected=np.array([x/z*camera['fx']+camera['cx'],y/z*camera['fy']+camera['cy']])
                error=float(np.linalg.norm(projected-uv))
                px,py=np.rint(uv).astype(int)
                if error>=1 or not (0<=px<camera['width'] and 0<=py<camera['height']):good=False;break
                if index not in cached:
                    obj,sky,dist,tree=masks.get_index_masks(camera['img_name'],(0,1,2,3),
                        (camera['height'],camera['width']),torch.device('cpu'))
                    cached[index]=((obj&sky&dist).numpy(),(~tree).numpy())
                known,tree=cached[index]
                if not known[py,px]:good=False;break
                ray=xyz-center;rays.append(ray/np.linalg.norm(ray))
                jac=np.array([[camera['fx']/z,0,-camera['fx']*x/z**2],
                              [0,camera['fy']/z,-camera['fy']*y/z**2]])@rotation.T
                info+=jac.T@jac
                observations.append(dict(index=index,image_name=camera['img_name'],uv=uv.tolist(),
                                         depth=float(z),canopy=bool(tree[py,px]),reprojection=error))
            if not good:continue
            counts['pinhole_reprojection_valid']+=1
            rays=np.array(rays);angles=np.degrees(np.arccos(np.clip(rays@rays.T,-1,1)))
            angle=float(np.quantile(angles[np.triu_indices(len(measurements),1)],.1))
            if angle<1 or np.linalg.eigvalsh(info)[0]<=1.e-8:continue
            counts['angular_valid']+=1
            if sum(o['canopy'] for o in observations)<2:continue
            counts['canopy_tracks']+=1
            records.append(dict(track_id=track_id,xyz=xyz.tolist(),observations=observations,
                                angle_p10=angle,covariance=np.linalg.inv(info).tolist()))
    result=dict(scope='original_observed_nvm_canopy_tracks__read_only_not_coverage_prior',
        counts=counts,records=records,excluded_views=sorted(excluded),
        position_source='training_only_retriangulation_no_original_xyz' if a.retriangulate_training_subset else 'original_nvm_xyz_all_observations_training',
        source_nvm_sha256=sha256_file(a.nvm),masks_sha256=sha256_file(a.masks),
        cameras_sha256=sha256_file(a.camera_audit/'cameras.json'),
        limitations=['Historical SfM tracks may have initialized rigid geometry',
                    'Historical track membership and camera poses are inherited; training-only refit is not a new blind matching benchmark',
                    'No assumption of independent three-edge descriptor cycles',
                    'Strict native reprojection rejects unsupported raw distortion rather than estimating it'])
    (a.output/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(**counts,views=sorted({o['index'] for r in records for o in r['observations']}))),flush=True)


if __name__=='__main__':main()
