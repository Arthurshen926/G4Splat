"""Read-only source-bearing association to original observed tree tracks.

Near source pixels are regional associations, not exact feature identities.
This audit does not move leaves, grant verification or export constraints for
production optimization.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'mast3r')]
import numpy as np
import torch
from scipy.spatial import cKDTree
from colmap.read_write_model import read_images_binary
from outdoor.hybrid_gaussian_renderer import persistent_static_evidence_mask
from outdoor.moge3_evidence import sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','tracks','thin-audit','dataset','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--pixel-radius',type=float,default=2.)
    a=p.parse_args();torch.set_num_threads(4)
    if not 0<a.pixel_radius<=3:raise ValueError('A local source-bearing radius is required')
    tracks=json.loads((a.tracks/'audit.json').read_text())
    thin=json.loads((a.thin_audit/'audit.json').read_text())
    if thin['track_audit_sha256']!=sha256_file(a.tracks/'audit.json'):
        raise ValueError('Track identity differs from the interval audit')
    support={}
    for o in thin['records']:
        if o['track_depth']<o['hit_lower']:support.setdefault(o['track_id'],set()).add(o['index'])
    selected=[r for r in tracks['records'] if len(support.get(r['track_id'],()))>=3]
    cameras=read_images_binary(str(a.dataset/'sparse/0/images.bin'))
    # Runtime CameraInfo.uid is the calibrated camera ID, not sorted view index.
    ids={Path(c.name).stem:int(c.camera_id) for c in cameras.values()}
    state=torch.load(a.checkpoint,map_location='cpu');f=state['foliage']
    obj=SimpleNamespace(**f,dynamic_leaf_mask=f['layer_role']==2)
    persistent=(persistent_static_evidence_mask(obj)&f['static_detail']&(f['layer_role']==0)).numpy()
    xyz=f['xyz'].numpy();uv=f['observation_uv'][:,0].numpy()*np.array([640,360])-.5
    source=f['observation_camera_ids'][:,0].numpy()
    valid=persistent&np.isfinite(uv).all(1)
    trees={};rows=[]
    for track in selected:
        associated=set();by_view=[]
        for observation in track['observations']:
            if not observation['canopy']:continue
            camera_id=ids[observation['image_name']]
            if camera_id not in trees:
                indices=np.flatnonzero(valid&(source==camera_id))
                trees[camera_id]=(indices,cKDTree(uv[indices]) if len(indices) else None)
            indices,tree=trees[camera_id]
            near=[] if tree is None else tree.query_ball_point(observation['uv'],a.pixel_radius)
            found=indices[near].tolist();associated.update(found)
            by_view.append(dict(index=observation['index'],associated_source_rows=len(found)))
        leaf_rows=sorted(associated)
        displacement=(np.array(track['xyz'])-xyz[leaf_rows]) if leaf_rows else np.empty((0,3))
        rows.append(dict(track_id=track['track_id'],target_xyz=track['xyz'],associated_leaf_rows=leaf_rows,
            per_view=by_view,median_required_displacement=float(np.median(np.linalg.norm(displacement,axis=1))) if leaf_rows else None))
    a.output.mkdir(parents=True,exist_ok=False)
    result=dict(scope='regional_source_bearing_association__not_exact_features_or_training_authority',records=rows,
        selected_three_view_front_tracks=len(selected),associated_tracks=sum(bool(r['associated_leaf_rows']) for r in rows),
        distinct_associated_rows=len({i for r in rows for i in r['associated_leaf_rows']}),pixel_radius=a.pixel_radius,
        checkpoint_sha256=sha256_file(a.checkpoint),track_audit_sha256=sha256_file(a.tracks/'audit.json'),
        thin_audit_sha256=sha256_file(a.thin_audit/'audit.json'),
        camera_images_sha256=sha256_file(a.dataset/'sparse/0/images.bin'),
        limitations=['Nearby source pixels can lie on different leaves or background gaps',
                    'Old persistent verification is not an independent descriptor identity'])
    (a.output/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
