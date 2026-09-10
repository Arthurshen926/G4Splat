"""New unoptimized seed with measured position covariance; no optical edits."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import torch
from outdoor.canopy_position_uncertainty import CONTRACT,camera_ray_position_covariance
from outdoor.canonical_canopy_initialization import validate_fresh_canonical_seed


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('initialization','cameras','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    if a.output.exists():raise FileExistsError(a.output)
    manifest=json.loads((a.initialization/'initialization_manifest.json').read_text())
    source=Path(manifest['foliage_seed']);seed=torch.load(source,map_location='cpu')
    audit=seed['audit']['fresh_canonical_leaves']
    if audit.get('optimizer_steps')!=0 or audit.get('proposal_pool') is not False:
        raise ValueError('This operation accepts only fresh measured initialization')
    if 'position_uncertainty_contract' in audit:raise ValueError('Uncertainty already versioned')
    source_audit=json.loads((Path(audit['source_diagnostic'])/'audit.json').read_text())
    cameras=json.loads(a.cameras.read_text())
    records={int(r['camera_id']):r for r in source_audit['seed_records']}
    ids=seed['observation_camera_ids'][:,0]
    covariance=torch.empty_like(seed['position_covariance'])
    global_sigma=float(seed['audit']['moge3_exact_k_front_hit']['global_log_scale_sigma'])
    profiles={};max_reprojection=0.
    for camera_id in torch.unique(ids).tolist():
        record=records[camera_id];index=record['index'];c=cameras[index]
        name=c['img_name'];rows=ids==camera_id
        if name not in audit['depth_profiles_by_image']:raise ValueError('Wrong source camera metadata')
        uv=seed['observation_uv'][rows,0]*torch.tensor([c['width'],c['height']])-.5
        z=seed['observation_depth'][rows,0]
        rotation=torch.tensor(c['rotation'],dtype=z.dtype)
        camera_xyz=(seed['centers'][rows]-torch.tensor(c['position']))@rotation
        reprojection=torch.stack((c['fx']*camera_xyz[:,0]/camera_xyz[:,2]+c['cx'],
                                  c['fy']*camera_xyz[:,1]/camera_xyz[:,2]+c['cy']),1)
        error=float((reprojection-uv).norm(dim=1).max());max_reprojection=max(max_reprojection,error)
        if error>.05:raise ValueError('Source camera mapping or exact-K bearing changed')
        profile=source_audit['rigid_anchor_calibrations'][str(index)]
        sigma=max(.025,global_sigma,float(profile.get('robust_log_scatter',0.)))
        covariance[rows]=camera_ray_position_covariance(uv,z,torch.full_like(z,sigma),rotation,
            fx=c['fx'],fy=c['fy'],cx=c['cx'],cy=c['cy'])
        profiles[name]=sigma
    # No optimizer, no renderer, no witness reassignment. Only this metadata
    # tensor and its explicit contract change; historical source is untouched.
    seed['position_covariance']=covariance
    audit['position_uncertainty_contract']=CONTRACT
    audit['position_uncertainty']={'pixel_sigma':2.,'log_depth_sigma_by_image':profiles,
        'depth_sigma_policy':'max_global_scale_sigma_rigid_log_scatter_0.025',
        'source_seed_sha256':digest(source),'source_seed':str(source),
        'source_reprojection_max_pixels':max_reprojection,
        'optical_parameters_and_camera_authority_unchanged':True}
    original=torch.load(source,map_location='cpu')
    for key,value in original.items():
        if torch.is_tensor(value) and key!='position_covariance':
            torch.testing.assert_close(value,seed[key],atol=0,rtol=0,equal_nan=True)
    # Reuse source camera IDs from the audit for seeded cameras; complete
    # camera metadata for validation is loaded from the source COLMAP map.
    from scene.colmap_loader import read_extrinsics_binary
    dataset=Path(manifest['rgb_source']['image_root']).parent
    images=read_extrinsics_binary(dataset/'sparse/0/images.bin')
    fixed=[{'image_id':int(v.camera_id),'image_name':Path(v.name).stem,
            'sequence_id':Path(v.name).stem.split('__')[0]} for v in images.values()]
    validate_fresh_canonical_seed(seed,fixed)
    a.output.mkdir(parents=True,exist_ok=False)
    destination=a.output/'foliage_seed_gaussians.pth';torch.save(seed,destination)
    new=copy.deepcopy(manifest);new['version']='fresh-canonical-position-uncertainty-v153'
    new['foliage_seed']=str(destination.resolve())
    new['foliage']['fresh_canonical_leaves']=audit
    new['initialization_contract']['foliage_reuse']['foliage_seed_sha256']=digest(destination)
    new['position_covariance_repair']={'script_sha256':digest(Path(__file__)),
        'source_manifest':str(a.initialization/'initialization_manifest.json'),
        'all_other_tensors_bitwise_equal':True,'count':len(ids)}
    (a.output/'initialization_manifest.json').write_text(json.dumps(new,indent=2))
    print(json.dumps(new['position_covariance_repair']),flush=True)


if __name__=='__main__':main()
