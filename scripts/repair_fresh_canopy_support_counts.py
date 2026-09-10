"""Repair only stale cached counts of existing real camera support IDs."""
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import torch
from scripts.repair_fresh_canopy_position_covariance import digest
from outdoor.canonical_canopy_initialization import validate_fresh_canonical_seed
from scene.colmap_loader import read_extrinsics_binary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initialization',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    if a.output.exists():raise FileExistsError(a.output)
    manifest=json.loads((a.initialization/'initialization_manifest.json').read_text())
    source=Path(manifest['foliage_seed']);payload=torch.load(source,map_location='cpu')
    audit=payload['audit']['fresh_canonical_leaves']
    if audit.get('optimizer_steps')!=0 or audit.get('proposal_pool') is not False:raise ValueError('Fresh measured seed required')
    ids=payload['support_camera_ids'].sort(dim=1).values
    distinct=ids>=0;distinct[:,1:] &= ids[:,1:]!=ids[:,:-1]
    count=distinct.sum(1).to(payload['support_view_count'].dtype)
    if (count<payload['support_view_count']).any():raise ValueError('Cannot silently remove historical support')
    changed=int((count!=payload['support_view_count']).sum())
    payload['support_view_count']=count
    dataset=Path(manifest['rgb_source']['image_root']).parent
    images=read_extrinsics_binary(dataset/'sparse/0/images.bin')
    fixed=[{'image_id':int(v.camera_id),'image_name':Path(v.name).stem,
            'sequence_id':Path(v.name).stem.split('__')[0]} for v in images.values()]
    validate_fresh_canonical_seed(payload,fixed)
    original=torch.load(source,map_location='cpu')
    for key,value in original.items():
        if torch.is_tensor(value) and key!='support_view_count':
            torch.testing.assert_close(value,payload[key],atol=0,rtol=0,equal_nan=True)
    record={'contract':'cached_support_count_equals_existing_distinct_camera_ids_v1',
        'source_seed':str(source),'source_sha256':digest(source),'changed_rows':changed,
        'all_other_tensors_unchanged':True,'new_camera_ids_added':0,'new_witnesses_claimed':0}
    audit['support_count_repair']=record
    a.output.mkdir(parents=True,exist_ok=False)
    destination=a.output/'foliage_seed_gaussians.pth';torch.save(payload,destination)
    new=copy.deepcopy(manifest);new['version']='fresh-canonical-consistent-support-counts-v155'
    new['foliage_seed']=str(destination.resolve());new['foliage']['fresh_canonical_leaves']=audit
    new['initialization_contract']['foliage_reuse']['foliage_seed_sha256']=digest(destination)
    new['support_count_repair']={**record,'script_sha256':digest(Path(__file__))}
    (a.output/'initialization_manifest.json').write_text(json.dumps(new,indent=2))
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
