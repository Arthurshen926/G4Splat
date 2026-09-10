"""CPU-only calibration/interval check at actual observed multiview tracks."""
import argparse
import io
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from outdoor.moge3_evidence import sha256_file,canonical_json_sha256
from scripts.extract_canopy_render_state import DeferredUnpickler
from scripts.train_unified_outdoor_teacher import _moge3_depth_query_bounds


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('tracks','initialization','runtime-cache','moge-index','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--metric-scale',type=float,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    tracks=json.loads((a.tracks/'audit.json').read_text())
    if tracks['scope']!='original_observed_nvm_canopy_tracks__read_only_not_coverage_prior':
        raise ValueError('Original observed tracks required')
    m=json.loads((a.initialization/'initialization_manifest.json').read_text())
    reader=torch._C.PyTorchFileReader(m['foliage_seed'])
    profiles=DeferredUnpickler(io.BytesIO(reader.get_record('data.pkl'))).load()['audit']['fresh_canonical_leaves']['depth_profiles_by_image']
    cache=json.loads(a.runtime_cache.read_text());unhashed=dict(cache)
    if unhashed.pop('index_hash')!=canonical_json_sha256(unhashed):raise ValueError('Changed runtime metadata')
    if cache['source_index_sha256']!=sha256_file(a.moge_index):raise ValueError('Different MoGe source')
    names=('depth_m','valid_mask','refinement_log_depth_std','refinement_final_delta_log_depth')
    arrays={}
    # Verify every USED array. Unused normal arrays are not evidence for this audit.
    for name in names:
        record=cache['arrays'][name];path=(a.runtime_cache.parent/record['path']).resolve()
        if path.stat().st_size!=record['bytes'] or sha256_file(path)!=record['sha256']:
            raise ValueError('Changed runtime evidence array')
        value=np.load(path,mmap_mode='r',allow_pickle=False)
        if list(value.shape)!=record['shape'] or str(value.dtype)!=record['dtype']:
            raise ValueError('Changed raster format')
        arrays[name]=value
        print(json.dumps(dict(verified_array=name)),flush=True)
    if cache['raster_shape']!=[360,640]:raise ValueError('This audit requires the native640x360 raster')
    camera_index={name:i for i,name in enumerate(cache['camera_order'])}
    by_name={}
    for row in tracks['records']:
        for observation in row['observations']:
            by_name.setdefault(observation['image_name'],[]).append((row['track_id'],observation))
    a.output.mkdir(parents=True,exist_ok=False);rows=[]
    for name,observations in by_name.items():
        if name not in profiles:raise ValueError('Missing seed-bound calibration')
        ci=camera_index[name]
        fields={'moge3_'+key:torch.from_numpy(np.array(arrays[key][ci],copy=True)).float()[None]
                for key in names}
        fields['moge3_depth_m']*=a.metric_scale
        fields['moge3_canopy_depth_m']=fields['moge3_depth_m']*profiles[name]
        prehit,bounds,valid,_,_=_moge3_depth_query_bounds(fields,global_log_scale_sigma=.10099024234861136)
        for track_id,o in observations:
            x,y=np.rint(o['uv']).astype(int)
            if not (0<=x<640 and 0<=y<360):raise ValueError('Out-of-raster original feature')
            if not o['canopy'] or not bool(valid[y,x]):continue
            z=o['depth'];lower=float(bounds[0,y,x]);upper=float(bounds[1,y,x])
            row=dict(track_id=track_id,index=o['index'],image_name=name,uv=o['uv'],track_depth=z,
                     calibrated_moge=float(fields['moge3_canopy_depth_m'][0,y,x]),
                     hit_lower=lower,hit_upper=upper,inside_thin_interval=lower<=z<=upper,
                     before_negative_bound=z<float(prehit[0,y,x]))
            rows.append(row)
    report=dict(scope='observed_track_vs_moge_thin_prior__not_ground_truth_or_corrected_map',records=rows,
        valid_canopy_observations=len(rows),inside_thin_interval=sum(r['inside_thin_interval'] for r in rows),
        before_negative_bound=sum(r['before_negative_bound'] for r in rows),
        median_absolute_log_error=float(np.median([abs(np.log(r['calibrated_moge']/r['track_depth'])) for r in rows])) if rows else None,
        runtime_index_sha256=sha256_file(a.runtime_cache),track_audit_sha256=sha256_file(a.tracks/'audit.json'),
        used_array_sha256={name:cache['arrays'][name]['sha256'] for name in names},
        metric_scale=a.metric_scale,global_log_scale_sigma=.10099024234861136,
        limitations=['Historical sparse tracks are not dense canopy ground truth',
                    'Thin-prior incompatibility alone does not authorize rigid correction'])
    (a.output/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
