"""Bounded necessary constraints from exact native prefix artifacts, not targets."""
from pathlib import Path
import numpy as np
import torch
from outdoor.moge3_evidence import sha256_file
from scripts.canopy_radiance_prefix_bounds import optical_prefix_lower_bound


def select_prefix_samples(depths, cumulative_rgb, ray, maximum=6):
    depths=np.asarray(depths,dtype=np.float64)
    rgb=np.asarray(cumulative_rgb,dtype=np.float64)
    if (maximum<2 or depths.ndim!=1 or rgb.shape!=(len(depths),3)
            or not np.isfinite(depths).all() or not np.isfinite(rgb).all()
            or (depths<=0).any() or (np.diff(depths)<0).any()
            or (rgb<0).any() or (np.diff(rgb,axis=0)<-1e-6).any()):
        raise ValueError('Ordered finite nonnegative native prefixes required')
    # At tied depths all surfaces precede volumes; use the complete tied group.
    ends=np.flatnonzero(np.r_[np.diff(depths)!=0,True]) if len(depths) else np.array([],dtype=int)
    valid=[i for i in ends if optical_prefix_lower_bound(rgb[i],ray['target'],ray['wall_rgb'],leakage_fraction=0.)>.05]
    if not valid:return []
    chosen=np.unique(np.linspace(0,len(valid)-1,min(maximum,len(valid))).round().astype(int))
    return [dict(depth=float(depths[valid[i]]),surface_rgb=rgb[valid[i]].tolist()) for i in chosen]


def load_batch_prefix_records(path, report, old, manifest):
    artifact=Path(path).with_name('surface_prefixes.pth')
    if (report['source_checkpoint_sha256']!=manifest['source_checkpoint_sha256']
            or set(report['training_views'])!=set(old['training_views'])
            or set(report['excluded_views'])!=set(manifest['excluded_views'])
            or report['maximum_native_rgb_error']>3e-5
            or sha256_file(artifact)!=report['prefixes_sha256']):
        raise ValueError('Exact source, view and native prefix provenance required')
    data=torch.load(artifact,map_location='cpu',weights_only=False)
    if data['source_checkpoint_sha256']!=manifest['source_checkpoint_sha256']:
        raise ValueError('Prefix payload source mismatch')
    expected={(r['view'],r['x'],r['y']):r for r in old['rays']}
    seen=set();records=[]
    for entry in data['prefixes']:
        ray=entry['ray'];key=(ray['view'],ray['x'],ray['y'])
        if key in seen or expected.get(key)!=ray:
            raise ValueError('Duplicate or changed diagnostic ray')
        seen.add(key)
        if ray['kind']!='occlusion':continue
        samples=select_prefix_samples(entry['depths'],entry['cumulative_rgb'],ray)
        records.append(dict(ray=ray,target=ray['target'],samples=samples,maximum_nonmonotonic_rgb=0.))
    if seen!=set(expected):raise ValueError('Incomplete full-training prefix artifact')
    return records
