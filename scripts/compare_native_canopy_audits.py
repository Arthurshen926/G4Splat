"""Compare matched native RGB audits without mixing pixel resolutions."""
import argparse
import json
from pathlib import Path
from statistics import mean


def compare(left,right):
    for key in ('scope','leaf_optical_kernel','experimental_binary_sha256',
                'audit_script_sha256','rgb_region_helper_sha256'):
        if left[key]!=right[key]:raise ValueError('Unmatched audit: '+key)
    first={r['index']:r for r in left['records']}
    second={r['index']:r for r in right['records']}
    if len(first)!=len(left['records']) or len(second)!=len(right['records']):
        raise ValueError('Duplicate audit camera')
    if first.keys()!=second.keys() or not first:raise ValueError('Unmatched cameras')
    rows=[]
    for index,a in first.items():
        b=second[index]
        if a['image_shape']!=b['image_shape']:raise ValueError('Unmatched resolution')
        if a['rgb_regions']['source']!=b['rgb_regions']['source']:
            raise ValueError('Unmatched source rendering')
        delta={k:b['rgb_regions']['updated'][k]-v for k,v in a['rgb_regions']['updated'].items()}
        rows.append(dict(index=index,image_shape=a['image_shape'],delta=delta,
            canopy_surface_alpha_delta=b['canopy_surface_alpha_after']-a['canopy_surface_alpha_after']))
    return dict(scope='matched_resolution_descriptive_regression__not_promotion',
        checkpoints=[left['checkpoint_sha256'],right['checkpoint_sha256']],
        mean_delta={k:mean(r['delta'][k] for r in rows) for k in rows[0]['delta']},
        per_view=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('control','experiment','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();report=compare(json.loads(a.control.read_text()),json.loads(a.experiment.read_text()))
    with a.output.open('x') as out:json.dump(report,out,indent=2)
    print(json.dumps(report['mean_delta']))
