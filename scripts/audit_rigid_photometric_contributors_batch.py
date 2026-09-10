"""Sequential bounded native attribution of mutually corroborated tracks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from outdoor.moge3_evidence import sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('dataset','checkpoint','tracks','photometric','support','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('1','2'):raise ValueError('Only physical GPUs1/2')
    photo=json.loads(a.photometric.read_text())
    if photo['tracks_sha256']!=sha256_file(a.tracks):raise ValueError('Different track evidence')
    selected=[]
    for row in photo['records']:
        cameras=sorted({p['source'] for p in row['pairs']})
        pairs={(p['source'],p['target']) for p in row['pairs']}
        if len(cameras)<3 or len(pairs)!=len(cameras)*(len(cameras)-1):continue
        if all(p['observed_ncc']>=.8 and p['observed_ncc']-p['native_ncc']>=.1 for p in row['pairs']):
            selected.append((row['track_id'],cameras))
    if not selected:raise ValueError('No mutually corroborated candidate')
    a.output.mkdir(parents=True,exist_ok=False)
    candidates={};fingerprints=None;individual=[]
    for tid,cameras in selected:
        output=a.output/f'track_{tid}'
        subprocess.run([sys.executable,str(ROOT/'scripts/audit_rigid_track_contributors.py'),
            '-s',str(a.dataset),'--checkpoint',str(a.checkpoint),'--tracks',str(a.tracks),
            '--track-id',str(tid),'--observation-views',*map(str,cameras),
            '--support',str(a.support),'--output',str(output)],check=True)
        report=json.loads((output/'audit.json').read_text())
        if fingerprints is not None and fingerprints!=report['surface_fingerprints']:
            raise ValueError('Rigid state changed between observations')
        fingerprints=report['surface_fingerprints']
        for row in report['candidates']:
            key=row['row']
            if key not in candidates:candidates[key]={**row,'corroborating_tracks':[]}
            candidates[key]['corroborating_tracks'].append(tid)
            candidates[key]['contradictory_views']=max(candidates[key]['contradictory_views'],row['contradictory_views'])
        individual.append(dict(track_id=tid,path=str(output/'audit.json'),sha256=sha256_file(output/'audit.json')))
        print(json.dumps(dict(completed_track=tid,union_candidates=len(candidates))),flush=True)
    result=dict(scope='local_native_contributor_attribution__not_removal_authority',
        surface_fingerprints=fingerprints,candidates=list(candidates.values()),individual=individual,
        photometric_sha256=sha256_file(a.photometric),tracks_sha256=sha256_file(a.tracks),
        limitations=['Historical point correspondence is not dense leaf identity',
                    'Every ordered pair favors observed depth, but a visible background feature alone does not prove foreground emptiness',
                    'Known building support remains protected; no model was changed'])
    (a.output/'audit.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
