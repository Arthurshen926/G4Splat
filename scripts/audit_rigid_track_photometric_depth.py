"""Compare measured-track and native-rigid depths with actual image patches.

No depth is fitted here; the two previously computed hypotheses receive the
same camera-tangent patch warp. The report does not authorize surfel removal.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import numpy as np
import torch
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.canopy_patch_stereo import candidate_patch_scores
from outdoor.moge3_evidence import sha256_file
from scripts.evaluate_canopy_validation import FIXED,ADDITIONAL,tensor_digest


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=640,white_background=True)
    for key in ('tracks','depth-audit','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.10)
    data=json.loads(a.tracks.read_text());depth_report=json.loads((a.depth_audit/'audit.json').read_text())
    if depth_report['input_sha256']['original_track_audit']!=sha256_file(a.tracks):
        raise ValueError('Track identity changed')
    by_track={};hashes={}
    for path in sorted(a.depth_audit.glob('observations_*.npz')):
        hashes[path.name]=sha256_file(path);index=int(path.stem.split('_')[-1])
        with np.load(path,allow_pickle=False) as archive:
            selected=archive['valid']&archive['canopy']&(archive['track_depth']-archive['rigid_depth']>archive['margin'])
            for row in np.flatnonzero(selected):
                tid=int(archive['track_ids'][row]);by_track.setdefault(tid,{})[index]=float(archive['rigid_depth'][row])
    selected={tid:obs for tid,obs in by_track.items() if len(obs)>=3}
    if any(set(obs)&set(FIXED+ADDITIONAL) for obs in selected.values()):raise ValueError('Training-only images required')
    a.output.mkdir(parents=True,exist_ok=False)
    dataset=model.extract(a);dataset.model_path=str(a.output)
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=6).getTrainCameras()
    records=[];image_hashes={}
    for tid,native in selected.items():
        track=data['records'][tid];observations={o['index']:o for o in track['observations']}
        pairs=[]
        for source_index,native_depth in native.items():
            source=views[source_index];o=observations[source_index]
            if source.image_name!=o['image_name']:raise ValueError('Source camera changed')
            image_hashes[source.image_name]=tensor_digest(source.original_image)
            uv=torch.tensor([o['uv'],o['uv']],device='cuda')
            depths=torch.tensor([o['depth'],native_depth],device='cuda')
            for target_index in native:
                if target_index==source_index:continue
                target=views[target_index]
                score=candidate_patch_scores(source,target,uv,depths,torch.zeros(1,device='cuda'))[:,0]
                pairs.append(dict(source=source_index,target=target_index,observed_depth=o['depth'],
                    native_depth=native_depth,observed_ncc=float(score[0]),native_ncc=float(score[1])))
        strong=sum(p['observed_ncc']>=.8 and p['observed_ncc']-p['native_ncc']>=.1 for p in pairs)
        records.append(dict(track_id=track['track_id'],pairs=pairs,strong_preference_pairs=strong,
            median_observed_ncc=float(np.median([p['observed_ncc'] for p in pairs])),
            median_native_ncc=float(np.median([p['native_ncc'] for p in pairs]))))
        print(json.dumps({k:v for k,v in records[-1].items() if k!='pairs'}),flush=True)
    report=dict(scope='native_vs_observed_depth_patch_evidence__not_removal_authority',records=records,
        tracks_sha256=sha256_file(a.tracks),depth_report_sha256=sha256_file(a.depth_audit/'audit.json'),
        observation_file_sha256=hashes,image_tensor_sha256=image_hashes,
        limitations=['Sparse feature may be visible through legitimate thin foreground',
                    'Plane-tangent patch model can fail on nonplanar moving foliage',
                    'NCC is comparative image evidence, not calibrated geometric certainty'])
    (a.output/'audit.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
