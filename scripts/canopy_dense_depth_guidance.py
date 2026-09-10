"""Diagnostic depth guidance from reciprocal TRAINING-image hypotheses only."""
import json
from pathlib import Path
import numpy as np
import torch
from outdoor.moge3_evidence import sha256_file
from scripts.evaluate_canopy_validation import tensor_digest


def load_dense_guide(path, views, training, holdout):
    data=json.loads(Path(path).read_text())
    if (len(set(data['sources']))!=3 or set(data['sources'])&set(holdout)
        or not set(data['sources'])<=set(training)):
        raise ValueError('Independent training-only guide views required')
    paths=list(data['input_sha256'])
    for filename,digest in data['input_sha256'].items():
        if sha256_file(Path(filename))!=digest:raise ValueError('Changed reciprocal evidence')
    by_source={}
    for filename in paths:
        if Path(filename).name!='audit.json':continue
        report=json.loads(Path(filename).read_text())
        metadata_path=Path(filename).parent/'cameras.json'
        if sha256_file(metadata_path)!=report['camera_metadata_sha256']:
            raise ValueError('Changed camera metadata')
        metadata=json.loads(metadata_path.read_text())
        for index in data['sources']:
            view=views[index]
            if tensor_digest(view.original_image)!=report['image_tensor_sha256'][view.image_name]:
                raise ValueError('Different guide image')
            camera=metadata[index]
            if (camera['img_name']!=view.image_name or
                not np.allclose(camera['position'],view.camera_center.detach().cpu().numpy(),atol=1e-5,rtol=0) or
                not np.allclose(camera['rotation'],view.world_view_transform[:3,:3].detach().cpu().numpy(),atol=1e-5,rtol=0) or
                not np.allclose([camera[k] for k in ['fx','fy','cx','cy']],
                                [view.focal_x,view.focal_y,view.cx,view.cy],atol=1e-5,rtol=0)):
                raise ValueError('Different guide calibration')
        by_source[report['source']]=Path(filename).parent
    strict=data['summaries'][0]
    if strict['pixel_tolerance']!=2. or strict['log_depth_tolerance']!=.02:
        raise ValueError('Expected strict reciprocal diagnostic contract')
    triplets=np.asarray(strict['indices'],dtype=np.int64)
    if triplets.ndim!=2 or triplets.shape[1]!=3 or not len(triplets):raise ValueError('No reciprocal triplets')
    result=[]
    for column,index in enumerate(data['sources']):
        values=np.load(by_source[index]/'hypotheses_radius_0.4.npz',allow_pickle=False)
        rows=triplets[:,column]
        result.append(dict(index=index,uv=torch.tensor(values['uv'][rows],device='cuda').round().long(),
            depth=torch.tensor(values['depth'][rows],device='cuda')))
    return result


def interval_log_depth_loss(prediction, target, alpha, known):
    valid=known & (alpha.detach()>.01) & torch.isfinite(prediction.detach()) & (prediction.detach()>0)
    valid &= torch.isfinite(target) & (target>0)
    if not valid.any():return torch.nan_to_num(prediction).sum()*0,0
    deficit=((prediction[valid].clamp_min(1e-6)/target[valid]).log().abs()-.02).clamp_min(0)
    return deficit.mean(),int(valid.sum())
