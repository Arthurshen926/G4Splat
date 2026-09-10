"""CPU-only check of source-pixel correspondence drift after leaf refinement."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import torch
from scene.colmap_loader import read_extrinsics_binary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('images','cameras','initial','refined','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    if args.output.exists():raise FileExistsError(args.output)
    images=read_extrinsics_binary(args.images)
    if len({image.camera_id for image in images.values()})!=len(images):
        raise ValueError('This camera container uses ambiguous intrinsic IDs; explicit image identity mapping required')
    cameras={camera['img_name']:camera for camera in json.loads(args.cameras.read_text())}
    by_id={image.camera_id:cameras[Path(image.name).stem] for image in images.values()}
    initial=torch.load(args.initial,map_location='cpu')['foliage']
    refined=torch.load(args.refined,map_location='cpu')['foliage']
    for field in ('observation_camera_ids','observation_uv'):
        torch.testing.assert_close(initial[field],refined[field],rtol=0,atol=0,equal_nan=True)
    ids=initial['observation_camera_ids'][:,0]
    results={}
    for label,payload in [('initial',initial),('refined',refined)]:
        errors=torch.zeros(len(ids))
        for camera in torch.unique(ids).tolist():
            c=by_id[camera];rows=ids==camera
            xyz=(payload['xyz'][rows]-torch.tensor(c['position']))@torch.tensor(c['rotation'])
            uv=torch.stack((c['fx']*xyz[:,0]/xyz[:,2]+c['cx'],c['fy']*xyz[:,1]/xyz[:,2]+c['cy']),dim=1)
            observed=payload['observation_uv'][rows,0]*torch.tensor([c['width'],c['height']])-.5
            errors[rows]=(uv-observed).norm(dim=1)
        selected=errors[refined['verification_state']==1]
        results[label]={'verified_rows':len(selected),
            'pixel_drift_quantiles_50_90_99_100':torch.quantile(selected,torch.tensor([.5,.9,.99,1.])).tolist(),
            'fraction_over_two_pixels':float((selected>2).float().mean())}
    audit={'initial':str(args.initial.resolve()),'refined':str(args.refined.resolve()),
           'image_count':len(images),'unique_camera_ids':len(by_id),
           'scope':'source_bearing_drift__not_reconstruction_quality_or_depth_ground_truth','results':results}
    args.output.write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
