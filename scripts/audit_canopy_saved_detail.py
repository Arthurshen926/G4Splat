"""CPU regional detail checks on saved GT|native-render PNG pairs.

These are explicitly PNG-derived metrics, not replacements for float native PSNR.
Gradient strength alone is not quality: report gradient error alongside it.
"""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from utils.loss_utils import create_window
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.evaluate_canopy_validation import FIXED,ADDITIONAL


def detail_metrics(prediction,target,mask):
    if prediction.shape!=target.shape or prediction.shape!=(3,*mask.shape):
        raise ValueError('Aligned RGB and mask required')
    errors=[];strength=[];reference=[]
    for axis in (1,2):
        grad=prediction.diff(dim=axis);truth=target.diff(dim=axis)
        valid=mask[:,1:]&mask[:,:-1] if axis==2 else mask[1:]&mask[:-1]
        errors.append((grad[:,valid]-truth[:,valid]).abs().sum())
        strength.append(grad[:,valid].abs().sum());reference.append(truth[:,valid].abs().sum())
    count=3*((mask[:,1:]&mask[:,:-1]).sum()+(mask[1:]&mask[:-1]).sum())
    kernel=create_window(11,3).to(prediction)
    x=prediction[None];y=target[None]
    blur=lambda value:F.conv2d(value,kernel,padding=5,groups=3)
    mx=blur(x);my=blur(y)
    vx=blur(x*x)-mx*mx;vy=blur(y*y)-my*my;cov=blur(x*y)-mx*my
    similarity=((2*mx*my+.01**2)*(2*cov+.03**2)/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))).mean(dim=1)[0]
    core=F.avg_pool2d(mask.float()[None,None],11,1,5,count_include_pad=True)[0,0]>=1-1.e-6
    return {'gradient_mae':float(sum(errors)/count) if count else None,
            'gradient_strength_ratio':float(sum(strength)/sum(reference)) if sum(reference)>0 else None,
            'interior_ssim':float(similarity[core].mean()) if core.any() else None,
            'ssim_interior_pixels':int(core.sum())}


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('renders','cameras','dataset','masks','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--step',type=int)
    args=parser.parse_args();torch.set_num_threads(4)
    if args.output.exists():raise FileExistsError(args.output)
    cameras=json.loads(args.cameras.read_text())
    masks=CambridgeMaskLookup(args.dataset,args.masks)
    rows=[]
    for index in FIXED+ADDITIONAL:
        suffix='' if args.step is None else f'_{args.step:04d}'
        path=args.renders/f'view_{index}{suffix}.png'
        rgb=torch.from_numpy(np.array(Image.open(path).convert('RGB'),copy=True)).permute(2,0,1).float()/255
        if rgb.shape[2]%2:raise ValueError('Expected side-by-side GT and prediction')
        target,prediction=rgb.chunk(2,dim=2)
        obj,sky,distortion,tree=masks.get_index_masks(cameras[index]['img_name'],(0,1,2,3),target.shape[1:],torch.device('cpu'))
        known=obj&sky&distortion
        rows.append({'index':index,'tree':detail_metrics(prediction,target,known&~tree),
                     'rigid':detail_metrics(prediction,target,known&tree)})
    summary={}
    for region in ('tree','rigid'):
        summary[region]={}
        for metric in ('gradient_mae','gradient_strength_ratio','interior_ssim'):
            values=[row[region][metric] for row in rows if row[region][metric] is not None]
            summary[region][metric]=sum(values)/len(values) if values else None
    result={'scope':'canonical48__8bit_png_derived_detail_metrics','renders':str(args.renders.resolve()),
            'step':args.step,'summary':summary,'per_view':rows}
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
