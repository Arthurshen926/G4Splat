"""Independent RGB-descriptor geometry audit; no pose or training mutation.

Reciprocal descriptor matches are hypotheses, not measured motion or depth.
Report rigid controls alongside canopy; never use epipolar filtering alone to
claim arbitrary learned correspondences are correct.
"""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'mast3r')]
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.audit_canopy_epipolar_consistency import fundamental_from_cameras, sampson_pixel_distance
from scripts.evaluate_canopy_validation import FIXED, ADDITIONAL


def calibrated_projection(camera):
    rotation = np.asarray(camera['rotation'], dtype=np.float64).T
    translation = -rotation @ np.asarray(camera['position'], dtype=np.float64)
    intrinsic = np.array([[camera['fx'],0,camera['cx']],
                          [0,camera['fy'],camera['cy']],[0,0,1.]])
    return intrinsic @ np.c_[rotation, translation]


def triangulate_matches(left, right, source, target):
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 2:
        raise ValueError('Aligned Nx2 native pixel coordinates required')
    if not len(left):
        return np.empty((0,3)),np.zeros(0,dtype=bool),np.empty(0),np.empty(0)
    a, b = calibrated_projection(source), calibrated_projection(target)
    homogeneous = cv2.triangulatePoints(a, b, left.T.astype(np.float64), right.T.astype(np.float64)).T
    valid = np.abs(homogeneous[:,3]) > 1.e-12
    points = np.full((len(left),3), np.nan)
    points[valid] = homogeneous[valid,:3]/homogeneous[valid,3:]
    errors = []
    for projection, observation in ((a,left),(b,right)):
        camera = points @ projection[:,:3].T + projection[:,3]
        valid &= np.isfinite(camera).all(1) & (camera[:,2] > 0)
        errors.append(np.linalg.norm(camera[:,:2]/np.maximum(camera[:,2:],1.e-12)-observation,axis=1))
    rays = [points-np.asarray(c['position']) for c in (source,target)]
    rays = [ray/np.maximum(np.linalg.norm(ray,axis=1,keepdims=True),1.e-12) for ray in rays]
    angle = np.degrees(np.arccos(np.clip((rays[0]*rays[1]).sum(1),-1,1)))
    error = np.maximum(*errors)
    return points, valid & (error < 1.) & (angle >= .5), error, angle


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('cameras','dataset','masks','cohort','weights','output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--closing-pairs-only', action='store_true',
                   help='Additional edges to close the original training-only triangles')
    p.add_argument('--pairs',type=int,nargs='+',help='Explicit non-validation endpoint pairs for follow-up diagnostics')
    p.add_argument('--gpu-memory-fraction',type=float,default=.40)
    a = p.parse_args()
    cameras = json.loads(a.cameras.read_text())
    cohort = json.loads(a.cohort.read_text())
    allowed = set(cohort['calibrated_training_views'])
    if allowed & set(FIXED+ADDITIONAL):
        raise ValueError('Validation cameras cannot enter the matching audit')
    # Fixed difficult training neighborhoods plus spatially separated controls.
    pairs = [(658,659),(659,661),(661,662),(686,687),(687,688),
             (725,726),(726,727),(582,583),(583,584)]
    if a.closing_pairs_only:
        pairs = [(658,661),(659,662),(686,688),(725,727),(582,584)]
    if a.pairs is not None:
        if a.closing_pairs_only or len(a.pairs)%2 or not a.pairs:
            raise ValueError('Explicit pairs require an even endpoint list and no preset override')
        pairs=list(zip(a.pairs[::2],a.pairs[1::2]))
        if len(set(pairs))!=len(pairs) or any(i==j for i,j in pairs):
            raise ValueError('Distinct non-self descriptor pairs required')
    if not 0<a.gpu_memory_fraction<=1:
        raise ValueError('A bounded CUDA memory fraction is required')
    if any(i not in allowed or j not in allowed for i,j in pairs):
        raise ValueError('All diagnostic pair endpoints must be training cameras')
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); cv2.setNumThreads(4)
    torch.cuda.set_per_process_memory_fraction(a.gpu_memory_fraction)
    model = AsymmetricMASt3R.from_pretrained(str(a.weights)).cuda().eval()
    masks = CambridgeMaskLookup(a.dataset, a.masks)
    cache = {}
    def image(index):
        if index not in cache:
            c = cameras[index]; w,h = c['width'],c['height']
            path = a.dataset/'images'/(c['img_name']+'.png')
            rgb = Image.open(path).convert('RGB').resize((w,h), Image.Resampling.BILINEAR)
            tensor = torch.from_numpy(np.asarray(rgb).copy()).permute(2,0,1).float()/127.5-1
            # Pad only bottom/right to patch multiples. Pixel coordinates and K
            # are unchanged; padded pixels are excluded from every measurement.
            tensor = F.pad(tensor[None], (0,(-w)%16,0,(-h)%16), mode='replicate')
            cache[index] = {'img': tensor, 'true_shape': np.int32([tensor.shape[-2:]]),
                            'idx': index, 'instance': c['img_name']}
        return dict(cache[index])
    def regions(index):
        c = cameras[index]
        obj,sky,dist,tree = masks.get_index_masks(c['img_name'], (0,1,2,3),
            (c['height'],c['width']), torch.device('cpu'))
        known = obj & sky & dist
        return [(known & ~tree).numpy(), (known & tree).numpy()]
    records = []
    for i,j in pairs:
        output = inference([(image(i),image(j))],model,'cuda',batch_size=1,verbose=False)
        pred0,pred1 = output['pred1'],output['pred2']
        left,right = fast_reciprocal_NNs(pred0['desc'][0],pred1['desc'][0],
            subsample_or_initxy1=8,device='cuda',dist='dot',block_size=2048)
        keep = np.ones(len(left),dtype=bool)
        for xy,index in ((left,i),(right,j)):
            c=cameras[index]
            keep &= (xy[:,0]>=3)&(xy[:,1]>=3)&(xy[:,0]<c['width']-3)&(xy[:,1]<c['height']-3)
        left,right=left[keep],right[keep]
        if not len(left): raise RuntimeError('No reciprocal in-frame matches')
        error=sampson_pixel_distance(left,right,fundamental_from_cameras(cameras[i],cameras[j]))
        points,geometric,reprojection,angle=triangulate_matches(left,right,cameras[i],cameras[j])
        row={'source':i,'target':j,'reciprocal_matches':len(left),'regions':{}}
        labels=np.zeros(len(left),dtype=np.uint8)
        for label,source_mask,target_mask,code in zip(('canopy','rigid'),regions(i),regions(j),(1,2)):
            selected=source_mask[left[:,1],left[:,0]]&target_mask[right[:,1],right[:,0]]
            labels[selected]=code
            values=error[selected]
            row['regions'][label]={'matches':len(values),
                'median_epipolar_px':float(np.median(values)) if len(values) else None,
                'p90_epipolar_px':float(np.quantile(values,.9)) if len(values) else None,
                'triangulation_candidates':int((selected&geometric).sum())}
        np.savez_compressed(a.output/f'matches_{i}_{j}.npz',left=left,right=right,
            points=points,geometric=geometric,reprojection=reprojection,angle=angle,labels=labels)
        records.append(row); print(json.dumps(row),flush=True)
        del output,pred0,pred1
    (a.output/'audit.json').write_text(json.dumps({'scope':'descriptor_hypotheses_with_fixed_camera_rigid_controls__not_ground_truth',
        'pixel_contract':'640_native_camera_grid_bottom_right_pad_only_no_K_change',
        'pairs':pairs,'excluded':FIXED+ADDITIONAL,'records':records},indent=2))


if __name__ == '__main__': main()
