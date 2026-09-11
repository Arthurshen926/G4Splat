"""CPU census of semantic mixing in the existing nine-sample depth patch."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from matcha.cambridge_masks import CambridgeMaskLookup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads((args.run / 'manifest.json').read_text())
    cameras = json.loads((args.run / 'cameras.json').read_text())
    mask_lookup = CambridgeMaskLookup(Path(manifest['args']['source_path']), manifest['args']['masks'])
    indices = [row['index'] for row in manifest['seed_audit'] if row['rays']]
    if not set(indices) <= set(manifest['training_views']) or set(indices) & set(manifest['excluded_views']):
        raise ValueError('Only seed TRAINING cameras permitted')
    kernel = torch.zeros(1, 1, 5, 5)
    kernel[:, :, ::2, ::2] = 1
    records = []
    total = torch.zeros(10, dtype=torch.int64)
    for index in indices:
        camera = cameras[index]
        obj, ns, dist, nt = mask_lookup.get_index_masks(camera['img_name'], (0,1,2,3),
            (camera['height'],camera['width']), torch.device('cpu'))
        tree = obj & ns & dist & ~nt
        counts = F.conv2d(tree.float()[None,None],kernel,padding=2)[0,0].round().long()
        valid = torch.zeros_like(tree); valid[2:-2,2:-2] = True
        histogram = torch.bincount(counts[tree & valid], minlength=10)
        total += histogram
        records.append(dict(index=index, tree_patch_count_histogram=histogram.tolist()))
    report = dict(scope='training_seed_camera_semantic_patch_census__not_selected_ray_or_match_error_rate',
        offsets=[-2,0,2], seed_camera_count=len(indices), histogram=total.tolist(),
        mixed_fraction=float(total[:9].sum()/total.sum().clamp_min(1)),
        limitations=['All valid tree centers, not the RGB-priority selected seed rays',
                     'Semantic masks may themselves include holes or incorrect labels',
                     'Mixing is not proof of incorrect depth; neighbor projections not audited'], records=records)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps({key:value for key,value in report.items() if key!='records'}))


if __name__ == '__main__':
    main()
