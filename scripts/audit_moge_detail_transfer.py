"""Training-view native-to-cell evidence audit, with explicit layer proposals."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from outdoor.moge3_depth_layers import reduce_depth_layers
from scripts.build_moge3_chart_base import _resize_scalar, _resize_normal
from matcha.cambridge_masks import CambridgeMaskLookup


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--depth-directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((a.run/'manifest.json').read_text())
    cameras = json.loads((a.run/'cameras.json').read_text())
    train = manifest['training_views']; excluded = manifest['excluded_views']
    if set(train) & set(excluded): raise ValueError('Training/evaluation overlap')
    masks = CambridgeMaskLookup(Path(manifest['args']['source_path']), Path(manifest['args']['masks']))
    rows = []; torch.set_num_threads(4)
    for ordinal, index in enumerate(train):
        camera = cameras[index]; path = a.depth_directory/(camera['img_name']+'.npz')
        with np.load(path, allow_pickle=False) as z:
            depth = z['depth_m']; valid = z['valid_mask'].astype(bool) & z['refinement_valid_mask'].astype(bool)
            shape = (camera['height'], camera['width'])
            layers = reduce_depth_layers(depth, valid, shape)
            area, dv = _resize_scalar(depth, valid, shape)
            _, nv = _resize_normal(z['normal_direct_camera'], valid, shape)
            _, derived = _resize_normal(z['normal_depth_exact_k_camera'], valid & z['depth_normal_valid_mask'].astype(bool), shape)
        obj, sky, dist, non_tree = masks.get_index_masks(camera['img_name'], (0,1,2,3), shape, torch.device('cpu'))
        common = (obj & sky & dist).numpy()
        regions = {'rigid': common & non_tree.numpy(), 'tree': common & ~non_tree.numpy()}
        stats = {}
        for name, region in regions.items():
            selected = region & dv
            stats[name] = dict(valid_depth_cells=int(selected.sum()),
                mixed_cells=int((selected & layers['mixed']).sum()),
                exactly_two_mode_cells=int((selected & (layers['mode_count']==2)).sum()),
                unresolved_cells=int((selected & layers['unresolved']).sum()),
                depth_rejected_by_normal=int((selected & ~(nv & derived)).sum()),
                continuous_depth_without_normal=int((selected & layers['continuous_valid'] & ~(nv & derived)).sum()))
        rows.append(dict(index=index, image=camera['img_name'], regions=stats))
        # A bounded evidence packet for every training view: original sample
        # rays survive and can later be re-associated across views. No point
        # has been granted geometry authority or inserted into the model.
        mixed = layers['mixed'] & common & ~layers['unresolved']
        y, x = np.nonzero(mixed)
        packet = {k: v[y,x] for k,v in layers.items()}
        packet.update(cell_x=x.astype(np.int32), cell_y=y.astype(np.int32),
                      rigid=non_tree.numpy()[y,x], raw_area_depth=area[y,x])
        np.savez_compressed(a.output/(camera['img_name']+'.npz'), **packet)
        if ordinal % 25 == 0: print(json.dumps(dict(completed=ordinal+1,total=len(train))), flush=True)
    totals = {region: {key:sum(r['regions'][region][key] for r in rows)
                       for key in rows[0]['regions'][region]} for region in ('rigid','tree')}
    report = dict(scope='training_only_native_depth_transfer__unverified_layer_proposals',
                  training_views=train, excluded_views=excluded, totals=totals, per_view=rows,
                  limitations=['Native predictions are not independently verified geometry',
                               'No Chart/initialization/handoff coverage claim yet',
                               'Layer representatives retain original rays; no cell-centre backprojection',
                               'Normal acceptance counts isolate normal validity, not all Chart gates'])
    with (a.output/'audit.json').open('x') as f: json.dump(report,f,indent=2)
    print(json.dumps(totals),flush=True)


if __name__ == '__main__': main()
