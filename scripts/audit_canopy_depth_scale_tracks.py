"""CPU-only comparison of saved descriptor depths and seed-bound MoGe scales.

Read-only: neither descriptor matches nor monocular predictions are ground
truth. Pair hypotheses are deliberately rejected; no correction is exported.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from outdoor.moge3_evidence import load_index, load_view, sha256_file
from scripts.extract_canopy_render_state import DeferredUnpickler


def depth_errors(prediction, reference):
    prediction = np.asarray(prediction)
    reference = np.asarray(reference)
    valid = np.isfinite(prediction) & np.isfinite(reference) & (prediction > 0) & (reference > 0)
    if not valid.any():
        return dict(count=0, median_log_ratio=None, median_absolute_log_error=None)
    residual = np.log(prediction[valid] / reference[valid])
    return dict(count=int(valid.sum()), median_log_ratio=float(np.median(residual)),
                median_absolute_log_error=float(np.median(np.abs(residual))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('track-audit', 'initialization', 'moge-index', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--metric-scale', type=float, required=True)
    a = p.parse_args()
    if not np.isfinite(a.metric_scale) or a.metric_scale <= 0:
        raise ValueError('Positive finite global scale required')
    audit = json.loads((a.track_audit / 'audit.json').read_text())
    kind = audit.get('evidence_kind')
    if kind is None and audit.get('input_sha256', {}).get('cycle_files'):
        for filename, digest in audit['input_sha256']['cycle_files'].items():
            path = Path(filename)
            if len(path.stem.split('_')) != 4 or not path.name.startswith('tracks_') or sha256_file(path) != digest:
                raise ValueError('Unverifiable legacy three-edge cycle source')
        kind = 'mast3r_reciprocal_three_edge_cycles'
    if not audit['complete_scan'] or kind not in (
            'mast3r_reciprocal_descriptor_union_find', 'mast3r_reciprocal_three_edge_cycles'):
        raise ValueError('Complete independently closed track screen required; two-view hypotheses rejected')
    cameras = {c['img_name']: c for c in json.loads((a.track_audit / 'cameras.json').read_text())}
    manifest = json.loads((a.initialization / 'initialization_manifest.json').read_text())
    seed = Path(manifest['foliage_seed'])
    reader = torch._C.PyTorchFileReader(str(seed))
    metadata = DeferredUnpickler(io.BytesIO(reader.get_record('data.pkl'))).load()
    profiles = metadata['audit']['fresh_canonical_leaves']['depth_profiles_by_image']
    index = load_index(a.moge_index, verify_views=False)
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    rows = []
    for record in audit['records']:
        name = record['image_name']
        if name not in profiles:
            continue  # Report only cameras to which this seed calibration applies.
        if record['index'] in audit['excluded_views']:
            raise ValueError('Validation camera in diagnostic source')
        source = index['records'][name]
        path = Path(source['path'])
        if path.stat().st_size != source['bytes'] or sha256_file(path) != source['sha256']:
            raise ValueError('Changed MoGe input')
        depth = load_view(path)
        camera = cameras[name]
        shape = (camera['height'], camera['width'])
        z = F.interpolate(torch.from_numpy(depth['depth_m'].copy())[None, None],
                          size=shape, mode='bilinear', align_corners=False)[0, 0].numpy() * a.metric_scale
        mask = depth['valid_mask'].astype(bool) & depth['refinement_valid_mask'].astype(bool)
        mask = F.interpolate(torch.from_numpy(mask.astype(np.float32))[None, None],
                             size=shape, mode='nearest')[0, 0].numpy() > .5
        observation_path = a.track_audit / ('observations_%s.npz' % record['index'])
        with np.load(observation_path, allow_pickle=False) as observations:
            x, y = np.rint(observations['pixels']).astype(int).T
            inside = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
            x = x.clip(0, shape[1] - 1); y = y.clip(0, shape[0] - 1)
            valid = observations['valid'] & inside & mask[y, x]
            regions = {}
            for label, subset in (('canopy', observations['canopy']), ('rigid', ~observations['canopy'])):
                keep = valid & subset
                reference = observations['track_depth'][keep]
                raw = z[y, x][keep]
                regions[label] = dict(global_moge=depth_errors(raw, reference),
                    calibrated_moge=depth_errors(raw * profiles[name], reference),
                    native_rigid=depth_errors(observations['rigid_depth'][keep], reference))
        row = dict(index=record['index'], image_name=name, scale=profiles[name], regions=regions,
                   observations_sha256=sha256_file(observation_path), moge_sha256=source['sha256'])
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = dict(scope='read_only_depth_calibration_track_comparison_not_correction_authority',
                  metric_scale=a.metric_scale, records=rows,
                  source_audit_sha256=sha256_file(a.track_audit / 'audit.json'),
                  source_seed=str(seed), seed_metadata_sha256=hashlib.sha256(reader.get_record('data.pkl')).hexdigest(),
                  moge_index_sha256=sha256_file(a.moge_index),
                  limitations=['Sparse tracks are not canopy ground truth',
                               'Valid observations inherit the source native rigid visibility screen',
                               'Three-view closure does not guarantee a physically static leaf'])
    (a.output / 'audit.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
