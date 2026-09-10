"""Bind canonical refinement cameras to a checkpoint, without emitting priors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / '2d-gaussian-splatting')]
import torch
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from scripts.evaluate_canopy_validation import FIXED, ADDITIONAL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    parser.set_defaults(data_device='cpu', resolution=640, white_background=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    before = args.checkpoint.stat()
    digest = hashlib.sha256()
    with args.checkpoint.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    state = torch.load(args.checkpoint, map_location='cpu')
    canonical = state['training_contract']['static_scene_canonical_rgb']['canonical_sequence']
    del state
    dataset = model.extract(args)
    dataset.model_path = str(args.output)
    views = LazyScene(dataset, GaussianModel(dataset.sh_degree), image_cache_size=2).getTrainCameras()
    excluded = FIXED + ADDITIONAL
    if any(not views[i].image_name.startswith(canonical + '__') for i in excluded):
        raise ValueError('All validation cameras must belong to the canonical sequence')
    training = [i for i, view in enumerate(views)
                if view.image_name.startswith(canonical + '__') and i not in excluded]
    if not training:
        raise ValueError('No canonical training views')
    after = args.checkpoint.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError('Checkpoint changed during cohort construction')
    manifest = dict(contract='canonical_camera_cohort_only__not_depth_or_opacity_evidence',
        source_checkpoint=str(args.checkpoint.resolve()), source_checkpoint_sha256=digest.hexdigest(),
        calibrated_training_views=training, excluded_views=excluded,
        canonical_sequence=canonical, source_path=str(Path(dataset.source_path).resolve()),
        camera_names={i: views[i].image_name for i in training + excluded})
    (args.output / 'cohort.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'training_views': len(training), 'excluded_views': len(excluded),
                      'manifest': str(args.output / 'cohort.json')}), flush=True)


if __name__ == '__main__':
    main()
