"""Read-only native visibility on the complete training occluder certificate.

Low native leaf alpha is an attribution measurement, not ground-truth error:
real holes can be transparent. No reporting mask changes the native render.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / '2d-gaussian-splatting')]
import torch
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import VolumetricFoliageModel
from outdoor.canopy_occluder_loss import validate_consensus_coverage
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.evaluate_canopy_validation import FIXED, ADDITIONAL, tensor_digest
from scripts.train_unified_outdoor_teacher import _file_sha256


def alpha_statistics(alpha, selected):
    values = alpha.reshape_as(selected)[selected]
    coverage = {f'below_{label}_pixels': int((values < threshold).sum())
                for label, threshold in [('1e_6',1.e-6),('0_01',.01),('0_1',.1),('0_5',.5)]}
    if not values.numel():
        return {'pixels': 0, 'alpha_sum': 0., 'below_0_9_pixels': 0, **coverage}
    return {'pixels': values.numel(), 'alpha_sum': float(values.sum()),
            'mean': float(values.mean()),
            'q10_q50_q90': values.quantile(values.new_tensor([.1,.5,.9])).tolist(),
            'below_0_9_pixels': int((values < .9).sum()), **coverage}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    parser.set_defaults(data_device='cpu', resolution=640, white_background=True)
    for name in ('checkpoint', 'evidence', 'masks', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--foliage', type=Path)
    parser.add_argument('--intrinsic-visibility-counterfactual', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((args.evidence/'audit.json').read_text())
    validate_consensus_coverage(manifest, manifest['calibrated_training_views'], FIXED+ADDITIONAL)
    if (Path(manifest['source_checkpoint']).resolve() != args.checkpoint.resolve()
        or manifest['source_checkpoint_sha256'] != _file_sha256(args.checkpoint)):
        raise ValueError('Evidence and source checkpoint differ')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.25)
    dataset = model.extract(args); dataset.model_path = str(args.output)
    teacher = load_hybrid_teacher(args.checkpoint, sh_degree=dataset.sh_degree)
    if args.foliage:
        capture = torch.load(args.foliage, map_location='cpu')
        if (not capture.get('diagnostic_only') or
            Path(capture['source_checkpoint']).resolve() != args.checkpoint.resolve()):
            raise ValueError('Matched diagnostic capture required')
        foliage = VolumetricFoliageModel(dataset.sh_degree,
            dynamic_rank=capture['foliage']['dynamic_rank'], device='cuda')
        foliage.restore(capture['foliage']); teacher.foliage = foliage
    views = LazyScene(dataset, GaussianModel(dataset.sh_degree), image_cache_size=6).getTrainCameras()
    masks = CambridgeMaskLookup(Path(dataset.source_path), args.masks)
    fingerprints = {name: tensor_digest(getattr(teacher.surface, name))
                    for name in ('_xyz', '_scaling', '_rotation', '_opacity')}
    records = []
    for index in manifest['canopy_training_views']:
        view = views[index]
        obj, sky, distortion, tree = masks.get_index_masks(view.image_name, (0,1,2,3),
            (view.image_height, view.image_width), torch.device('cuda'))
        canopy = obj & sky & distortion & ~tree
        certificate = torch.load(args.evidence/f'evidence_{index}.pth', map_location='cpu')
        candidate = certificate['candidate'].cuda()
        confirmed = certificate['confirmed'].cuda()
        if (candidate.dtype != torch.bool or confirmed.dtype != torch.bool
            or candidate.shape != canopy.shape or confirmed.shape != canopy.shape
            or (confirmed & ~candidate).any() or (candidate & ~canopy).any()):
            raise ValueError('Certificate does not match current observed canopy')
        native = teacher.render(view, task=None, conditioned=False)
        alpha = native['volume_alpha'].reshape_as(canopy)
        record = {'index': index, 'image': view.image_name,
                  'regions': {name: alpha_statistics(alpha, mask) for name, mask in
                              [('canopy', canopy), ('candidate', candidate), ('confirmed', confirmed)]}}
        if args.intrinsic_visibility_counterfactual:
            intrinsic=teacher.render(view,task=None,conditioned=False,volume_only=True)['volume_alpha'].reshape_as(canopy)
            for name,mask in [('canopy',canopy),('candidate',candidate),('confirmed',confirmed)]:
                record['regions'][name]['intrinsic']=alpha_statistics(intrinsic,mask)
                record['regions'][name]['native_low_intrinsic_high_pixels']=int((mask&(alpha<.5)&(intrinsic>=.9)).sum())
        records.append(record)
        print(json.dumps(record), flush=True)
    totals = {}
    for name in ('canopy', 'candidate', 'confirmed'):
        totals[name] = {key: sum(row['regions'][name][key] for row in records)
                       for key in ('pixels', 'alpha_sum', 'below_0_9_pixels',
                                   'below_1e_6_pixels','below_0_01_pixels',
                                   'below_0_1_pixels','below_0_5_pixels')}
        if args.intrinsic_visibility_counterfactual:
            totals[name]['intrinsic']={key:sum(row['regions'][name]['intrinsic'][key] for row in records)
                                      for key in ('pixels','alpha_sum','below_1e_6_pixels','below_0_01_pixels',
                                                  'below_0_1_pixels','below_0_5_pixels','below_0_9_pixels')}
            totals[name]['native_low_intrinsic_high_pixels']=sum(row['regions'][name]['native_low_intrinsic_high_pixels'] for row in records)
    result = {'scope': 'native_training_certificate_attribution_not_opacity_ground_truth',
              'source_checkpoint': str(args.checkpoint.resolve()),
              'foliage': str(args.foliage.resolve()) if args.foliage else None,
              'evidence': str(args.evidence.resolve()), 'surface_fingerprints': fingerprints,
              'intrinsic_counterfactual_is_not_a_repair':args.intrinsic_visibility_counterfactual,
              'totals': totals, 'records': records}
    (args.output/'audit.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
