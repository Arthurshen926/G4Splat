"""Read-only native RGB gradient decomposition on non-validation training views."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / '2d-gaussian-splatting')]
import torch
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import VolumetricFoliageModel
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.evaluate_canopy_validation import FIXED, ADDITIONAL, tensor_digest
from scripts.evaluate_hybrid_teacher import _tree_boundary_masks


def gradient_summary(canopy, other):
    want_growth = canopy < -1.e-10
    opposed = want_growth & (other > 0)
    reversed_direction = want_growth & (canopy + other >= 0)
    denominator = (-canopy[want_growth]).sum().clamp_min(1.e-20)
    return {'canopy_growth_rows': int(want_growth.sum()), 'opposed_rows': int(opposed.sum()),
            'reversed_rows': int(reversed_direction.sum()),
            'reversed_canopy_gradient_mass_fraction': float((-canopy[reversed_direction]).sum() / denominator),
            'cosine': float((canopy * other).sum() / (canopy.norm() * other.norm()).clamp_min(1.e-20))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    parser.set_defaults(data_device='cpu', resolution=640, white_background=True)
    for name in ('checkpoint', 'foliage', 'masks', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--views', type=int, default=12)
    args = parser.parse_args()
    if args.views < 1:
        raise ValueError('Positive view count required')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.30)
    dataset = model.extract(args)
    dataset.model_path = str(args.output)
    teacher = load_hybrid_teacher(args.checkpoint, sh_degree=dataset.sh_degree)
    original = teacher.foliage
    capture = torch.load(args.foliage, map_location='cpu')
    if not capture.get('diagnostic_only') or Path(capture['source_checkpoint']).resolve() != args.checkpoint.resolve():
        raise ValueError('Matched diagnostic required')
    leaf = VolumetricFoliageModel(dataset.sh_degree, dynamic_rank=capture['foliage']['dynamic_rank'], device='cuda')
    leaf.restore(capture['foliage'])
    if leaf.dynamic_leaf_mask.any():
        raise ValueError('Static canopy required')
    for parameter in leaf.parameters():
        parameter.requires_grad_(False)
    leaf.opacity_logits.requires_grad_(True)
    teacher.foliage = leaf
    before = {key: tensor_digest(getattr(leaf, key)) for key in ('xyz', 'opacity_logits', 'features', 'verification_state')}
    views = LazyScene(dataset, GaussianModel(dataset.sh_degree), image_cache_size=6).getTrainCameras()
    masks = CambridgeMaskLookup(Path(dataset.source_path), args.masks)
    canonical = teacher.state['training_contract']['static_scene_canonical_rgb']['canonical_sequence']
    eligible = []
    for index, view in enumerate(views):
        if index in set(FIXED + ADDITIONAL) or not view.image_name.startswith(canonical + '__'):
            continue
        obj, sky, distortion, tree = masks.get_index_masks(view.image_name, (0, 1, 2, 3),
            (view.image_height, view.image_width), torch.device('cpu'))
        if int((obj & sky & distortion & ~tree).sum()) >= 128:
            eligible.append(index)
    if not eligible:
        raise ValueError('No eligible canopy training camera')
    selected = [eligible[i] for i in torch.linspace(0, len(eligible)-1, min(args.views, len(eligible))).round().long().tolist()]
    records = []
    cumulative_tree = torch.zeros_like(leaf.opacity_logits)
    cumulative_other = torch.zeros_like(leaf.opacity_logits)
    for index in selected:
        view = views[index]
        obj, sky, distortion, tree = masks.get_index_masks(view.image_name, (0, 1, 2, 3),
            (view.image_height, view.image_width), torch.device('cuda'))
        canopy = obj & sky & distortion & ~tree
        rigid = obj & sky & distortion & tree
        known_sky = obj & distortion & ~sky
        _, outside, _ = _tree_boundary_masks(~canopy)
        hard = rigid & outside
        with torch.no_grad():
            teacher.foliage = original
            try:
                reference = teacher.render(view, task=None, conditioned=False)['rgb']
            finally:
                teacher.foliage = leaf
        native = teacher.render(view, task=None, conditioned=False)
        error = (native['rgb'] - view.original_image.cuda()).abs().mean(dim=0)
        delta = (native['rgb'] - reference).abs().mean(dim=0)
        mean = lambda image, mask: (image * mask).sum() / mask.sum().clamp_min(1)
        tree_loss = mean(error, canopy)
        other_loss = mean(error, rigid) + 3 * mean(delta, rigid) + 3 * mean(delta, hard)
        other_loss = other_loss + .1 * mean(torch.log1p(native['volume_alpha'].reshape_as(known_sky).clamp_min(0) / .005), known_sky)
        grad_tree = torch.autograd.grad(tree_loss, leaf.opacity_logits, retain_graph=True)[0].detach()
        grad_other = torch.autograd.grad(other_loss, leaf.opacity_logits)[0].detach()
        cumulative_tree += grad_tree
        cumulative_other += grad_other
        record = {'index': index, 'canopy_loss': float(tree_loss), 'other_rgb_loss': float(other_loss),
                  **gradient_summary(grad_tree, grad_other)}
        records.append(record)
        print(json.dumps(record), flush=True)
        del native
    assert before == {key: tensor_digest(getattr(leaf, key)) for key in before}
    audit = {'source_foliage': str(args.foliage.resolve()), 'records': records,
             'aggregate_opacity_rgb_gradient': gradient_summary(cumulative_tree, cumulative_other),
             'scope': 'read_only_rgb_components__excludes_moge_and_geometry_gradients',
             'selected_training_views': selected, 'excluded_view_indices': sorted(FIXED + ADDITIONAL)}
    (args.output / 'audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit['aggregate_opacity_rgb_gradient']), flush=True)


if __name__ == '__main__':
    main()
