"""Paired persistent-leaf optics/scale calibration; all buildings frozen."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'2d-gaussian-splatting')]
import numpy as np
import torch
from PIL import Image
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import (
    render_hybrid, static_detail_forward_visibility_gate,
    persistent_static_evidence_mask, projected_gaussian_cross_section,
)
from outdoor.mass_consistent_foliage import MassConsistentFoliage, IdentityPreservingMassFoliage
from outdoor.moge3_evidence import sha256_file
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.evaluate_canopy_validation import FIXED, ADDITIONAL, tensor_digest
from scripts.evaluate_hybrid_teacher import _tree_boundary_masks
from scripts.canopy_candidate_rgb_ownership import candidate_rgb_loss
from scripts.canopy_candidate_ray_refinement import rigid_rgb_preservation


def main():
    p = argparse.ArgumentParser(description=__doc__); model = ModelParams(p)
    p.set_defaults(data_device='cpu', resolution=640, white_background=True)
    for key in ('checkpoint', 'cohort', 'masks', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--consistent', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--steps', type=int, default=800)
    p.add_argument('--eval-every', type=int, default=400)
    p.add_argument('--scale-lr', type=float, default=.0004)
    p.add_argument('--opacity-lr', type=float, default=.02)
    p.add_argument('--color-lr', type=float, default=.0025)
    a = p.parse_args()
    if a.steps < 1 or a.eval_every < 1 or not all(0 < x <= .05 for x in (a.scale_lr, a.opacity_lr, a.color_lr)):
        raise ValueError('Bounded positive calibration schedule required')
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(.36)
    cohort = json.loads(a.cohort.read_text()); train = sorted(cohort['calibrated_training_views']); holdout = FIXED+ADDITIONAL
    source_hash = sha256_file(a.checkpoint)
    if source_hash != cohort['source_checkpoint_sha256'] or set(train)&set(holdout) or set(holdout) != set(cohort['excluded_views']):
        raise ValueError('Source-matched camera cohort excluding all48 required')
    teacher = load_hybrid_teacher(a.checkpoint, sh_degree=a.sh_degree); source = teacher.foliage
    cls = MassConsistentFoliage if a.consistent else IdentityPreservingMassFoliage
    leaf = cls(a.sh_degree, dynamic_rank=source.dynamic_rank).cuda(); leaf.restore(teacher.state['foliage'])
    immutable = {f'leaf.{k}': v for k, v in leaf.named_parameters() if k not in ('log_scales', 'opacity_logits', 'features')}
    immutable.update({f'rigid.{k}': getattr(teacher.surface, k) for k in
                      ('_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest')})
    for label, module in [('source', source), ('sky', teacher.sky), ('appearance', teacher.appearance)]:
        immutable.update({f'{label}.{k}': v for k, v in module.named_parameters()})
    for parameter in immutable.values(): parameter.requires_grad_(False)
    eligible = persistent_static_evidence_mask(leaf)&leaf.static_leaf_mask
    if not eligible.any() or leaf.dynamic_leaf_mask.any(): raise ValueError('Persistent static foliage required')
    for parameter in (leaf.log_scales, leaf.opacity_logits, leaf.features):
        parameter.requires_grad_(True)
        parameter.register_hook(lambda grad: grad*eligible.reshape((-1,)+(1,)*(grad.ndim-1)))
    initial_scales = leaf.log_scales.detach().clone()
    immutable.update({f'ineligible.{k}': getattr(leaf, k)[~eligible].detach().clone()
                      for k in ('log_scales', 'opacity_logits', 'features')})
    before = {k: tensor_digest(v) for k, v in immutable.items()}
    dataset = model.extract(a); dataset.model_path = str(a.output)
    views = LazyScene(dataset, GaussianModel(a.sh_degree), image_cache_size=8).getTrainCameras()
    masks = CambridgeMaskLookup(Path(dataset.source_path), a.masks)
    canonical = teacher.state['training_contract']['static_scene_canonical_rgb']['canonical_sequence']
    if any(not views[i].image_name.startswith(canonical+'__') for i in train+holdout):
        raise ValueError('One static canonical sequence required')
    def regions(v):
        obj, sky, dist, tree = masks.get_index_masks(v.image_name, (0, 1, 2, 3),
            (v.image_height, v.image_width), torch.device('cuda'))
        known = obj&sky&dist; canopy = known&~tree; rigid = known&tree
        inside, outside, _ = _tree_boundary_masks(~canopy)
        return dict(tree=canopy, tree_interior=canopy&~inside, tree_boundary=canopy&inside,
                    rigid=rigid, hard=rigid&outside, sky=obj&dist&~sky)
    def render(v, cloud):
        gate = static_detail_forward_visibility_gate(cloud, int(v.colmap_id), include_pending_exact=False)
        out = render_hybrid(v, teacher.surface, cloud, background=torch.zeros(3, device='cuda'),
            include_dynamic=False, optical_replacement_policy='disabled', structural_trainable_start=None,
            volume_gate=gate, volume_geometry_gradient_gate=eligible.float())
        return out.render+(1-out.alpha)*teacher.sky(v)
    manifest = dict(scope='persistent_leaf_scale_color_opacity__all_other_scene_parameters_frozen',
        args=vars(a), source_checkpoint_sha256=source_hash, training_views=train, excluded_views=holdout,
        eligible_rows=int(eligible.sum()), masks_sha256=sha256_file(a.masks),
        script_sha256=sha256_file(Path(__file__)),
        helper_sha256={name: sha256_file(ROOT/name) for name in
            ('outdoor/optical_mass_gradient.py', 'outdoor/mass_consistent_foliage.py',
             'scripts/canopy_candidate_rgb_ownership.py', 'scripts/canopy_candidate_ray_refinement.py')},
        caveat='Historical training images, not blind generalization; diagnostic delta, not production export')
    (a.output/'manifest.json').write_text(json.dumps(manifest, default=str, indent=2))
    optimizer = torch.optim.Adam([dict(params=[leaf.log_scales], lr=a.scale_lr),
        dict(params=[leaf.opacity_logits], lr=a.opacity_lr), dict(params=[leaf.features], lr=a.color_lr)], eps=1e-15)
    initial_lrs = [a.scale_lr, a.opacity_lr, a.color_lr]; results = []
    @torch.no_grad()
    def evaluate(step):
        rows = []
        for index in holdout:
            v = views[index]; m = regions(v); target = v.original_image.cuda()
            predictions = {'refined': render(v, leaf).clamp(0, 1)}
            if not step:
                predictions['source'] = teacher.render(v, task=None, conditioned=False)['rgb']
                if not torch.allclose(predictions['refined'], predictions['source'], atol=1e-6, rtol=0):
                    raise RuntimeError('Zero-step optics differs from native source')
            row = dict(index=index, modes={})
            for name, rgb in predictions.items():
                row['modes'][name] = {k: float(-10*(rgb[:, mask]-target[:, mask]).square().mean().clamp_min(1e-12).log10())
                    if mask.any() else None for k, mask in m.items() if k != 'sky'}
            pair = torch.cat((target, predictions['refined']), 2).permute(1, 2, 0).cpu().numpy()
            Image.fromarray((pair*255).round().astype('uint8')).save(a.output/f'view_{index}_{step:04d}.png')
            rows.append(row)
        summary = {mode: {k: float(np.mean([r['modes'][mode][k] for r in rows if r['modes'][mode][k] is not None]))
            for k in rows[0]['modes'][mode]} for mode in rows[0]['modes']}
        results.append(dict(step=step, summary=summary, per_view=rows))
        (a.output/'metrics.json').write_text(json.dumps(dict(results=results), indent=2))
        print(json.dumps(dict(evaluation=step, summary=summary)), flush=True)
        if step:
            torch.save(dict(diagnostic_only=True, step=step, manifest=manifest,
                leaf={k: getattr(leaf, k).detach().cpu() for k in ('log_scales', 'opacity_logits', 'features')}),
                a.output/f'optics_{step:04d}.pth')
    evaluate(0)
    order = [train[i] for i in torch.randperm(len(train), generator=torch.Generator().manual_seed(1701)).tolist()]
    for step in range(1, a.steps+1):
        v = views[order[(step-1)%len(order)]]; m = regions(v)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(): reference = render(v, source)
        rgb = render(v, leaf)
        loss = candidate_rgb_loss(rgb, v.original_image.cuda(), reference, m, noncanopy_target='source_reference')
        loss = loss+3.*rigid_rgb_preservation(rgb, reference, m['rigid'])
        loss.backward()
        parameters = (leaf.log_scales, leaf.opacity_logits, leaf.features)
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters):
            raise RuntimeError('Missing/nonfinite permitted gradient')
        if any(v.grad is not None for k, v in immutable.items() if not k.startswith('ineligible.')):
            raise RuntimeError('Frozen parameter received gradient')
        scale_grad = float(leaf.log_scales.grad.abs().sum())
        with torch.no_grad(): old_area = projected_gaussian_cross_section(leaf.scales)
        for group, lr in zip(optimizer.param_groups, initial_lrs):
            group['lr'] = lr*.1**max(0., 2*(step-1)/max(a.steps-1, 1)-1)
        optimizer.step()
        with torch.no_grad():
            tau = -torch.log1p(-leaf.opacities.clamp(0., 1.-1e-6))
            mass = tau*old_area
            leaf.log_scales[eligible] = torch.maximum(torch.minimum(leaf.log_scales[eligible],
                initial_scales[eligible]+math.log(2.)), initial_scales[eligible]-math.log(2.))
            target_mass = leaf.integrated_optical_mass().clone()
            target_mass[eligible] = mass[eligible]
            leaf.restore_integrated_optical_mass(target_mass, maximum_opacity=.995, rows=eligible)
            rest = leaf.features[eligible, 1:]
            factor = (4./rest.flatten(1).norm(dim=1).clamp_min(1e-12)).clamp_max(1.)
            leaf.features[eligible, 1:] = rest*factor[:, None, None]
        if step == 1 or step%25 == 0 or step == a.steps:
            record = dict(step=step, loss=float(loss), scale_gradient_l1=scale_grad,
                mean_eligible_opacity=float(leaf.opacities[eligible].mean()),
                maximum_scale_log_shift=float((leaf.log_scales-initial_scales).abs().max()),
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
            with (a.output/'trace.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
        if step%a.eval_every == 0 or step == a.steps: evaluate(step)
    after = {k: tensor_digest(getattr(leaf, k.split('.')[1])[~eligible]) if k.startswith('ineligible.')
             else tensor_digest(v) for k, v in immutable.items()}
    if before != after: raise RuntimeError('Protected source state changed')
    (a.output/'frozen_parameter_audit.json').write_text(json.dumps(dict(unchanged=True, fingerprints=after), indent=2))


if __name__ == '__main__': main()
