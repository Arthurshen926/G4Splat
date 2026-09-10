"""Read-only real-scene finite differences of the scale/mass retraction.

No model is exported. Only persistent foliage scale/logits are temporarily
perturbed for finite differences; building parameters are never changed.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'2d-gaussian-splatting')]
import torch
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import (
    render_hybrid, static_detail_forward_visibility_gate,
    projected_gaussian_cross_section, VERIFICATION_VERIFIED,
)
from outdoor.mass_consistent_foliage import MassConsistentFoliage
from outdoor.moge3_evidence import sha256_file
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.evaluate_canopy_validation import tensor_digest


def main():
    p = argparse.ArgumentParser(description=__doc__); model = ModelParams(p)
    p.set_defaults(data_device='cpu', resolution=640, white_background=True)
    for key in ('checkpoint', 'cohort', 'masks', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--views', type=int, nargs='+', default=[664, 665, 666])
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(.32)
    cohort = json.loads(a.cohort.read_text()); source_hash = sha256_file(a.checkpoint)
    if source_hash != cohort['source_checkpoint_sha256'] or not set(a.views) <= set(cohort['calibrated_training_views']):
        raise ValueError('Source-matched training-only audit cameras required')
    teacher = load_hybrid_teacher(a.checkpoint, sh_degree=a.sh_degree); raw = teacher.foliage
    fixed = MassConsistentFoliage(a.sh_degree, dynamic_rank=raw.dynamic_rank).cuda()
    fixed.restore(teacher.state['foliage'])
    for module in (raw, fixed, teacher.sky, teacher.appearance):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    buildings = {k: getattr(teacher.surface, k) for k in
                 ('_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest')}
    for parameter in buildings.values(): parameter.requires_grad_(False)
    building_before = {k: tensor_digest(v) for k, v in buildings.items()}
    for cloud in (raw, fixed):
        cloud.log_scales.requires_grad_(True); cloud.opacity_logits.requires_grad_(True)
    dataset = model.extract(a); dataset.model_path = str(a.output)
    views = LazyScene(dataset, GaussianModel(a.sh_degree), image_cache_size=3).getTrainCameras()
    masks = CambridgeMaskLookup(Path(dataset.source_path), a.masks)
    allowed = (raw.verification_state == VERIFICATION_VERIFIED) & raw.static_leaf_mask
    original_scales = fixed.log_scales.detach().clone()
    original_logits = fixed.opacity_logits.detach().clone()
    original_area = projected_gaussian_cross_section(original_scales.exp())
    original_tau = -torch.log1p(-original_logits.sigmoid().reshape(-1).clamp(0., 1.-1e-6))
    records = []
    for index in a.views:
        camera = views[index]
        obj, sky, dist, tree = masks.get_index_masks(camera.image_name, (0, 1, 2, 3),
            (camera.image_height, camera.image_width), torch.device('cuda'))
        canopy = obj & sky & dist & ~tree
        if not canopy.any(): raise ValueError('Canopy audit requires real canopy pixels')
        target = camera.original_image.cuda()
        gate = static_detail_forward_visibility_gate(raw, int(camera.colmap_id), include_pending_exact=False)
        def objective(cloud):
            package = render_hybrid(camera, teacher.surface, cloud,
                background=torch.zeros(3, device='cuda'), include_dynamic=False,
                optical_replacement_policy='disabled', structural_trainable_start=None,
                volume_gate=gate, volume_geometry_gradient_gate=allowed.float())
            rgb = package.render+(1-package.alpha)*teacher.sky(camera)
            return (rgb[:, canopy].double()-target[:, canopy].double()).square().mean(), rgb
        gradients = []
        source_rgb = None
        for cloud in (raw, fixed):
            loss, rgb = objective(cloud)
            if source_rgb is None: source_rgb = rgb.detach()
            elif not torch.equal(source_rgb, rgb.detach()): raise RuntimeError('Mass wrapper changed forward RGB')
            gradients.append(torch.autograd.grad(loss, (cloud.opacity_logits, cloud.log_scales)))
        raw_l, raw_s = gradients[0]; fixed_l, fixed_s = gradients[1]
        if not torch.allclose(raw_l, fixed_l, atol=1e-8, rtol=1e-5): raise RuntimeError('Opacity gradient changed')
        if torch.count_nonzero(fixed_s[~allowed]): raise RuntimeError('Forbidden row gained scale authority')
        active = (raw_s.abs().sum(1)+fixed_s.abs().sum(1)) > 1e-12
        dot = (raw_s*fixed_s).sum(1)
        delta = fixed_s-raw_s
        direction = delta/delta.abs().max().clamp_min(1e-20)
        changed = direction.abs().sum(1) > 0
        samples = []
        try:
            with torch.no_grad():
                for h in (.01, .003):
                    losses = []
                    for sign in (1., -1.):
                        fixed.log_scales.copy_(original_scales+sign*h*direction)
                        area = projected_gaussian_cross_section(fixed.scales).clamp_min(1e-12)
                        alpha = -torch.expm1(-original_tau*original_area/area)
                        fixed.opacity_logits.copy_(original_logits)
                        fixed.opacity_logits[changed] = torch.logit(alpha[changed].clamp(1e-6, 1.-1e-6))[:, None]
                        losses.append(float(objective(fixed)[0]))
                    samples.append(dict(epsilon=h, derivative=(losses[0]-losses[1])/(2*h)))
        finally:
            with torch.no_grad():
                fixed.log_scales.copy_(original_scales); fixed.opacity_logits.copy_(original_logits)
        records.append(dict(index=index, active_persistent_rows=int(active.sum()),
            opposite_scale_descent_rows=int((active & (dot < 0)).sum()),
            raw_prediction=float((raw_s*direction).sum()),
            corrected_prediction=float((fixed_s*direction).sum()), finite_differences=samples,
            raw_scale_gradient_l1=float(raw_s.abs().sum()), corrected_scale_gradient_l1=float(fixed_s.abs().sum())))
        print(json.dumps(records[-1]), flush=True)
    unchanged = (all(tensor_digest(v) == building_before[k] for k, v in buildings.items())
                 and torch.equal(fixed.log_scales, original_scales)
                 and torch.equal(fixed.opacity_logits, original_logits))
    if not unchanged: raise RuntimeError('Read-only audit failed to restore source')
    report = dict(scope='training_camera_read_only_scale_mass_jacobian', unchanged=True,
        source_checkpoint_sha256=source_hash, views=records,
        helper_sha256=sha256_file(ROOT/'outdoor/optical_mass_gradient.py'),
        wrapper_sha256=sha256_file(ROOT/'outdoor/mass_consistent_foliage.py'),
        script_sha256=sha256_file(Path(__file__)))
    (a.output/'audit.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__': main()
