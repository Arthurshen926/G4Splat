"""Read-only counterfactual: missing thin geometry vs excluded contributors."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "2d-gaussian-splatting")]
import torch
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import render_hybrid,VolumetricFoliageModel
from outdoor.training_evidence import OutdoorGeometryEvidence
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.train_unified_outdoor_teacher import (
    _moge3_depth_query_bounds, _moge3_canopy_hit_owner_mask,
)
from scripts.evaluate_canopy_validation import tensor_digest


def main():
    parser = argparse.ArgumentParser()
    model = ModelParams(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evidence-store", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", default="632,690,707,768")
    parser.add_argument("--metric-scale", type=float, required=True)
    parser.add_argument('--initialization',type=Path,required=True,
                        help='Checkpoint-bound scene-canonical calibration, not global scale alone')
    parser.add_argument('--replacement-foliage',type=Path)
    parser.add_argument('--opacity-snapshot',type=Path)
    parser.add_argument("--allow-intrinsic-depth-query-repair", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = model.extract(args)
    dataset.model_path = str(args.output)
    teacher = load_hybrid_teacher(args.checkpoint, sh_degree=dataset.sh_degree,
                                 allow_intrinsic_depth_query_repair=args.allow_intrinsic_depth_query_repair)
    scene = LazyScene(dataset, GaussianModel(dataset.sh_degree), image_cache_size=8)
    views = scene.getTrainCameras()
    geometry = OutdoorGeometryEvidence(args.evidence_store, chart_base_source="moge3_adaptive")
    from scripts.seed_bound_moge_diagnostic import configure_seed_bound_moge
    calibration_audit=configure_seed_bound_moge(geometry,args.initialization,teacher.state['training_contract'],args.metric_scale)
    print(json.dumps({'diagnostic_calibration':calibration_audit}),flush=True)
    masks = CambridgeMaskLookup(Path(dataset.source_path), args.masks)
    foliage = teacher.foliage
    if args.replacement_foliage is not None:
        payload=torch.load(args.replacement_foliage,map_location='cpu')
        if not payload.get('diagnostic_only') or Path(payload['source_checkpoint']).resolve()!=args.checkpoint.resolve():
            raise ValueError('Source-matched diagnostic foliage required')
        foliage=VolumetricFoliageModel(dataset.sh_degree,dynamic_rank=payload['foliage']['dynamic_rank'],device='cuda')
        foliage.restore(payload['foliage']);teacher.foliage=foliage
    if args.opacity_snapshot is not None:
        snapshot=torch.load(args.opacity_snapshot,map_location='cpu')
        base={name:tensor_digest(getattr(foliage,name)) for name in ('xyz','log_scales','quaternions','features')}
        if (not snapshot.get('diagnostic_only') or snapshot.get('opacity_only_replay_valid') is not True
            or snapshot.get('replay_base')!=base
            or Path(snapshot['source_checkpoint']).resolve()!=args.checkpoint.resolve()):
            raise ValueError('Opacity snapshot must match the complete frozen leaf base')
        with torch.no_grad():foliage.opacity_logits.copy_(snapshot['opacity_logits'].to(foliage.opacity_logits))
    results = []
    with torch.no_grad():
        for index in map(int, args.indices.split(",")):
            view = views[index]
            fields = geometry.fields(view.image_name, device=torch.device("cuda"),
                                     shape=(view.image_height, view.image_width))
            # Global scale uncertainty affects reliability/prehit, not this
            # hit interval. This audit does not report training loss weights.
            _, bounds, valid, _, _ = _moge3_depth_query_bounds(fields, global_log_scale_sigma=0.)
            obj, sky, distortion, tree = masks.get_index_masks(
                view.image_name, (0, 1, 2, 3),
                (view.image_height, view.image_width), torch.device("cuda"))
            canopy = obj & sky & distortion & ~tree & valid
            exact = (foliage.support_camera_ids == int(view.colmap_id)).any(dim=1)
            eligible = _moge3_canopy_hit_owner_mask(foliage, exact)
            variants = {
                "eligible": eligible,
                "all_detail": foliage.static_leaf_mask,
                "all_static": ~foliage.dynamic_leaf_mask,
            }
            row = {"index": index, "image": view.image_name, "canopy_pixels": int(canopy.sum())}
            original=foliage.opacity_logits.detach().clone()
            for scale in (1.,100.,'all099'):
                try:
                    if scale=='all099':foliage.opacity_logits.fill_(torch.logit(torch.tensor(.99)).item())
                    elif scale!=1.:
                        opacity=(original.sigmoid()*scale).clamp(1e-6,.99)
                        foliage.opacity_logits.copy_(torch.logit(opacity))
                    native=teacher.render(view,task=None,conditioned=False)
                    name='native_all099' if scale=='all099' else f'native_scale{scale:g}'
                    row[name]={
                        'tree_volume_alpha':float(native['volume_alpha'].reshape_as(canopy)[canopy].mean()),
                        'tree_volume_alpha_below_half_fraction':float((native['volume_alpha'].reshape_as(canopy)[canopy]<.5).float().mean())}
                    del native
                    if scale=='all099':
                        intrinsic=render_hybrid(view,teacher.surface,foliage,background=torch.zeros(3,device='cuda'),
                            include_dynamic=False,surface_gate=torch.zeros_like(teacher.surface.get_opacity.reshape(-1)),
                            volume_gate=(~foliage.dynamic_leaf_mask).float(),optical_replacement_policy='disabled',structural_trainable_start=None)
                        row['intrinsic_all099']=float(intrinsic.volume_alpha.reshape_as(canopy)[canopy].mean())
                        del intrinsic
                finally:
                    foliage.opacity_logits.copy_(original)
            wall=render_hybrid(view,teacher.surface,foliage,background=torch.ones(3,device='cuda'),
                include_dynamic=False,volume_gate=torch.zeros_like(foliage.opacity_logits.reshape(-1)),
                optical_replacement_policy='disabled',structural_trainable_start=None)
            wall_depth=wall.median_depth.reshape_as(canopy)
            known=(wall.surface_alpha.reshape_as(canopy)>=.95)&torch.isfinite(wall_depth)&(wall_depth>0)
            center=bounds.mean(dim=0)
            row['canopy_moge_behind_known_wall_fraction']=float((known&(center>=wall_depth))[canopy].float().mean())
            del wall
            for name, gate in variants.items():
                for scale in (1., 100.):
                    package = render_hybrid(
                        view, teacher.surface, foliage,
                        background=torch.zeros(3, device="cuda"), include_dynamic=False,
                        surface_gate=torch.zeros_like(teacher.surface.get_opacity.reshape(-1)),
                        volume_gate=gate.float(), volume_depth_query_bounds=bounds,
                        volume_opacity_scale=scale, optical_replacement_policy="disabled",
                        structural_trainable_start=None,
                    )
                    a = package.volume_hit_interval_alpha.reshape_as(canopy)[canopy]
                    full = package.volume_alpha.reshape_as(canopy)[canopy]
                    row[f"{name}_scale{scale:g}"] = {
                        "rows": int(gate.sum()), "thin_mean": float(a.mean()),
                        "thin_zero_fraction": float((a <= 0).float().mean()),
                        "thin_gt035_fraction": float((a >= .35).float().mean()),
                        "full_mean": float(full.mean()),
                    }
                    del package
            results.append(row)
            print(json.dumps(row), flush=True)
    (args.output / "coverage.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "implementation_validation": teacher.state.get("_render_implementation_validation"),
        "scope": "counterfactual_only__no_parameter_or_checkpoint_edits__100x_is_not_a_proposed_fix",
        "views": results}, indent=2))


if __name__ == "__main__":
    main()
