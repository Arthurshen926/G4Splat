"""Read-only canonical MoGe cross-view depth audit; disagreement is not proof of a bug.

Occlusion and different visible leaves can legitimately disagree. Report signed
residuals and local depth ranges, never turn these statistics into render gates.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "2d-gaussian-splatting")]
import torch
import torch.nn.functional as F
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.training_evidence import OutdoorGeometryEvidence
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.train_unified_outdoor_teacher import _moge3_depth_query_bounds
from scripts.evaluate_canopy_validation import FIXED, ADDITIONAL
from outdoor.canopy_depth_calibration import fit_rigid_depth_scale
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import render_hybrid


def reproject_depth_pixels(uv, depth, source, target):
    """Exact calibrated K and row-vector world-view transforms, CPU or GPU."""
    camera = torch.stack(((uv[:, 0]-source.cx)/source.focal_x*depth,
                          (uv[:, 1]-source.cy)/source.focal_y*depth, depth,
                          torch.ones_like(depth)), dim=1)
    world = camera @ source.world_view_transform.to(camera).inverse()
    other = world @ target.world_view_transform.to(camera)
    z = other[:, 2]
    safe = torch.where(z.abs() > 1e-8, z, torch.ones_like(z))
    pixels = torch.stack((target.focal_x*other[:, 0]/safe+target.cx,
                          target.focal_y*other[:, 1]/safe+target.cy), dim=1)
    return pixels, z


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    parser.set_defaults(data_device="cpu", resolution=640, white_background=True)
    parser.add_argument("--evidence-store", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric-scale", type=float, required=True)
    parser.add_argument("--samples-per-view", type=int, default=1024)
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--rigid-anchor-scale',action='store_true')
    args = parser.parse_args()
    if args.samples_per_view <= 0 or args.neighbors <= 0:
        raise ValueError("Sample and neighbor budgets must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    dataset = model.extract(args); dataset.model_path = str(args.output)
    views = LazyScene(dataset, GaussianModel(dataset.sh_degree), image_cache_size=4).getTrainCameras()
    geometry = OutdoorGeometryEvidence(args.evidence_store, chart_base_source="moge3_adaptive")
    geometry.configure_moge3_metric_scale(args.metric_scale)
    masks = CambridgeMaskLookup(Path(dataset.source_path), args.masks)
    teacher=load_hybrid_teacher(args.checkpoint,sh_degree=dataset.sh_degree) if args.checkpoint else None
    if args.rigid_anchor_scale and teacher is None:raise ValueError('Frozen rigid checkpoint required')
    canonical=teacher.state['training_contract']['static_scene_canonical_rgb']['canonical_sequence'] if teacher else 'seq2'
    selected = [i for i, v in enumerate(views) if v.image_name.startswith(canonical+'__')
                and Path(v.image_name).stem in geometry.moge3_records]
    cache = {}
    calibrations={}
    def fields(index):
        if index in cache: return cache[index]
        view = views[index]
        evidence = geometry.fields(view.image_name, device=torch.device("cpu"),
                                   shape=(view.image_height, view.image_width))
        _, bounds, valid, _, _ = _moge3_depth_query_bounds(evidence, global_log_scale_sigma=0.)
        obj, sky, distortion, tree = masks.get_index_masks(view.image_name, (0, 1, 2, 3),
            (view.image_height, view.image_width), torch.device("cpu"))
        keep = obj & sky & distortion & ~tree & valid
        depth = evidence["moge3_depth_m"][0]
        if args.rigid_anchor_scale:
            if index not in calibrations:
                package=render_hybrid(view,teacher.surface,teacher.foliage,background=torch.ones(3,device='cuda'),
                    include_dynamic=False,volume_gate=torch.zeros_like(teacher.foliage.opacity_logits.reshape(-1)),
                    optical_replacement_policy='disabled',structural_trainable_start=None)
                calibrations[index]=fit_rigid_depth_scale(depth,package.median_depth.cpu(),package.surface_alpha.cpu(),obj&sky&distortion&tree&valid)
                del package
            scale=calibrations[index]['scale'];depth=depth*scale;bounds=bounds*scale
        low = torch.where(keep, depth, torch.full_like(depth, float("inf")))
        high = torch.where(keep, depth, torch.full_like(depth, -float("inf")))
        result = (depth, keep, bounds, -F.max_pool2d(-low[None, None], 3, 1, 1)[0, 0],
                  F.max_pool2d(high[None, None], 3, 1, 1)[0, 0])
        # A bounded cache avoids retaining all dense multimodal evidence.
        if len(cache) >= 12: cache.pop(next(iter(cache)))
        cache[index] = result
        return result
    centers = {i:views[i].camera_center.detach().cpu() for i in selected}
    rows = []
    for source_index in FIXED + ADDITIONAL:
        if source_index not in centers: continue
        source = views[source_index]
        depth, keep, _, _, _ = fields(source_index)
        pixels = torch.nonzero(keep, as_tuple=False)
        if not len(pixels): continue
        take = torch.linspace(0, len(pixels)-1, min(len(pixels), args.samples_per_view)).long()
        pixels = pixels[take]; uv = pixels[:, [1, 0]].float()
        source_depth = depth[pixels[:, 0], pixels[:, 1]]
        neighbors = sorted((i for i in selected if i != source_index),
            key=lambda i:float((centers[i]-centers[source_index]).square().sum()))[:args.neighbors]
        for target_index in neighbors:
            target = views[target_index]
            target_depth, target_keep, bounds, low, high = fields(target_index)
            projected, z = reproject_depth_pixels(uv, source_depth, source, target)
            xy = projected.round().long()
            inside = (z > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < target.image_width) & (xy[:, 1] >= 0) & (xy[:, 1] < target.image_height)
            xy = xy[inside]; z = z[inside]
            accepted = target_keep[xy[:, 1], xy[:, 0]]
            xy = xy[accepted]; z = z[accepted]
            if not len(z): continue
            y, x = xy[:, 1], xy[:, 0]
            residual = z-target_depth[y, x]
            margin = (bounds[1, y, x]-bounds[0, y, x])/2
            row = {"source":source_index, "target":target_index, "samples":len(z),
                "baseline_world":float((centers[source_index]-centers[target_index]).norm()),
                "within_thin_interval":int(((z >= bounds[0,y,x]) & (z <= bounds[1,y,x])).sum()),
                "within_local_3x3_depth_range_plus_thin_margin":int(((z >= low[y,x]-margin) & (z <= high[y,x]+margin)).sum()),
                "in_front_of_target_interval":int((z < bounds[0,y,x]).sum()),
                "behind_target_interval":int((z > bounds[1,y,x]).sum()),
                "signed_depth_residual_quantiles":torch.quantile(residual, torch.tensor([.1,.5,.9])).tolist(),
                "absolute_depth_residual_quantiles":torch.quantile(residual.abs(), torch.tensor([.1,.5,.9])).tolist()}
            rows.append(row)
        print(json.dumps({"source":source_index,"pairs":len(rows)}), flush=True)
    total = sum(r["samples"] for r in rows)
    summary = {key:sum(r[key] for r in rows)/max(total,1) for key in
        ("within_thin_interval", "within_local_3x3_depth_range_plus_thin_margin",
         "in_front_of_target_interval", "behind_target_interval")}
    payload = {"pairs":rows, "samples":total, "summary":summary,
        "metric_scale":args.metric_scale, "resolution":args.resolution,
        "rigid_anchor_calibrations":calibrations,
        "caveat":"Same canonical sequence; different first-visible leaves and occlusion can disagree. This is not a ground-truth depth error measurement."}
    (args.output/"depth_consistency.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"samples":total,"summary":summary}), flush=True)


if __name__ == "__main__": main()
