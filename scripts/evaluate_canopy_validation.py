"""Read-only, prespecified multi-view native canopy validation.

Semantic/depth observations define reporting masks only, never render gates.
Surface contribution on canopy pixels is an attribution diagnostic: real
foliage gaps can legitimately reveal a background, so it is not a loss target.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "2d-gaussian-splatting")]
import torch
from PIL import Image
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.evaluate_hybrid_teacher import _tree_boundary_masks

FIXED = [558,632,633,634,663,671,677,682,690,707,736,743,756,768,776,909]
ADDITIONAL = [729,713,655,697,748,717,760,680,733,694,710,763,751,656,685,777,
              668,704,703,731,705,692,766,698,772,715,711,674,660,657,738,747]
DIAGNOSTIC = [408,409,410]


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def masked_metrics(prediction, target, mask):
    count = int(mask.sum())
    if not count:
        return {"pixels": 0, "squared_error_sum": 0., "psnr": None}
    error = float((prediction[:,mask]-target[:,mask]).square().sum())
    return {"pixels":count, "squared_error_sum":error,
            "psnr":float(-10*torch.log10(torch.tensor(max(error/(3*count),1e-12))))}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    parser.set_defaults(data_device="cpu", resolution=640, white_background=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--masks",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--allow-intrinsic-depth-query-repair",action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    dataset = model.extract(args); dataset.model_path = str(args.output)
    teacher = load_hybrid_teacher(args.checkpoint,sh_degree=dataset.sh_degree,
        allow_intrinsic_depth_query_repair=args.allow_intrinsic_depth_query_repair)
    views = LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=8).getTrainCameras()
    masks = CambridgeMaskLookup(Path(dataset.source_path),args.masks)
    # Geometry and opacity must remain bitwise identical for frozen-rigid arms.
    fingerprints = {name:tensor_digest(getattr(teacher.surface,name))
                    for name in ("_xyz","_scaling","_rotation","_opacity")}
    rows = []
    for index in FIXED+ADDITIONAL+DIAGNOSTIC:
        view = views[index]
        rendered = teacher.render(view,task=None,conditioned=False)
        prediction = rendered["rgb"]; target = view.original_image.cuda()
        obj,sky,distortion,tree = masks.get_index_masks(view.image_name,(0,1,2,3),
            (view.image_height,view.image_width),torch.device("cuda"))
        known = obj & sky & distortion; canopy = known & ~tree; rigid = known & tree
        inside,outside,_ = _tree_boundary_masks(~canopy)
        cohort = "fixed16" if index in FIXED else "additional32" if index in ADDITIONAL else "cross_sequence_diagnostic3"
        entry = {"index":index,"image":view.image_name,"cohort":cohort,"regions":{}}
        for name,mask in (("tree",canopy),("tree_interior",canopy & ~inside),
                          ("tree_boundary",canopy & inside),("rigid",rigid),("hard",rigid & outside)):
            region = masked_metrics(prediction,target,mask)
            for layer in ("surface_alpha","volume_alpha"):
                region[layer] = float(rendered[layer].reshape_as(mask)[mask].mean()) if region["pixels"] else None
            entry["regions"][name] = region
        pair = torch.cat((target,prediction),dim=2).clamp(0,1)
        Image.fromarray((pair.permute(1,2,0).cpu().numpy()*255).round().astype("uint8")).save(args.output/f"view_{index}.png")
        rows.append(entry)
        print(json.dumps({"index":index,"tree":entry["regions"]["tree"]["psnr"],
                          "rigid":entry["regions"]["rigid"]["psnr"]}),flush=True)
    summaries = {}
    for cohort in ("fixed16","additional32","canonical48","cross_sequence_diagnostic3"):
        selected = [r for r in rows if r["cohort"] == cohort or
                    (cohort == "canonical48" and r["cohort"] != "cross_sequence_diagnostic3")]
        summary = {}
        for name in ("tree","tree_interior","tree_boundary","rigid","hard"):
            regions = [r["regions"][name] for r in selected if r["regions"][name]["pixels"]]
            pixels = sum(r["pixels"] for r in regions)
            error = sum(r["squared_error_sum"] for r in regions)
            summary[name] = {"views":len(regions),"pixels":pixels,
                "mean_view_psnr":sum(r["psnr"] for r in regions)/len(regions) if regions else None,
                "pooled_psnr":float(-10*torch.log10(torch.tensor(max(error/(3*pixels),1e-12)))) if pixels else None,
                **{layer:sum(r[layer]*r["pixels"] for r in regions)/pixels if pixels else None
                   for layer in ("surface_alpha","volume_alpha")}}
        summaries[cohort] = summary
    payload = {"checkpoint":str(args.checkpoint.resolve()),"iteration":teacher.state.get("iteration"),
        "resolution":args.resolution,"render_contract":"native_task_none_conditioned_false_no_oracle_routing",
        "surface_fingerprints":fingerprints,"summaries":summaries,"per_view":rows,
        "implementation_validation":teacher.state.get("_render_implementation_validation"),
        "caveat":"Reconstruction views, not held-out RGB training. Layer alpha is attribution, not opacity ground truth."}
    (args.output/"metrics.json").write_text(json.dumps(payload,indent=2))
    print(json.dumps({"completed":True,"summaries":summaries}),flush=True)


if __name__ == "__main__":
    main()
