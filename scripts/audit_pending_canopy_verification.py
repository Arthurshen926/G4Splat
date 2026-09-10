"""Counterfactual real-render verification sweep on an in-memory checkpoint.

Never saves a model or changes the input checkpoint. Measures how much pending
ray-birth debt is waiting for visits rather than failing actual visibility.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/"2d-gaussian-splatting")]
import torch
from PIL import Image
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import (
    render_hybrid, static_detail_forward_visibility_gate,
    PROPOSAL_RAY_BIRTH, VERIFICATION_VERIFIED,
    VERIFICATION_MEASURED_SINGLE, PROPOSAL_NONE,
)
from outdoor.canopy_support_expansion import independent_canopy_camera_candidates, front_of_known_rigid_mask,canopy_material_audit_fields
from outdoor.training_evidence import OutdoorGeometryEvidence
from scripts.train_unified_outdoor_teacher import _moge3_depth_query_bounds
from outdoor.evidence_store import load_evidence_store, artifact_path
from outdoor.task_fields import OutdoorTaskFieldLookup
from scripts.train_unified_outdoor_teacher import _update_static_child_verification_from_render, sequence_id
from scripts.evaluate_canopy_validation import FIXED, ADDITIONAL, masked_metrics
from scripts.evaluate_hybrid_teacher import _tree_boundary_masks


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    parser.set_defaults(data_device="cpu",resolution=640,white_background=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--evidence-store",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gpu-memory-fraction",type=float,default=.25)
    parser.add_argument("--allow-intrinsic-depth-query-repair",action="store_true")
    parser.add_argument("--evaluate",action="store_true")
    parser.add_argument("--expand-measured-support",action="store_true",
        help="Independent exact-K/depth canopy candidates followed by real native verification; never direct promotion")
    parser.add_argument("--metric-scale",type=float)
    parser.add_argument('--initialization',type=Path)
    parser.add_argument("--target-only-validation",action="store_true")
    parser.add_argument("--save-foliage-capture",action="store_true",
        help="Save a clearly diagnostic, non-trainer-checkpoint foliage capture for isolated follow-up optimization")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    dataset=model.extract(args);dataset.model_path=str(args.output)
    teacher=load_hybrid_teacher(args.checkpoint,sh_degree=dataset.sh_degree,
        allow_intrinsic_depth_query_repair=args.allow_intrinsic_depth_query_repair)
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=8).getTrainCameras()
    store=load_evidence_store(args.evidence_store)
    manifest=Path(store["semantic_contract"]);semantic=json.loads(manifest.read_text())
    fields=OutdoorTaskFieldLookup(Path(dataset.source_path),Path(semantic["tree_mask_pickle"]),manifest,
        multiview_track_archive=artifact_path(store,"mast3r_multiview_tracks"),
        projected_rigid_posterior_archive=artifact_path(store,"projected_rigid_conflict_posterior",required=False),
        max_cached_views=0)
    foliage=teacher.foliage
    geometry=None
    if args.expand_measured_support:
        if args.metric_scale is None or args.initialization is None:
            raise ValueError("Support expansion requires checkpoint-bound initialization and metric scale")
        geometry=OutdoorGeometryEvidence(args.evidence_store,chart_base_source="moge3_adaptive")
        from scripts.seed_bound_moge_diagnostic import configure_seed_bound_moge
        calibration_audit=configure_seed_bound_moge(geometry,args.initialization,teacher.state['training_contract'],args.metric_scale)
        print(json.dumps({'diagnostic_calibration':calibration_audit}),flush=True)
    def evaluate(label):
        if not args.evaluate: return None
        results=[]
        for index in FIXED+ADDITIONAL:
            view=views[index];rendered=teacher.render(view,task=None,conditioned=False)
            target=view.original_image.cuda();prediction=rendered["rgb"]
            obj,sky,distortion,tree=fields.lookup.get_index_masks(view.image_name,(0,1,2,3),
                (view.image_height,view.image_width),torch.device("cuda"))
            known=obj & sky & distortion;canopy=known & ~tree;rigid=known & tree
            inside,outside,_=_tree_boundary_masks(~canopy)
            row={"index":index}
            for name,mask in (("tree",canopy),("rigid",rigid),("hard",rigid & outside),
                              ("interior",canopy & ~inside),("boundary",canopy & inside)):
                row[name]=masked_metrics(prediction,target,mask)["psnr"]
            row["tree_surface_alpha"]=float(rendered["surface_alpha"].reshape_as(canopy)[canopy].mean())
            results.append(row)
            if index in (632,690,707,713,768):
                pair=torch.cat((target,prediction),dim=2).clamp(0,1)
                Image.fromarray((pair.permute(1,2,0).cpu().numpy()*255).round().astype("uint8")).save(args.output/f"{label}_{index}.png")
        summary={key:sum(r[key] for r in results if r[key] is not None)/sum(r[key] is not None for r in results)
                 for key in ("tree","rigid","hard","interior","boundary","tree_surface_alpha")}
        return {"mean_view":summary,"per_view":results}
    before_metrics=evaluate("before")
    canonical=teacher.state["training_contract"]["static_scene_canonical_rgb"]["canonical_sequence"]
    names=sorted({sequence_id(v.image_name) for v in views})
    lookup=torch.full((max(int(v.colmap_id) for v in views)+1,),-1,dtype=torch.int16,device="cuda")
    for v in views: lookup[int(v.colmap_id)]=names.index(sequence_id(v.image_name))
    targets=foliage.proposal_kind==PROPOSAL_RAY_BIRTH
    if args.expand_measured_support:
        targets=foliage.static_leaf_mask & (foliage.verification_state==VERIFICATION_MEASURED_SINGLE) & (foliage.proposal_kind==PROPOSAL_NONE)
        source=foliage.verified_camera_ids
        source_valid=(source>=0) & (source<len(lookup))
        canonical_source=source_valid & (lookup[source.clamp(0,len(lookup)-1).long()]==names.index(canonical))
        targets &= canonical_source.any(dim=1)
    before_counts=foliage.verified_camera_count.clone()
    candidates=sorted({int(c) for c in foliage.support_camera_ids[targets].flatten().tolist() if c>=0})
    ordered=[v for v in views if int(v.colmap_id) in candidates and sequence_id(v.image_name)==canonical]
    if args.expand_measured_support:
        ordered=[v for v in views if sequence_id(v.image_name)==canonical
                 and Path(v.image_name).stem in geometry.moge3_records]
    records=[]
    for view in ordered:
        camera=int(view.colmap_id)
        rigid_depth=rigid_alpha=None
        added_rows=torch.empty(0,dtype=torch.long,device=foliage.xyz.device)
        added_slots=added_rows
        if args.expand_measured_support:
            existing=(foliage.support_camera_ids==camera).any(dim=1)
            free=(foliage.support_camera_ids<0)
            eligible=targets & (foliage.verification_state==VERIFICATION_MEASURED_SINGLE) & ~existing & free.any(dim=1)
            if not bool(eligible.any()): continue
            task=fields.fields(view.image_name,(view.image_height,view.image_width),torch.device("cuda"))
            evidence=geometry.fields(view.image_name,device=torch.device("cuda"),shape=(view.image_height,view.image_width))
            _,bounds,valid,_,_=_moge3_depth_query_bounds(evidence,global_log_scale_sigma=0.)
            obj,sky,distortion,tree=fields.lookup.get_index_masks(view.image_name,(0,1,2,3),
                (view.image_height,view.image_width),torch.device("cuda"))
            candidate_mask=obj & sky & distortion & ~tree & valid
            rigid_package=render_hybrid(view,teacher.surface,foliage,background=torch.ones(3,device="cuda"),
                include_dynamic=False,optical_replacement_policy="disabled",structural_trainable_start=None,
                volume_gate=torch.zeros_like(foliage.opacity_logits.reshape(-1)))
            added_rows=independent_canopy_camera_candidates(foliage.xyz,view,bounds,candidate_mask,eligible,
                rigid_depth=rigid_package.median_depth,rigid_alpha=rigid_package.surface_alpha)
            rigid_depth,rigid_alpha=rigid_package.median_depth,rigid_package.surface_alpha
            del rigid_package
            if not len(added_rows): continue
            added_slots=free[added_rows].to(torch.int8).argmax(dim=1)
            # Temporary candidate exposure, not a verified support assignment.
            foliage.support_camera_ids[added_rows,added_slots]=camera
        exact=(foliage.support_camera_ids==camera).any(dim=1)
        witnessed=(foliage.verified_camera_ids==camera).any(dim=1)
        pending=targets & (foliage.verification_state!=VERIFICATION_VERIFIED)
        missing=pending & exact & ~witnessed
        if not bool(missing.any()): continue
        task=fields.fields(view.image_name,(view.image_height,view.image_width),torch.device("cuda"))
        package=render_hybrid(view,teacher.surface,foliage,background=torch.ones(3,device="cuda"),
            include_dynamic=False,optical_replacement_policy="disabled",structural_trainable_start=None,
            volume_gate=static_detail_forward_visibility_gate(foliage,camera,include_pending_exact=True),
            audit_fields=canopy_material_audit_fields(task["p_canopy_core"],task["p_rigid"],task["p_sky"],view.original_image.cuda()))
        if args.target_only_validation:
            package.responsibility[package.structural_count:][~targets]=0
        if rigid_depth is None:
            rigid_package=render_hybrid(view,teacher.surface,foliage,background=torch.ones(3,device="cuda"),
                include_dynamic=False,optical_replacement_policy="disabled",structural_trainable_start=None,
                volume_gate=torch.zeros_like(foliage.opacity_logits.reshape(-1)))
            rigid_depth,rigid_alpha=rigid_package.median_depth,rigid_package.surface_alpha
            del rigid_package
        audit=_update_static_child_verification_from_render(foliage,package,camera_id=camera,
            canopy_color_audit=True,sky_audit_column=7,
            camera_sequence_lookup=lookup,required_sequence_count=1,
            candidate_geometry_gate=front_of_known_rigid_mask(foliage.xyz,view,rigid_depth,rigid_alpha))
        if len(added_rows):
            real_witness=(foliage.verified_camera_ids[added_rows]==camera).any(dim=1)
            # No native witness means no retained new support or global permission.
            foliage.support_camera_ids[added_rows[~real_witness],added_slots[~real_witness]]=-1
            audit["independent_depth_candidates"]=int(len(added_rows))
            audit["independent_depth_candidates_with_real_witness"]=int(real_witness.sum())
        record={"camera_id":camera,"image":view.image_name,"missing_ray_birth_witnesses":int(missing.sum()),
                "ray_births_now_verified":int((targets & (foliage.verification_state==VERIFICATION_VERIFIED)).sum()),
                **audit}
        records.append(record);print(json.dumps(record),flush=True)
        del package
    result={"checkpoint":str(args.checkpoint),"iteration":teacher.state["iteration"],
        "target_population":"measured_single" if args.expand_measured_support else "pending_ray_birth",
        "initial_target_rows":int(targets.sum()),
        "newly_verified_target_rows":int((targets & (foliage.verification_state==VERIFICATION_VERIFIED)).sum()),
        "scope":"diagnostic_real_mixed_render_witnesses__input_unchanged__not_a_trainer_checkpoint",
        "target_only_validation":args.target_only_validation,
        "initial_pending_ray_births":int(targets.sum()),"candidate_cameras":len(ordered),
        "executed_verification_cameras":len(records),
        "initial_zero_witness":int((targets & (before_counts==0)).sum()),
        "final_zero_witness":int((targets & (foliage.verified_camera_count==0)).sum()),
        "newly_verified_ray_births":int((targets & (foliage.verification_state==VERIFICATION_VERIFIED)).sum()),
        "before_metrics":before_metrics,"after_metrics":evaluate("after"),
        "records":records}
    (args.output/"verification_audit.json").write_text(json.dumps(result,indent=2))
    if args.save_foliage_capture:
        torch.save({"diagnostic_only":True,"source_checkpoint":str(args.checkpoint.resolve()),
            "foliage":foliage.capture(),"verification_audit":result},args.output/"diagnostic_foliage_capture.pth")
    print(json.dumps({k:v for k,v in result.items() if k!="records"}),flush=True)


if __name__=="__main__":
    main()
