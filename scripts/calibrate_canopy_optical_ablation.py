"""Isolated canonical foliage ablations; frozen rigid scene, immutable input."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "2d-gaussian-splatting")]
import torch
import torch.nn.functional as F
from PIL import Image
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import render_hybrid, persistent_static_evidence_mask, VolumetricFoliageModel
from outdoor.training_evidence import OutdoorGeometryEvidence
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.train_unified_outdoor_teacher import _moge3_depth_query_bounds, _moge3_canopy_hit_owner_mask
from scripts.train_unified_outdoor_teacher import _static_volume_native_color_inputs,_static_volume_intrinsic_color_inputs,_photo_loss
from scripts.train_unified_outdoor_teacher import _static_detail_canonical_ownership_gate
from scripts.evaluate_hybrid_teacher import _tree_boundary_masks
from outdoor.static_ray_birth import StaticRayBirthAccumulator
from scripts.train_unified_outdoor_teacher import _moge3_uncovered_hit_proposals
from scripts.evaluate_canopy_validation import FIXED, ADDITIONAL, tensor_digest
from outdoor.canopy_support_expansion import clip_canopy_depth_queries_before_rigid
from outdoor.canopy_detail_loss import masked_canopy_ssim_loss,canopy_owned_color_gradient
from outdoor.canopy_foreground_evidence import foreground_darkening_bound,foreground_darkening_loss


@torch.no_grad()
def source_ray_directions(positions,camera_ids,observation_uv,views):
    """Keep genuine source pixel bearings; never snap an off-ray checkpoint."""
    directions=torch.empty_like(positions)
    by_id={int(view.colmap_id):view for view in views}
    for camera in torch.unique(camera_ids).tolist():
        if int(camera) not in by_id:raise ValueError('Missing source camera')
        view=by_id[int(camera)];rows=camera_ids==camera
        xyz=positions[rows]@view.world_view_transform[:3,:3]+view.world_view_transform[3,:3]
        uv=torch.stack((view.focal_x*xyz[:,0]/xyz[:,2]+view.cx,view.focal_y*xyz[:,1]/xyz[:,2]+view.cy),dim=1)
        observed=observation_uv[rows]*positions.new_tensor([view.image_width,view.image_height])-.5
        if not torch.isfinite(observed).all() or not (xyz[:,2]>0).all() or not ((uv-observed).abs()<=.05).all():
            raise ValueError('Source-ray optimization requires an already aligned bearing, not displaced geometry')
        ray=positions[rows]-view.camera_center
        directions[rows]=ray/ray.norm(dim=1,keepdim=True).clamp_min(1.e-12)
    return directions


@torch.no_grad()
def apply_source_ray_update_(positions,initial,directions,displacement,previous,owner,radius):
    """A bounded scalar depth update cannot drift across its observed pixel."""
    displacement.copy_(torch.maximum(-radius,torch.minimum(displacement,radius)))
    displacement[~owner]=previous[~owner]
    positions[owner]=initial[owner]+directions[owner]*displacement[owner,None]


@torch.no_grad()
def project_leaf_center_updates_(positions,initial,previous,owner,radius):
    """Bound diagnostic motion without changing inactive rows or rigid state."""
    if not torch.isfinite(radius).all() or not (radius>0).all():
        raise ValueError('Finite positive per-point motion radii required')
    delta=positions-initial
    distance=delta.norm(dim=1)
    clipped=owner & (distance>radius)
    positions[clipped]=initial[clipped]+delta[clipped]*(radius[clipped]/distance[clipped])[:,None]
    positions[~owner]=previous[~owner]
    return int(clipped.sum())


@torch.no_grad()
def project_leaf_shape_updates_(scales,initial,previous,owner):
    """Local tangent refinement only: never increase the thin depth extent."""
    lower=initial+math.log(.25)
    upper=initial+math.log(1.5)
    upper[:,2]=initial[:,2]
    scales.copy_(torch.maximum(lower,torch.minimum(scales,upper)))
    scales[~owner]=previous[~owner]


@torch.no_grad()
def project_leaf_color_updates_(features,previous,owner,coefficient_count):
    """Bound learned SH bandwidth and apply rest/DC actual-step ratio."""
    if coefficient_count<1 or coefficient_count>features.shape[1]:
        raise ValueError('Valid trainable SH coefficient count required')
    features[:,1:coefficient_count]=previous[:,1:coefficient_count]+.05*(features[:,1:coefficient_count]-previous[:,1:coefficient_count])
    features[:,coefficient_count:]=previous[:,coefficient_count:]
    features[~owner]=previous[~owner]


@torch.no_grad()
def project_leaf_rotation_updates_(quaternions,initial,previous,owner,maximum_angle=math.pi/4):
    """Optimize thin-leaf orientation, not thickness; bound physical rotation."""
    if not 0<maximum_angle<math.pi or not torch.isfinite(quaternions).all():
        raise ValueError('Finite quaternions and rotation bound in (0,pi) required')
    reference=F.normalize(initial,dim=1)
    norm=quaternions.norm(dim=1,keepdim=True)
    proposed=torch.where(norm>1.e-8,quaternions/norm.clamp_min(1.e-8),reference)
    dot=(proposed*reference).sum(1)
    proposed=torch.where((dot<0)[:,None],-proposed,proposed);dot=dot.abs().clamp(0,1)
    limit=math.cos(maximum_angle/2)
    clipped=dot<limit
    tangent=F.normalize(proposed[clipped]-dot[clipped,None]*reference[clipped],dim=1)
    proposed[clipped]=reference[clipped]*limit+tangent*math.sin(maximum_angle/2)
    quaternions.copy_(proposed);quaternions[~owner]=previous[~owner]


def uncertainty_hit_bounds(pre_bounds, thin_bounds, rigid_depth, rigid_alpha):
    """Diagnostic epistemic envelope, not permission to fill a thick layer.

    Use the existing conservative lower bound and its log-symmetric upper
    bound, clipped before independently rendered opaque rigid geometry.
    This changes neither geometry nor deployment rendering.
    """
    center = pre_bounds[1]
    lower = pre_bounds[0]
    upper = center.square()/lower.clamp_min(1e-6)
    depth = rigid_depth.reshape_as(center)
    known = (rigid_alpha.reshape_as(center) >= .95) & torch.isfinite(depth) & (depth > 0)
    clearance = torch.maximum(torch.full_like(depth,.03),.01*depth)
    upper = torch.where(known, torch.minimum(upper,depth-clearance), upper)
    valid = torch.isfinite(thin_bounds).all(dim=0) & (upper > lower)
    nan = torch.full_like(center,float("nan"))
    return torch.stack((torch.where(valid,lower,nan),torch.where(valid,upper,nan)))


def _validate_joint_checkpoint_refinement(args):
    if args.refine_checkpoint_leaves and (
        args.depth_profile_initialization is None or args.replacement_foliage is not None
        or args.evaluate_opacity is not None or args.position_mode!='free'
    ):
        raise ValueError('Checkpoint refinement requires matched calibration, free local coordinates, and no replacement/replay')
    if (not math.isfinite(args.native_rigid_alpha_weight) or args.native_rigid_alpha_weight<0
        or (args.native_rigid_alpha_weight>0 and (args.production_geometry_objective!='none'
            or args.production_color_objective!='none'))):
        raise ValueError('Joint native rigid extinction requires a finite nonnegative weight and joint objective')


@torch.no_grad()
def _restore_parameter_update_authority_(parameter, previous, owner, state, *, high_order_owner=None):
    parameter[~owner]=previous[~owner]
    if high_order_owner is not None:parameter[~high_order_owner,1:]=previous[~high_order_owner,1:]
    for key in ('exp_avg','exp_avg_sq'):
        if key in state:
            state[key][~owner]=0
            if high_order_owner is not None:state[key][~high_order_owner,1:]=0


def main():
    p = argparse.ArgumentParser()
    model = ModelParams(p)
    p.set_defaults(resolution=640)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--evidence-store", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metric-scale", type=float, required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument('--position-mode',choices=('free','source-ray','source-group'),default='free')
    p.add_argument('--canopy-ssim-weight',type=float,default=0.)
    p.add_argument('--adam-eps',type=float,default=1.e-15,
        help='Match the production volume optimizer; 1e-8 is retained only as an explicit diagnostic control')
    p.add_argument('--lr-final-factor',type=float,default=1.,
        help='Explicit exponential annealing for this diagnostic; 1 preserves constant-LR controls')
    p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--skip-auxiliary-thin-metrics',action='store_true',
        help='Skip the extra uncalibrated thin-depth diagnostic, never native RGB/alpha or the fixed cohort')
    p.add_argument('--allow-rigid-leaf-color-gradients',action='store_true',
        help='Legacy diagnostic control only: allow foliage colors to camouflage rigid/sky ownership errors')
    p.add_argument("--lr", type=float, default=.05)
    p.add_argument("--position-lr", type=float, default=0.,
        help="Explicit geometry ablation: bounded persistent-leaf center refinement; rigid geometry stays frozen")
    p.add_argument("--position-prior-weight", type=float, default=.001)
    p.add_argument("--appearance-lr", type=float, default=0.,
        help="Fresh diagnostic only: optimize leaf color; DC-only unless a directional degree is explicit")
    p.add_argument("--shape-lr", type=float, default=0.,
        help="Fresh diagnostic only: bounded local scales; depth thickness cannot increase")
    p.add_argument('--rotation-lr',type=float,default=0.,help='Explicit bounded thin-leaf orientation ablation')
    p.add_argument('--appearance-sh-degree',type=int,default=0,
        help='Explicit low-order directional leaf appearance ablation; default remains DC-only')
    p.add_argument('--foreground-darkening-weight',type=float,default=0.,
        help='Conditional wall-front extinction lower bound with calibrated background-error slack')
    p.add_argument('--save-foliage-capture',action='store_true')
    p.add_argument('--production-color-objective',choices=('none','native','legacy_intrinsic'),default='none',
        help='Fixed geometry/opacity causal audit of the production isolated-color target')
    p.add_argument('--color-owner-scope',choices=('persistent','exact'),default='persistent',
        help='Pure color ablation: all canonical native-visible persistent leaves versus persisted camera IDs')
    p.add_argument('--opacity-owner-scope',choices=('persistent','exact'),default='persistent',
        help='Opacity-only gradient permission ablation; native and query forward contributors stay unchanged')
    p.add_argument('--refine-checkpoint-leaves',action='store_true',
        help='Explicit nonresumable refinement of checkpoint leaves; requires matched seed depth profiles')
    p.add_argument('--persistent-checkpoint-geometry',action='store_true',
        help='Match production: measured-single rows retain exact-camera DC/opacity, not geometry/high-order SH')
    p.add_argument('--confirmed-occluder-evidence',type=Path,
        help='Complete immutable nonvalidation multi-camera depth/semantic certificate')
    p.add_argument('--confirmed-occluder-weight',type=float,default=0.)
    p.add_argument('--occluder-evidence-scope',choices=['confirmed','source_candidate'],default='confirmed',
        help='Explicit weaker source-only RGB/depth prior ablation; not multiview-certified opacity')
    p.add_argument('--native-rigid-alpha-weight',type=float,default=0.,
        help='Native rigid extinction for jointly optimized leaf geometry/opacity, never a visibility mask')
    p.add_argument('--depth-profile-initialization',type=Path,
        help='Checkpoint-matched fresh seed providing immutable canopy depth calibration, not replacement geometry')
    p.add_argument('--production-geometry-objective',choices=('none','native','legacy_owner_subset'),default='none',
        help='Pure geometry RGB fixed-owner ablation; forward visibility is the only paired difference')
    p.add_argument('--geometry-owner-scope',choices=('exact','persistent'),default='exact',
        help='Isolated geometry diagnostic only: existing verified leaves over the canonical acquisition')
    p.add_argument('--geometry-rigid-alpha-weight',type=float,default=0.,
        help='Native visible foliage extinction penalty on eroded observed rigid pixels; geometry only')
    p.add_argument("--include-envelope", action="store_true")
    p.add_argument("--envelope-scope", choices=("persistent", "exact", "resolved"), default="persistent",
                   help="Diagnostic-only causal comparison, not production ownership authority")
    p.add_argument("--negative-weight", type=float, default=1.)
    p.add_argument("--optical-weight", type=float, default=1.)
    p.add_argument("--rgb-weight", type=float, default=0.)
    p.add_argument("--evaluate-opacity", type=Path)
    p.add_argument("--verified-foliage", type=Path,
        help="Explicit non-trainable-checkpoint verification diagnostic capture from the same source model")
    p.add_argument("--replacement-foliage", type=Path,
        help="Explicit fresh-geometry diagnostic, not a verification-only or production checkpoint")
    p.add_argument("--require-thin-growth", action="store_true")
    p.add_argument("--rigid-preservation-weight", type=float, default=0.)
    p.add_argument("--rigid-boundary-preservation-weight", type=float, default=0.)
    p.add_argument("--known-sky-weight", type=float, default=0.,
        help="Observed unoccluded sky is negative foliage evidence; no semantic gate in native rendering")
    p.add_argument("--independent-rigid-depth-ordering", action="store_true")
    p.add_argument('--rigid-anchor-scale',action='store_true')
    p.add_argument("--birth-audit", action="store_true")
    p.add_argument("--birth-proposals", type=int, default=256)
    p.add_argument("--canopy-sampling", action="store_true")
    p.add_argument("--global-log-scale-sigma", type=float, default=.10099024234861136)
    p.add_argument("--hit-interval-scope", choices=("thin", "scale_uncertainty"), default="thin",
                   help="Diagnostic only: test narrow-depth incompatibility without changing native render or geometry")
    p.add_argument("--gpu-memory-fraction", type=float, default=.30)
    p.add_argument("--validation-cohort", choices=("fixed16","canonical48"), default="fixed16")
    p.add_argument("--save-all-views", action="store_true")
    args = p.parse_args()
    if not math.isfinite(args.adam_eps) or args.adam_eps<=0:
        raise ValueError('Adam epsilon must be finite and positive')
    if not math.isfinite(args.lr_final_factor) or not 0<args.lr_final_factor<=1:
        raise ValueError('Final LR multiplier must be in (0,1]')
    if not math.isfinite(args.canopy_ssim_weight) or args.canopy_ssim_weight<0:
        raise ValueError('Canopy SSIM weight must be finite and nonnegative')
    if args.eval_every<=0:raise ValueError('Evaluation interval must be positive')
    if args.appearance_sh_degree<0:raise ValueError('Nonnegative appearance degree required')
    if not math.isfinite(args.foreground_darkening_weight) or args.foreground_darkening_weight<0:
        raise ValueError('Finite nonnegative darkening weight required')
    if args.foreground_darkening_weight>0 and not args.independent_rigid_depth_ordering:
        raise ValueError('Darkening bound requires independent rigid depth ordering')
    if args.production_color_objective!='none' and (args.lr!=0 or args.position_lr!=0 or args.shape_lr!=0 or args.rotation_lr!=0 or args.appearance_lr<=0 or args.allow_rigid_leaf_color_gradients):
        raise ValueError('Production color-target audit requires fixed geometry/opacity and canopy-owned appearance')
    if args.color_owner_scope!='persistent' and args.production_color_objective=='none':
        raise ValueError('Exact-color ownership ablation must not change geometry/optical training')
    if args.opacity_owner_scope != 'persistent' and any(
        v != 0 for v in (args.position_lr, args.appearance_lr, args.shape_lr, args.rotation_lr)
    ):
        raise ValueError('Exact opacity-owner diagnostic requires fixed geometry and appearance')
    if args.position_mode=='source-group' and (
        args.production_geometry_objective!='native' or args.geometry_owner_scope!='persistent'
        or args.shape_lr!=0 or args.rotation_lr!=0
    ):
        raise ValueError('Coherent source-depth correction is a persistent native pure-geometry diagnostic')
    if args.production_geometry_objective!='none' and (
            args.position_lr<=0 or args.lr!=0 or args.appearance_lr!=0
            or args.production_color_objective!='none' or args.include_envelope
            or any(v!=0 for v in (args.optical_weight,args.negative_weight,args.rgb_weight,
                args.canopy_ssim_weight,args.known_sky_weight,args.foreground_darkening_weight,
                args.rigid_preservation_weight,args.rigid_boundary_preservation_weight))):
        raise ValueError('Geometry forward comparison requires isolated exact-owner geometry RGB')
    if args.geometry_owner_scope!='exact' and args.production_geometry_objective!='native':
        raise ValueError('Persistent geometry scope requires the isolated native geometry objective')
    if (not math.isfinite(args.geometry_rigid_alpha_weight) or args.geometry_rigid_alpha_weight<0
            or (args.geometry_rigid_alpha_weight>0 and args.production_geometry_objective!='native')):
        raise ValueError('Geometry rigid alpha penalty requires native geometry objective and nonnegative weight')
    _validate_joint_checkpoint_refinement(args)
    if args.persistent_checkpoint_geometry and not args.refine_checkpoint_leaves:
        raise ValueError('Persistent checkpoint geometry requires explicit checkpoint refinement')
    if (not math.isfinite(args.confirmed_occluder_weight) or args.confirmed_occluder_weight<0
        or (args.confirmed_occluder_weight>0 and (args.confirmed_occluder_evidence is None
            or not args.persistent_checkpoint_geometry))):
        raise ValueError('Confirmed occlusion requires complete evidence and persistent checkpoint geometry')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = model.extract(args); dataset.model_path = str(args.output)
    teacher = load_hybrid_teacher(args.checkpoint, sh_degree=dataset.sh_degree,
                                 allow_intrinsic_depth_query_repair=True)
    views = LazyScene(dataset, GaussianModel(dataset.sh_degree), image_cache_size=8).getTrainCameras()
    masks = CambridgeMaskLookup(Path(dataset.source_path), args.masks)
    geometry = OutdoorGeometryEvidence(args.evidence_store, chart_base_source="moge3_adaptive")
    geometry.configure_moge3_metric_scale(args.metric_scale)
    holdout = FIXED + (ADDITIONAL if args.validation_cohort=="canonical48" else [])
    canonical=teacher.state['training_contract']['static_scene_canonical_rgb']['canonical_sequence']
    train = [i for i,v in enumerate(views) if i not in holdout
             and v.image_name.startswith(canonical+"__") and Path(v.image_name).stem in geometry.moge3_records]
    generator = torch.Generator().manual_seed(1701)
    order = [train[i] for i in torch.randperm(len(train), generator=generator).tolist()]
    confirmed_cache={}
    confirmed_manifest=None
    if args.confirmed_occluder_evidence is not None:
        from outdoor.canopy_occluder_loss import validate_consensus_coverage
        from scripts.train_unified_outdoor_teacher import _file_sha256
        confirmed_manifest=json.loads((args.confirmed_occluder_evidence/'audit.json').read_text())
        validate_consensus_coverage(confirmed_manifest,train,holdout)
        if (Path(confirmed_manifest['source_checkpoint']).resolve()!=args.checkpoint.resolve()
            or confirmed_manifest['source_checkpoint_sha256']!=_file_sha256(args.checkpoint)
            or confirmed_manifest['foliage_seed_sha256']!=teacher.state['training_contract']['foliage_seed_sha256']):
            raise ValueError('Confirmed occluder evidence differs from the input checkpoint/calibration')
    def confirmed_pixels(index,canopy):
        if confirmed_manifest is None or index not in confirmed_manifest['canopy_training_views']:
            return torch.zeros_like(canopy)
        if index not in confirmed_cache:
            evidence=torch.load(args.confirmed_occluder_evidence/f'evidence_{index}.pth',map_location='cpu')
            from outdoor.canopy_occluder_loss import select_occluder_evidence
            mask=select_occluder_evidence(evidence,args.occluder_evidence_scope)
            if mask.shape!=canopy.shape:
                raise ValueError('Invalid confirmed-occluder certificate')
            confirmed_cache[index]=mask
        mask=confirmed_cache[index].to(canopy.device)
        if bool((mask & ~canopy).any()):raise ValueError('Occluder certificate extends outside current observed canopy')
        return mask
    foliage = teacher.foliage
    reference_foliage = foliage
    depth_calibrations={}
    if args.depth_profile_initialization is not None:
        from scripts.train_unified_outdoor_teacher import _file_sha256
        manifest_path=args.depth_profile_initialization/'initialization_manifest.json'
        contract=teacher.state['training_contract']
        if _file_sha256(manifest_path)!=contract.get('initialization_manifest_sha256'):
            raise ValueError('Depth profile initialization differs from the checkpoint')
        manifest=json.loads(manifest_path.read_text())
        seed_path=Path(manifest['foliage_seed'])
        if _file_sha256(seed_path)!=contract.get('foliage_seed_sha256'):
            raise ValueError('Depth profile seed content differs from the checkpoint')
        seed=torch.load(seed_path,map_location='cpu')
        profiles=seed['audit']['fresh_canonical_leaves']['depth_profiles_by_image']
        if args.replacement_foliage is not None or not args.rigid_anchor_scale:
            raise ValueError('Matched seed profiles require rigid-anchor scale and no geometry replacement')
        depth_calibrations={i:{'scale':profiles[Path(v.image_name).stem]}
                            for i,v in enumerate(views) if Path(v.image_name).stem in profiles}
        del seed
    if args.replacement_foliage is not None:
        if args.verified_foliage is not None:
            raise ValueError("Fresh geometry and verification-only ablations are distinct experiments")
        replacement=torch.load(args.replacement_foliage,map_location="cpu")
        if (not replacement.get("diagnostic_only")
            or Path(replacement["source_checkpoint"]).resolve()!=args.checkpoint.resolve()
            or not replacement.get("scope","").startswith("fresh_dense_canonical_canopy_only__")):
            raise ValueError("Fresh foliage must be an explicitly matched diagnostic capture")
        capture=replacement["foliage"]
        if bool(replacement.get('rigid_anchor_scale',False))!=args.rigid_anchor_scale:
            raise ValueError('Fresh foliage and optical evidence need the same depth calibration')
        depth_calibrations=replacement.get('rigid_anchor_calibrations',{})
        # A fresh static diagnostic is a different representation, not a
        # resume into the old model's dynamic basis schema.
        foliage=VolumetricFoliageModel(dataset.sh_degree,
            dynamic_rank=int(capture["dynamic_rank"]),device="cuda")
        foliage.restore(capture)
        if bool(foliage.dynamic_leaf_mask.any()):
            raise ValueError("Fresh canonical diagnostic cannot contain dynamic leaves")
        teacher.foliage=foliage
    if args.verified_foliage is not None:
        verified=torch.load(args.verified_foliage,map_location="cpu")
        if not verified.get("diagnostic_only") or Path(verified["source_checkpoint"]).resolve()!=args.checkpoint.resolve():
            raise ValueError("Verified foliage diagnostic does not belong to this source")
        if not torch.equal(verified["foliage"]["xyz"],foliage.xyz.detach().cpu()):
            raise ValueError("Verification diagnostic unexpectedly changed geometry")
        if not torch.equal(verified["foliage"]["opacity_logits"],foliage.opacity_logits.detach().cpu()):
            raise ValueError("Verification diagnostic unexpectedly changed opacity")
        foliage.restore(verified["foliage"])
    # Start frozen, then enable only the explicitly requested leaf variables.
    # Rigid geometry/appearance and sky never participate in these optimizers.
    for parameter in foliage.parameters():
        parameter.requires_grad_(False)
    for name in ("_xyz","_scaling","_rotation","_opacity","_features_dc","_features_rest"):
        getattr(teacher.surface,name).requires_grad_(False)
    for module in (teacher.appearance,teacher.sky):
        if hasattr(module,"parameters"):
            for parameter in module.parameters(): parameter.requires_grad_(False)
    foliage.opacity_logits.requires_grad_(True)
    groups=[{'params':[foliage.opacity_logits],'lr':args.lr}]
    initial_positions=foliage.xyz.detach().clone()
    position_radius=(.1*foliage.observation_depth[:,0].detach()).nan_to_num(.1).clamp(.03,2.)
    ray_displacement=None
    source_group_factors=None
    if args.position_lr<0: raise ValueError('Position learning rate must be nonnegative')
    if args.position_lr>0:
        if (args.replacement_foliage is None and not args.refine_checkpoint_leaves) or args.evaluate_opacity is not None:
            raise ValueError('Geometry ablation requires an explicit fresh diagnostic; opacity-only replay is not valid')
        foliage.xyz.requires_grad_(True)
        if args.position_mode=='source-ray':
            ray_directions=source_ray_directions(initial_positions,foliage.observation_camera_ids[:,0],foliage.observation_uv[:,0],views)
            ray_displacement=torch.nn.Parameter(torch.zeros(len(foliage),device=foliage.xyz.device))
            groups.append({'params':[ray_displacement],'lr':args.position_lr})
        elif args.position_mode=='source-group':
            from outdoor.canopy_source_depth_transform import source_depth_transform, source_depth_chain_gradient
            source_ray_directions(initial_positions,foliage.observation_camera_ids[:,0],foliage.observation_uv[:,0],views)
            source_ids,source_group_index=torch.unique(foliage.observation_camera_ids[:,0].long(),sorted=True,return_inverse=True)
            by_id={int(v.colmap_id):v for v in views}
            source_origins=torch.stack([by_id[int(i)].camera_center.to(initial_positions) for i in source_ids])
            source_group_factors=torch.nn.Parameter(initial_positions.new_zeros(len(source_ids)))
            foliage.log_scales.requires_grad_(True)
            groups.append({'params':[source_group_factors],'lr':args.position_lr})
        else:
            groups.append({'params':[foliage.xyz],'lr':args.position_lr})
    extra_parameters=[]
    if args.appearance_sh_degree>min(foliage.sh_degree,teacher.surface.active_sh_degree):
        raise ValueError('Requested leaf appearance degree exceeds native renderer degree')
    trainable_color_coefficients=(args.appearance_sh_degree+1)**2
    initial_scales=foliage.log_scales.detach().clone()
    initial_rotations=foliage.quaternions.detach().clone()
    for parameter,rate in ((foliage.features,args.appearance_lr),(foliage.log_scales,args.shape_lr),(foliage.quaternions,args.rotation_lr)):
        if rate<0:raise ValueError('Leaf refinement learning rates must be nonnegative')
        if rate>0:
            if (args.replacement_foliage is None and not args.refine_checkpoint_leaves) or args.evaluate_opacity is not None:
                raise ValueError('Leaf refinement requires a fresh diagnostic; opacity replay omits learned state')
            parameter.requires_grad_(True)
            groups.append({'params':[parameter],'lr':rate})
            extra_parameters.append(parameter)
    full_refinement=args.position_lr>0 or bool(extra_parameters)
    replay_base={name:tensor_digest(getattr(foliage,name)) for name in ('xyz','log_scales','quaternions','features')}
    optimizer = torch.optim.Adam(groups,eps=args.adam_eps)
    initial_learning_rates=[group['lr'] for group in optimizer.param_groups]
    baseline = foliage.opacity_logits.detach().clone()
    original_opacity = baseline.clone()
    rigid_reference = {}
    rigid_depth_reference = {}
    rigid_background_reference = {}
    results = []

    def fields(view):
        obj, sky, distortion, tree = masks.get_index_masks(
            view.image_name, (0,1,2,3), (view.image_height,view.image_width), torch.device("cuda"))
        known = obj & sky & distortion
        return known & ~tree, known & tree

    def sky_field(view):
        obj,sky,distortion=masks.get_index_masks(view.image_name,(0,1,2),
            (view.image_height,view.image_width),torch.device('cuda'))
        return obj & distortion & ~sky

    @torch.no_grad()
    def evaluate(step):
        rows = []
        for index in holdout:
            view = views[index]; canopy, rigid = fields(view)
            rendered = teacher.render(view, task=None, conditioned=False)
            # Native deployment, without oracle mask/depth routing.
            prediction = rendered["rgb"]
            target = view.original_image.cuda()
            entry = {"index":index, "native_tree_volume_alpha":float(rendered["volume_alpha"].reshape_as(canopy)[canopy].mean())}
            entry['auxiliary_thin_depth_scope']='skipped' if args.skip_auxiliary_thin_metrics else 'original_global_moge_scale__not_per_view_calibrated'
            if not args.skip_auxiliary_thin_metrics:
                evidence = geometry.fields(view.image_name, device=torch.device("cuda"), shape=canopy.shape)
                _,bounds,valid,_,_ = _moge3_depth_query_bounds(evidence,global_log_scale_sigma=args.global_log_scale_sigma)
                query = render_hybrid(view,teacher.surface,foliage,
                    background=torch.zeros(3,device="cuda"),include_dynamic=False,
                    surface_gate=torch.zeros_like(teacher.surface.get_opacity.reshape(-1)),
                    volume_gate=(~foliage.dynamic_leaf_mask).float(),volume_depth_query_bounds=bounds,
                    optical_replacement_policy="disabled",structural_trainable_start=None)
                thin = query.volume_hit_interval_alpha.reshape_as(canopy)[canopy & valid]
                entry["all_static_thin_alpha"] = float(thin.mean()) if len(thin) else None
                entry["all_static_thin_zero_fraction"] = float((thin<=0).float().mean()) if len(thin) else None
                del query
            inside,outside,_ = _tree_boundary_masks(~canopy)
            for name,mask in [("tree",canopy),("rigid",rigid),
                              ("tree_interior",canopy & ~inside),
                              ("tree_boundary",canopy & inside),
                              ("hard",rigid & outside)]:
                entry[name] = float(-10 * torch.log10((prediction[:,mask]-target[:,mask]).square().mean().clamp_min(1e-12)))
            if args.save_all_views or index in (632,690,707,713,717,768):
                pair = torch.cat((target,prediction),dim=2).clamp(0,1)
                Image.fromarray((pair.permute(1,2,0).cpu().numpy()*255).round().astype("uint8")).save(args.output/f"view_{index}_{step:04d}.png")
            rows.append(entry)
        summary = {"step":step,"per_view":rows,
                   "tree":sum(x["tree"] for x in rows)/len(rows),
                   "rigid":sum(x["rigid"] for x in rows)/len(rows)}
        results.append(summary)
        (args.output / "metrics.json").write_text(json.dumps({"args":vars(args),"train_view_count":len(train),
            "sampled_view_indices":order,
            "implementation_validation":teacher.state.get("_render_implementation_validation"),
            "scope":("diagnostic_bounded_local_leaf_refinement" if full_refinement else "diagnostic_opacity_only")+"__holdout_excluded_from_this_calibration_not_original_training",
            "results":results}, default=str, indent=2))
        print(json.dumps(summary), flush=True)

    if args.birth_audit:
        accumulators = {
            "coarse":StaticRayBirthAccumulator(voxel_size=.15,visual_hull_voxel_size=.30,
                maximum_proposals_per_update=args.birth_proposals),
            "fine":StaticRayBirthAccumulator(voxel_size=.04,visual_hull_voxel_size=.08,
                maximum_proposals_per_update=args.birth_proposals,minimum_birth_separation=.04),
        }
        records = {name:[] for name in accumulators}
        born = {name:[] for name in accumulators}
        with torch.no_grad():
            for number,index in enumerate(order):
                view = views[index]; canopy,_ = fields(view)
                if int(canopy.sum()) < 128:
                    continue
                evidence = geometry.fields(view.image_name,device=torch.device("cuda"),shape=canopy.shape)
                _,bounds,valid,reliability,_ = _moge3_depth_query_bounds(evidence,global_log_scale_sigma=args.global_log_scale_sigma)
                coverage = render_hybrid(view,teacher.surface,foliage,
                    background=torch.zeros(3,device="cuda"),include_dynamic=False,
                    surface_gate=torch.zeros_like(teacher.surface.get_opacity.reshape(-1)),
                    volume_gate=(~foliage.dynamic_leaf_mask).float(),volume_depth_query_bounds=bounds,
                    optical_replacement_policy="disabled",structural_trainable_start=None).volume_hit_interval_alpha
                inside,_,_ = _tree_boundary_masks(~canopy)
                proposals,audit = _moge3_uncovered_hit_proposals(view,bounds,coverage,
                    {"p_canopy":canopy.float(),"p_canopy_core":(canopy & ~inside).float()},
                    valid,reliability,maximum_proposals=args.birth_proposals)
                for name,accumulator in accumulators.items():
                    accumulator.add(proposals,sequence_id=canonical)
                    result = accumulator.drain(maximum_births=2048,minimum_sequences=1)
                    records[name].append({"index":index,**result[4]})
                    if len(result[0]):
                        born[name].append(result[0])
                if number % 20 == 0:
                    print(json.dumps({"view":number,**{k:v.total_births for k,v in accumulators.items()}}),flush=True)
        (args.output/"birth_audit.json").write_text(json.dumps({"scope":"identical_read_only_proposals__not_trained_or_verified_births",
            "args":vars(args),"records":records},default=str,indent=2))
        torch.save({name:torch.cat(parts) if parts else torch.empty(0,3) for name,parts in born.items()},args.output/"candidate_centers.pth")
        print(json.dumps({"completed":True,**{k:v.total_births for k,v in accumulators.items()}}),flush=True)
        return
    evaluate(0)
    if args.evaluate_opacity is not None:
        payload = torch.load(args.evaluate_opacity, map_location="cpu")
        if Path(payload["source_checkpoint"]).resolve() != args.checkpoint.resolve() or not payload.get("diagnostic_only"):
            raise ValueError("Opacity snapshot does not belong to this diagnostic source")
        if payload.get('replay_base')!=replay_base or payload.get('opacity_only_replay_valid') is not True:
            raise ValueError('Opacity replay requires identical leaf geometry/appearance and no omitted learned state')
        with torch.no_grad():
            foliage.opacity_logits.copy_(payload["opacity_logits"].to(foliage.opacity_logits))
        evaluate(int(payload["step"]))
        return
    if args.canopy_sampling:
        # A scene-wide foliage subproblem, never selected from held-out errors.
        order = [i for i in order if int(fields(views[i])[0].sum()) >= 128]
        if not order:
            raise RuntimeError("No canopy-bearing training cameras")
    print(json.dumps({"sampled_view_count":len(order),"sampled_indices":order}),flush=True)
    for step in range(1,args.steps+1):
        lr_multiplier=args.lr_final_factor**((step-1)/max(args.steps-1,1))
        for group,rate in zip(optimizer.param_groups,initial_learning_rates):group['lr']=rate*lr_multiplier
        view = views[order[(step-1)%len(order)]]
        canopy, rigid = fields(view)
        if (args.rigid_preservation_weight > 0 or args.rigid_boundary_preservation_weight > 0) and view.image_name not in rigid_reference:
            with torch.no_grad():
                if args.replacement_foliage is not None:
                    # Preserve the achieved INPUT reconstruction, not any
                    # wall contamination introduced by the new initializer.
                    teacher.foliage=reference_foliage
                    try:
                        rigid_reference[view.image_name]=teacher.render(view,task=None,conditioned=False)["rgb"].cpu()
                    finally:
                        teacher.foliage=foliage
                else:
                    current = foliage.opacity_logits.detach().clone()
                    foliage.opacity_logits.copy_(original_opacity)
                    try:
                        rigid_reference[view.image_name] = teacher.render(view, task=None, conditioned=False)["rgb"].cpu()
                    finally:
                        foliage.opacity_logits.copy_(current)
        evidence = geometry.fields(view.image_name, device=torch.device("cuda"), shape=canopy.shape)
        if args.rigid_anchor_scale:
            from outdoor.canopy_depth_calibration import calibrated_canopy_evidence
            index=order[(step-1)%len(order)]
            calibration=depth_calibrations.get(index,depth_calibrations.get(str(index)))
            if calibration is None:raise ValueError('Missing immutable seed depth calibration for this camera')
            evidence=calibrated_canopy_evidence(evidence,calibration['scale'])
        pre_bounds, hit_bounds, valid, reliability, _ = _moge3_depth_query_bounds(evidence,global_log_scale_sigma=args.global_log_scale_sigma)
        ordering_audit=None
        if args.hit_interval_scope == "scale_uncertainty" or args.independent_rigid_depth_ordering or args.rigid_anchor_scale:
            if view.image_name not in rigid_depth_reference:
                with torch.no_grad():
                    rigid_package = render_hybrid(view,teacher.surface,foliage,
                        background=torch.zeros(3,device="cuda"),include_dynamic=False,
                        volume_gate=torch.zeros_like(foliage.opacity_logits.reshape(-1)),
                        optical_replacement_policy="disabled",structural_trainable_start=None)
                    rigid_depth_reference[view.image_name] = (rigid_package.median_depth.cpu(),rigid_package.surface_alpha.cpu())
                    if args.foreground_darkening_weight>0:
                        background=(rigid_package.render+(1-rigid_package.alpha)*teacher.sky(view)).clamp(0,1)
                        core=(1-F.max_pool2d((~rigid).float()[None,None],5,1,2)[0,0])>.5
                        residual=(background-view.original_image.cuda()).abs().amax(dim=0)
                        anchors=core&(rigid_package.surface_alpha.reshape_as(core)>=.98)&torch.isfinite(residual)
                        slack=float(torch.quantile(residual[anchors],.95))+.02 if int(anchors.sum())>=512 else None
                        rigid_background_reference[view.image_name]=(background.cpu(),slack)
                    del rigid_package
            rigid_depth,rigid_alpha = rigid_depth_reference[view.image_name]
            if args.hit_interval_scope == "scale_uncertainty":
                hit_bounds = uncertainty_hit_bounds(pre_bounds,hit_bounds,rigid_depth.cuda(),rigid_alpha.cuda())
                valid = valid & torch.isfinite(hit_bounds).all(dim=0)
            if args.independent_rigid_depth_ordering:
                pre_bounds,hit_bounds,valid,ordering_audit=clip_canopy_depth_queries_before_rigid(
                    pre_bounds,hit_bounds,valid,canopy.float(),rigid.float(),rigid_depth.cuda(),rigid_alpha.cuda())
        exact = (foliage.support_camera_ids == int(view.colmap_id)).any(dim=1)
        owner = _moge3_canopy_hit_owner_mask(foliage,exact)
        if args.production_geometry_objective!='none':
            from scripts.train_unified_outdoor_teacher import _static_detail_refinement_gate
            owner &= _static_detail_refinement_gate(foliage,
                _static_detail_canonical_ownership_gate(foliage,int(view.colmap_id),None)
                if args.geometry_owner_scope=='exact' else None)>0
        if args.color_owner_scope=='exact':
            owner &= _static_detail_canonical_ownership_gate(foliage,int(view.colmap_id),None)>0
        if args.include_envelope:
            envelope = foliage.persistent_envelope_mask & persistent_static_evidence_mask(foliage)
            if args.envelope_scope in ("exact", "resolved"):
                resolved = (foliage.verification_state == 1) & (foliage.proposal_kind == 0)
                envelope |= foliage.persistent_envelope_mask & resolved & (exact if args.envelope_scope == "exact" else torch.ones_like(exact))
            owner |= envelope
        optimizer.zero_grad(set_to_none=True)
        if ray_displacement is not None or source_group_factors is not None:foliage.xyz.grad=None
        if source_group_factors is not None:foliage.log_scales.grad=None
        # Two independent readouts: positive thin hit and conservative rigid
        # free space. Only explicitly enabled leaf variables may update.
        common = dict(background=torch.zeros(3,device="cuda"),include_dynamic=False,
            surface_gate=torch.zeros_like(teacher.surface.get_opacity.reshape(-1)),
            volume_gate=owner.float(), optical_replacement_policy="disabled",structural_trainable_start=None)
        positive = canopy & valid; negative = rigid & valid
        # A zero-weight query cannot affect optimization. Avoid building its
        # million-leaf CUDA graph, particularly in RGB-only depth ablations.
        # Explicit growth permission still requires a real differentiable hit.
        need_hit = args.require_thin_growth or (args.optical_weight != 0 and bool(positive.any()))
        need_pre = args.negative_weight != 0 and bool(negative.any())
        hit = (render_hybrid(view,teacher.surface,foliage,volume_depth_query_bounds=hit_bounds,**common).volume_hit_interval_alpha.reshape_as(canopy)
               if need_hit else torch.zeros_like(canopy,dtype=foliage.xyz.dtype))
        pre = (render_hybrid(view,teacher.surface,foliage,volume_depth_query_bounds=pre_bounds,**common).volume_prehit_alpha.reshape_as(canopy)
               if need_pre else torch.zeros_like(canopy,dtype=foliage.xyz.dtype))
        tau = -torch.log1p(-hit.clamp(max=1-1e-5))
        deficit = (math.log(-math.log(.25)) - torch.log(tau+1e-6)).clamp_min(0)
        pos_loss = (F.smooth_l1_loss(deficit,torch.zeros_like(deficit),reduction="none") * positive * reliability).sum()/(positive*reliability).sum().clamp_min(1)
        neg_loss = (torch.log1p(pre.clamp_min(0)/.005) * negative * reliability).sum()/(negative*reliability).sum().clamp_min(1)
        rgb_loss = pos_loss.new_zeros(())
        canopy_rgb_loss=pos_loss.new_zeros(())
        preservation_loss = pos_loss.new_zeros(())
        boundary_preservation_loss = pos_loss.new_zeros(())
        sky_loss=pos_loss.new_zeros(())
        detail_loss=pos_loss.new_zeros(())
        darkening_loss=pos_loss.new_zeros(());darkening_audit=None
        native_rigid_alpha_loss=pos_loss.new_zeros(())
        confirmed_loss=pos_loss.new_zeros(());confirmed_count=0
        if args.rgb_weight > 0 or args.rigid_preservation_weight > 0 or args.rigid_boundary_preservation_weight > 0 or args.known_sky_weight>0 or args.canopy_ssim_weight>0 or args.foreground_darkening_weight>0 or args.production_color_objective!='none' or args.native_rigid_alpha_weight>0 or args.confirmed_occluder_weight>0:
            native=teacher.render(view, task=None, conditioned=False)
            prediction = native["rgb"]
            if args.confirmed_occluder_weight>0:
                from outdoor.canopy_occluder_loss import confirmed_occluder_loss
                confirmed=confirmed_pixels(order[(step-1)%len(order)],canopy)
                confirmed_count=int(confirmed.sum())
                confirmed_loss=confirmed_occluder_loss(native['volume_alpha'],confirmed)
            if args.native_rigid_alpha_weight>0:
                from outdoor.canopy_detail_loss import native_rigid_foliage_extinction_loss
                native_rigid_alpha_loss=native_rigid_foliage_extinction_loss(native['volume_alpha'],rigid)
            rgb_error = (prediction-view.original_image.cuda()).abs().mean(dim=0)
            canopy_rgb_loss = (rgb_error*canopy).sum()/canopy.sum().clamp_min(1)
            rgb_loss = canopy_rgb_loss
            rgb_loss = rgb_loss + (rgb_error*rigid).sum()/rigid.sum().clamp_min(1)
            if args.canopy_ssim_weight>0:
                detail_loss=masked_canopy_ssim_loss(prediction,view.original_image.cuda(),canopy)
            if args.rigid_preservation_weight > 0 or args.rigid_boundary_preservation_weight > 0:
                delta = (prediction-rigid_reference[view.image_name].cuda()).abs().mean(dim=0)
                preservation_loss = (delta*rigid).sum()/rigid.sum().clamp_min(1)
                if args.rigid_boundary_preservation_weight > 0:
                    _,outside,_=_tree_boundary_masks(~canopy)
                    hard=rigid & outside
                    boundary_preservation_loss=(delta*hard).sum()/hard.sum().clamp_min(1)
            if args.known_sky_weight>0:
                observed_sky=sky_field(view)
                sky_alpha=native['volume_alpha'].reshape_as(observed_sky)
                sky_loss=(torch.log1p(sky_alpha.clamp_min(0)/.005)*observed_sky).sum()/observed_sky.sum().clamp_min(1)
            if args.foreground_darkening_weight>0:
                background,slack=rigid_background_reference[view.image_name]
                darkening_audit={'background_slack':slack,'applicable_pixels':0,'deficit_pixels':0}
                if slack is not None and canopy.any():
                    wall=rigid_depth.cuda().reshape_as(canopy);wall_alpha=rigid_alpha.cuda().reshape_as(canopy)
                    upper=wall-torch.maximum(torch.full_like(wall,.03),wall*.01)
                    known=(wall_alpha>=.99)&torch.isfinite(upper)&(upper>.2001)
                    bounds=torch.stack((torch.full_like(wall,.2001),upper));bounds[:,~known]=float('nan')
                    front=render_hybrid(view,teacher.surface,foliage,volume_depth_query_bounds=bounds,**common).volume_hit_interval_alpha.reshape_as(canopy)
                    core=F.avg_pool2d(canopy.float()[None,None],7,1,3,count_include_pad=True)[0,0]>=1-1.e-6
                    # A scalar layer bound is inappropriate when native mixed
                    # ordering disagrees with the isolated front contribution.
                    applicable=core&known&((front.detach()-native['volume_alpha'].detach().reshape_as(canopy)).abs()<.01)
                    required=foreground_darkening_bound(view.original_image.cuda(),background.cuda(),slack)
                    darkening_loss=foreground_darkening_loss(front,required,applicable)
                    darkening_audit.update(applicable_pixels=int(applicable.sum()),
                        deficit_pixels=int((applicable&(required>front.detach()+.01)).sum()))
        growth_permission = owner
        if args.require_thin_growth:
            hit_gradient = torch.autograd.grad(-(hit*positive).sum(),foliage.opacity_logits,retain_graph=True)[0]
            growth_permission = owner & (hit_gradient.reshape(-1) < -1e-12)
        position_prior=pos_loss.new_zeros(())
        if args.position_lr>0:
            errors=(foliage.xyz-initial_positions).square().sum(dim=1)/position_radius.square()
            position_prior=((errors*owner).sum()/owner.sum().clamp_min(1)
                            if args.production_geometry_objective!='none' else errors[owner].mean())
        loss = args.optical_weight * pos_loss + args.negative_weight * neg_loss + args.rgb_weight * rgb_loss + args.rigid_preservation_weight * preservation_loss + args.rigid_boundary_preservation_weight * boundary_preservation_loss + args.position_prior_weight*position_prior+args.known_sky_weight*sky_loss
        loss = loss + args.canopy_ssim_weight*detail_loss
        loss = loss + args.foreground_darkening_weight*darkening_loss
        loss = loss + args.native_rigid_alpha_weight*native_rigid_alpha_loss
        loss = loss + args.confirmed_occluder_weight*confirmed_loss
        geometry_objective=None;geometry_rigid_alpha_loss=None
        if args.production_geometry_objective!='none':
            from outdoor.hybrid_gaussian_renderer import static_detail_forward_visibility_gate
            from outdoor.directional_sky import composite_white_background
            from scripts.train_unified_outdoor_teacher import _oriented_multiscale_high_frequency_loss
            visibility=(static_detail_forward_visibility_gate(foliage,int(view.colmap_id),include_pending_exact=False)
                        if args.production_geometry_objective=='native' else owner.float())
            geometry_package=render_hybrid(view,teacher.surface,foliage,
                background=torch.ones(3,device='cuda'),include_dynamic=False,
                structural_trainable_start=None,optical_replacement_policy='disabled',
                volume_gate=visibility,volume_geometry_gradient_gate=owner.float(),
                volume_appearance_gradient_gate=torch.zeros_like(visibility),
                volume_opacity_gradient_gate=torch.zeros_like(visibility))
            geometry_rgb=composite_white_background(geometry_package.render,geometry_package.alpha,teacher.sky(view))
            geometry_objective=(_photo_loss(geometry_rgb,view.original_image.cuda(),canopy.float(),.2)
                +.1*_oriented_multiscale_high_frequency_loss(geometry_rgb,view.original_image.cuda(),canopy.float()))
            if args.geometry_rigid_alpha_weight>0:
                # Observation masks are losses, never deployment render gates.
                # Frozen wall colors cannot absorb an incorrect foreground.
                # Avoid asking uncertain one-pixel semantic boundaries to
                # establish empty rays. Hidden foliage has zero native mass.
                from outdoor.canopy_detail_loss import native_rigid_foliage_extinction_loss
                geometry_rigid_alpha_loss=native_rigid_foliage_extinction_loss(geometry_package.volume_alpha,rigid)
                geometry_objective=geometry_objective+args.geometry_rigid_alpha_weight*geometry_rigid_alpha_loss
            loss=loss+geometry_objective
        # Empty-evidence views have an exact zero optical update, not an
        # artificial hit. Keep backward valid when every active mask is empty.
        loss = loss + foliage.opacity_logits.sum()*0
        owned_color_gradient=None
        optimized_color_loss=None
        if args.appearance_lr>0 and not args.allow_rigid_leaf_color_gradients:
            color_loss=args.rgb_weight*canopy_rgb_loss+args.canopy_ssim_weight*detail_loss
            if args.production_color_objective=='native':
                color_prediction,color_weight=_static_volume_native_color_inputs(native['rgb'],native['volume_alpha'],canopy.float())
                color_loss=_photo_loss(color_prediction,view.original_image.cuda(),color_weight,.1)
            elif args.production_color_objective=='legacy_intrinsic':
                from outdoor.hybrid_gaussian_renderer import static_detail_forward_visibility_gate
                isolated=render_hybrid(view,teacher.surface,foliage,background=torch.ones(3,device='cuda'),
                    include_dynamic=False,optical_replacement_policy='disabled',structural_trainable_start=None,
                    surface_gate=torch.zeros_like(teacher.surface.get_opacity.reshape(-1)),volume_role_mask=foliage.static_leaf_mask,
                    volume_gate=static_detail_forward_visibility_gate(foliage,int(view.colmap_id),include_pending_exact=False),
                    volume_opacity_gradient_gate=torch.zeros_like(foliage.opacity_logits.reshape(-1)))
                color_prediction,color_weight=_static_volume_intrinsic_color_inputs(isolated.render,isolated.volume_alpha,torch.ones(3,device='cuda'),canopy.float())
                color_loss=_photo_loss(color_prediction,view.original_image.cuda(),color_weight,.1)
            owned_color_gradient=canopy_owned_color_gradient(color_loss,foliage.features)
            optimized_color_loss=float(color_loss.detach())
        loss.backward()
        if owned_color_gradient is not None:
            # Negative/background pixels must remove or relocate the wrong
            # optical contributor, not recolor it to impersonate the wall.
            # Geometry/opacity keep the FULL loss gradient unchanged.
            foliage.features.grad=owned_color_gradient
        opacity_owner = owner & exact if args.opacity_owner_scope == 'exact' else owner
        geometry_owner=(owner & persistent_static_evidence_mask(foliage)
                        if args.persistent_checkpoint_geometry else owner)
        foliage.opacity_logits.grad[~opacity_owner] = 0
        previous_positions=foliage.xyz.detach().clone() if args.position_lr>0 else None
        if args.position_lr>0: foliage.xyz.grad[~geometry_owner]=0
        previous_ray_displacement=None
        if ray_displacement is not None:
            # Chain rule for xyz = initial + unit_ray * scalar. Optimize the
            # scalar itself, not three independently normalized Adam axes.
            ray_displacement.grad=(foliage.xyz.grad*ray_directions).sum(dim=1).detach()
            previous_ray_displacement=ray_displacement.detach().clone()
        if source_group_factors is not None:
            source_group_factors.grad=source_depth_chain_gradient(
                foliage.xyz.grad,foliage.log_scales.grad,foliage.xyz.detach(),source_origins,
                source_group_index,len(source_ids),owner).detach()
        previous_extra=[parameter.detach().clone() for parameter in extra_parameters]
        for parameter in extra_parameters:
            parameter_owner=owner if parameter is foliage.features else geometry_owner
            if parameter.grad is not None:parameter.grad[~parameter_owner]=0
        if args.appearance_lr>0 and foliage.features.grad is not None:
            foliage.features.grad[:,trainable_color_coefficients:]=0
            if args.persistent_checkpoint_geometry:foliage.features.grad[~geometry_owner,1:]=0
        optimizer.step()
        with torch.no_grad():
            # Inactive rows cannot drift on optimizer history.
            inactive = ~opacity_owner | (~growth_permission & (foliage.opacity_logits.reshape(-1) > baseline.reshape(-1)))
            foliage.opacity_logits[inactive] = baseline[inactive]
            baseline.copy_(foliage.opacity_logits)
            for key in ("exp_avg","exp_avg_sq"):
                optimizer.state[foliage.opacity_logits][key][inactive] = 0
            if args.position_lr>0:
                position_parameter=foliage.xyz
                if source_group_factors is not None:
                    # Roughly two times the calibrated .10 log-depth uncertainty.
                    # This changes metric depth coherently, not source RGB footprint.
                    source_group_factors.clamp_(-.2,.2)
                    proposed_xyz,proposed_scales=source_depth_transform(
                        initial_positions,initial_scales,source_origins,source_group_index,source_group_factors,owner)
                    foliage.xyz.copy_(proposed_xyz);foliage.log_scales.copy_(proposed_scales)
                    position_parameter=source_group_factors
                elif ray_displacement is not None:
                    apply_source_ray_update_(foliage.xyz,initial_positions,ray_directions,ray_displacement,previous_ray_displacement,owner,position_radius)
                    position_parameter=ray_displacement
                else:
                    project_leaf_center_updates_(foliage.xyz,initial_positions,previous_positions,geometry_owner,position_radius)
                if source_group_factors is None:
                    for key in ('exp_avg','exp_avg_sq'):
                        optimizer.state[position_parameter][key][~geometry_owner]=0
            for parameter,previous in zip(extra_parameters,previous_extra):
                parameter_owner=owner if parameter is foliage.features else geometry_owner
                if parameter is foliage.features:
                    # Match standard GS rest/DC LR ratio by scaling the actual
                    # Adam displacement, not the gradient (Adam normalizes it).
                    project_leaf_color_updates_(parameter,previous,owner,trainable_color_coefficients)
                if parameter is foliage.log_scales:
                    project_leaf_shape_updates_(parameter,initial_scales,previous,geometry_owner)
                if parameter is foliage.quaternions:
                    project_leaf_rotation_updates_(parameter,initial_rotations,previous,geometry_owner)
                _restore_parameter_update_authority_(parameter,previous,parameter_owner,
                    optimizer.state.get(parameter,{}),high_order_owner=(geometry_owner
                    if parameter is foliage.features and args.persistent_checkpoint_geometry else None))
                if parameter is foliage.features:
                    for key in ('exp_avg','exp_avg_sq'):
                        if key in optimizer.state[parameter]:optimizer.state[parameter][key][:,trainable_color_coefficients:]=0
        if step % 25 == 0:
            state=optimizer.state[foliage.opacity_logits]
            beta2=optimizer.param_groups[0]['betas'][1]
            variance_root=(state['exp_avg_sq']/(1-beta2**float(state['step']))).sqrt().reshape(-1)
            active=owner & (variance_root>0)
            damping=variance_root[active]/(variance_root[active]+args.adam_eps)
            trace = {"step":step,"loss":float(loss),"owner_rows":int(owner.sum()),
                              'learning_rate_multiplier':lr_multiplier,
                              'adam_eps':args.adam_eps,
                              'appearance_gradient_scope':'legacy_all_rgb_regions' if args.allow_rigid_leaf_color_gradients else 'canopy_only',
                              'foreground_darkening':darkening_audit,
                              'production_color_objective':args.production_color_objective,
                              'production_geometry_objective':args.production_geometry_objective,
                              'geometry_owner_scope':args.geometry_owner_scope,
                              'opacity_owner_scope':args.opacity_owner_scope,
                              'opacity_owner_rows':int(opacity_owner.sum()),
                              'source_group_log_scale_quantiles':(torch.quantile(source_group_factors.detach(),source_group_factors.new_tensor([0.,.1,.5,.9,1.])).tolist() if source_group_factors is not None else None),
                              'geometry_rgb_loss':float(geometry_objective) if geometry_objective is not None else None,
                              'geometry_rigid_alpha_loss':float(geometry_rigid_alpha_loss) if geometry_rigid_alpha_loss is not None else None,
                              'native_rigid_alpha_loss':float(native_rigid_alpha_loss),
                              'confirmed_occluder_loss':float(confirmed_loss) if args.occluder_evidence_scope=='confirmed' else None,
                              'confirmed_occluder_pixels':confirmed_count if args.occluder_evidence_scope=='confirmed' else None,
                              'occluder_evidence_scope':args.occluder_evidence_scope,
                              'occluder_loss':float(confirmed_loss),
                              'occluder_pixels':confirmed_count,
                              'geometry_owner_rows':int(geometry_owner.sum()),
                              'optimized_color_loss':optimized_color_loss,
                              'opacity_adam_epsilon_update_fraction_q10_q50_q90':torch.quantile(damping,damping.new_tensor([.1,.5,.9])).tolist() if len(damping) else None,
                              'same_state_counterfactual_eps1e8_fraction_q10_q50_q90':torch.quantile(variance_root[active]/(variance_root[active]+1.e-8),damping.new_tensor([.1,.5,.9])).tolist() if len(damping) else None,
                              "independent_rigid_ordering":ordering_audit,
                              "known_sky_negative_loss":float(sky_loss),
                              "canopy_structural_loss":float(detail_loss) if args.canopy_ssim_weight>0 else None,
                              "positive_pixels":int(positive.sum()),
                              "hit_query_executed":need_hit,"prehit_query_executed":need_pre,
                              "hit_mean":float(hit[positive].mean()) if need_hit and positive.any() else None,
                              "rigid_prehit":float(pre[negative].mean()) if need_pre and negative.any() else None}
            print(json.dumps(trace),flush=True)
            with (args.output / "trace.jsonl").open("a") as stream:
                stream.write(json.dumps(trace)+"\n")
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(step)
            torch.save({"opacity_logits":foliage.opacity_logits.detach().cpu(),"step":step,
                        "source_checkpoint":str(args.checkpoint),"diagnostic_only":True,
                        "replay_base":replay_base,"opacity_only_replay_valid":not full_refinement,
                        "hit_interval_scope":args.hit_interval_scope},args.output/f"opacity_{step:04d}.pth")
            if (full_refinement or args.save_foliage_capture) and (step%400==0 or step==args.steps):
                torch.save({'diagnostic_only':True,'source_checkpoint':str(args.checkpoint.resolve()),
                    'scope':'fresh_dense_canonical_canopy_only__bounded_local_refinement__requires_reverification_before_production',
                    'rigid_anchor_scale':args.rigid_anchor_scale,
                    'rigid_anchor_calibrations':depth_calibrations,
                    'position_mode':args.position_mode,
                    'step':step,'foliage':foliage.capture()},args.output/f'foliage_{step:04d}.pth')


if __name__ == "__main__":
    main()
