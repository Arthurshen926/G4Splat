"""Read-only checkpoint audit of split-time color evidence integration."""
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
from outdoor.training_evidence import OutdoorGeometryEvidence
from outdoor.task_fields import OutdoorTaskFieldLookup
from outdoor.evidence_store import load_evidence_store, artifact_path
from scripts.train_unified_outdoor_teacher import _refresh_split_child_owner_colors
from scripts.evaluate_canopy_validation import tensor_digest


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(p)
    p.set_defaults(data_device='cpu', resolution=640, white_background=True)
    for name in ('checkpoint', 'initialization', 'evidence-store', 'masks', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--synthetic-split-smoke',action='store_true',help='Exercise a two-parent native transaction, not quality training')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.30)
    dataset = model.extract(a)
    dataset.model_path = str(a.output)
    teacher = load_hybrid_teacher(a.checkpoint, sh_degree=dataset.sh_degree)
    leaf = teacher.foliage
    views = LazyScene(dataset, GaussianModel(dataset.sh_degree), image_cache_size=4).getTrainCameras()
    lookup = {int(v.colmap_id): v for v in views}
    manifest = json.loads((a.initialization / 'initialization_manifest.json').read_text())
    seed = torch.load(manifest['foliage_seed'], map_location='cpu')
    profiles = seed['audit']['fresh_canonical_leaves']['depth_profiles_by_image']
    scale = seed['audit']['moge3_exact_k_front_hit']['metric_to_cambridge_scale']
    del seed
    geometry = OutdoorGeometryEvidence(a.evidence_store, chart_base_source='moge3_adaptive')
    geometry.configure_moge3_metric_scale(scale)
    geometry.configure_moge3_canopy_depth_scales(profiles)
    store = load_evidence_store(a.evidence_store, verify_hashes=False)
    task = OutdoorTaskFieldLookup(
        Path(dataset.source_path), a.masks, Path(store['semantic_contract']),
        multiview_track_archive=artifact_path(store, 'mast3r_multiview_tracks'),
        projected_rigid_posterior_archive=artifact_path(store, 'projected_rigid_conflict_posterior', required=False),
        max_cached_views=0,
    )
    if a.synthetic_split_smoke:
        from types import SimpleNamespace
        from scripts.train_unified_outdoor_teacher import (
            _adapt_volume, _volume_stats, _finalize_synchronous_measured_splits,
            _verify_measured_split_children_now,
        )
        eligible=leaf.static_leaf_mask & (leaf.verification_state==1) & (leaf.proposal_kind==0) & (leaf.support_view_count>=2)
        selected=torch.nonzero(eligible,as_tuple=False).flatten()[:2]
        if len(selected)!=2:raise RuntimeError('Two persistent parents required')
        stats=_volume_stats(leaf)
        for key in ('contribution','residual','gradient','gradient_count'):stats[key][selected]=1.
        stats['radius'][selected]=6.
        count=len(leaf);step=int(teacher.state['iteration'])+1
        args=SimpleNamespace(volume_split_radius=3.,maximum_volume_gaussians=count+8,maximum_volume_splits=2,
            reconstruction_target='static',static_canonical_sequence_policy='scene')
        surface_before={key:tensor_digest(getattr(teacher.surface,key)) for key in ('_xyz','_scaling','_rotation','_opacity')}
        event=_adapt_volume(args,leaf,stats,volume_budget=count+8,phase='static_foliage',current_iteration=step,
            view_by_camera_id=lookup,canonical_rgb_sequence='seq2',geometry_evidence=geometry,
            measured_canopy_task_fields=task)
        if not event['split_parents']:raise RuntimeError('Synthetic demand failed to exercise splitting')
        sequence_lookup=torch.full((max(lookup)+1,),-1,dtype=torch.int16,device='cuda')
        for camera_id,view in lookup.items():
            if view.image_name.startswith('seq2__'):sequence_lookup[camera_id]=0
        _finalize_synchronous_measured_splits(args,leaf,event,current_iteration=step,
            verify_children=lambda rows:_verify_measured_split_children_now(leaf,rows,surface=teacher.surface,
                geometry=geometry,task_fields=task,view_by_camera_id=lookup,
                camera_sequence_lookup=sequence_lookup,canonical_sequence='seq2'))
        assert len(event['_new_to_old'])==len(leaf)
        assert int(event['_new_to_old'].min())>=0 and int(event['_new_to_old'].max())<count
        assert surface_before=={key:tensor_digest(getattr(teacher.surface,key)) for key in surface_before}
        result={'scope':'two_parent_synthetic_demand_real_cuda_witnesses_not_quality_training',
            'source_checkpoint':str(a.checkpoint),'selected_old_rows':selected.tolist(),
            'old_count':count,'new_count':len(leaf),'attempted_split_parents':event['attempted_split_parents'],
            'published_split_parents':event['split_parents'],'surface_geometry_opacity_unchanged':True,
            'single_optimizer_mapping_valid':True,
            'publication':event['child_verification_lifecycle']['synchronous_publication']}
        (a.output/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        return
    rows = torch.nonzero(leaf.split_generation > 0, as_tuple=False).flatten().cpu().tolist()
    if not rows:
        raise RuntimeError('Checkpoint has no split descendants')
    # Last contiguous descendant block, bounded independently of RGB quality.
    end = rows[-1] + 1
    start = rows[-1]
    rowset = set(rows)
    while start - 1 in rowset and end - start < 2048:
        start -= 1
    baseline = leaf.features.detach().clone()
    before = {k: tensor_digest(getattr(leaf, k)) for k in ('xyz', 'log_scales', 'quaternions', 'opacity_logits')}
    results = {}
    for label, fields in [('legacy', None), ('measured', task)]:
        leaf.features.copy_(baseline)
        audit = _refresh_split_child_owner_colors(
            leaf, child_start=start,
            owner_camera_ids=torch.full((end-start,), -1, device='cuda'),
            view_by_camera_id=lookup, canonical_rgb_sequence='seq2',
            geometry_evidence=geometry, measured_canopy_task_fields=fields,
        )
        audit['changed_dc_rows'] = int((leaf.features[start:end, 0] != baseline[start:end, 0]).any(dim=1).sum())
        results[label] = audit
    leaf.features.copy_(baseline)
    assert before == {k: tensor_digest(getattr(leaf, k)) for k in before}
    result = {'checkpoint': str(a.checkpoint), 'range': [start, end],
              'scope': 'current_split_descendants_read_only_audit_not_quality_evaluation',
              'results': results, 'geometry_and_opacity_unchanged': True}
    (a.output / 'audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
