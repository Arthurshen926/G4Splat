"""Read-only TRAINING contribution cohort; not a visible-wall truth certificate."""
import json
import torch
from scripts.canopy_selective_shape import initial_geometry_sha256,select_conflicting_candidates


@torch.no_grad()
def audit_selectivity(teacher,base,adapter,candidate,views,masks,manifest,output,*,snapshot_sha256):
    from outdoor.hybrid_gaussian_renderer import render_hybrid,static_detail_forward_visibility_gate
    from scripts.evaluate_hybrid_teacher import _tree_boundary_masks
    if manifest['args'].get('leaf_optical_kernel','native')!='native' or manifest['args']['candidate_gain']!=1:
        raise ValueError('Native enabled candidates required')
    if set(manifest['training_views'])&set(manifest['excluded_views']):raise ValueError('Training views only')
    mass=torch.zeros(len(candidate.xyz),3,device='cuda',dtype=torch.float64)
    counts=torch.zeros_like(mass,dtype=torch.int32)
    common=dict(background=torch.zeros(3,device='cuda'),include_dynamic=False,
                optical_replacement_policy='disabled',structural_trainable_start=None)
    for ordinal,index in enumerate(manifest['training_views']):
        v=views[index]
        obj,ns,dist,nt=masks.get_index_masks(v.image_name,(0,1,2,3),(v.image_height,v.image_width),torch.device('cuda'))
        tree=obj&ns&dist&~nt;rigid=obj&ns&dist&nt;_,boundary,_=_tree_boundary_masks(~tree)
        gate=static_detail_forward_visibility_gate(base,int(v.colmap_id),include_pending_exact=False)
        out=render_hybrid(v,teacher.surface,adapter,volume_gate=torch.cat((gate,torch.ones(len(candidate.xyz),device='cuda'))),
                         audit_fields=torch.stack((tree,rigid&~boundary,rigid&boundary)).float(),**common)
        contribution=out.responsibility[len(teacher.surface.get_xyz)+len(base):,1:]
        mass+=contribution;counts+=contribution>1e-5
        del out
        if ordinal%25==0:print(json.dumps(dict(selectivity_training_views=ordinal+1)),flush=True)
    eligible=select_conflicting_candidates(mass,counts)
    report=dict(scope='candidate_shape_freedom_only__not_verified_support_or_negative_truth',
        snapshot_sha256=snapshot_sha256,source_checkpoint_sha256=manifest['source_checkpoint_sha256'],
        initial_geometry_sha256=initial_geometry_sha256(candidate.cloud),
        training_views=manifest['training_views'],excluded_views=manifest['excluded_views'],
        candidate_count=len(candidate.xyz),selected_count=int(eligible.sum()),
        criteria='tree contribution>1e-5 in>=2 views; eroded rigid contribution>1e-5 in>=2 views; rigid fraction>1%',
        limitations=['Segmentation-based audit grants bounded experimental shape freedom, never suppression or verification authority',
                     'Cohort measured at a fixed trained state; geometry identity bound separately to immutable initial candidates'])
    torch.save(dict(eligible=eligible.cpu(),mass=mass.cpu(),counts=counts.cpu(),report=report),output/'shape_cohort.pth')
    (output/'shape_cohort.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
