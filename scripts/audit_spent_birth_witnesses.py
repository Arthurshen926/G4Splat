"""Read-only bounded comparison of spent-witness birth feasibility."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from outdoor.static_ray_birth import StaticRayBirthAccumulator,_Cell


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--maximum-cells',type=int,default=5000)
    a=p.parse_args()
    if a.maximum_cells<1:raise ValueError('Positive bounded sample required')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    state=torch.load(a.checkpoint,map_location='cpu')['static_ray_birth_state']
    repaired=StaticRayBirthAccumulator(voxel_size=state['voxel_size'],
        visual_hull_voxel_size=state['visual_hull_voxel_size'])
    original=StaticRayBirthAccumulator(voxel_size=state['voxel_size'],
        visual_hull_voxel_size=state['visual_hull_voxel_size'])
    repaired.consumed_first_hit_witnesses={tuple(row) for row in state['consumed_first_hit_witnesses']}
    selected=[];partial=0;eligible_partial=0;support_width_histogram={}
    # Ordered bounded sample, explicitly not a random population estimate.
    for row in state['visual_hull_cells']:
        cameras=set(row['cameras'])
        spent={int(cam) for cam,token in row.get('first_hit_witnesses',{}).items()
               if tuple(token) in repaired.consumed_first_hit_witnesses}
        available=cameras-spent
        if len(available)>=2:
            width=len(available & row['depth_constraints'].keys())
            support_width_histogram[width]=support_width_histogram.get(width,0)+1
        if not spent:continue
        partial+=1
        if len(cameras-spent)<2:continue
        eligible_partial+=1
        if len(selected)>=a.maximum_cells:continue
        cell=_Cell(torch.as_tensor(row['weighted_center']).float(),float(row['weight']),
            torch.as_tensor(row['weighted_color']).float(),float(row['color_weight']),
            cameras,set(row['sequences']),
            {int(cam):(torch.as_tensor(v[0]).float(),float(v[1]),float(v[2]))
             for cam,v in row['depth_constraints'].items()},
            {int(k):v for k,v in row.get('camera_sequences',{}).items()},
            {int(k):tuple(v) for k,v in row.get('first_hit_witnesses',{}).items()})
        selected.append(('visual_hull',tuple(row['key']),cell))
    old_center,old_valid=original._resolve_depth_centers(selected)
    new_center,new_valid=repaired._resolve_depth_centers(selected)
    result=dict(scope='ordered_bounded_candidate_sample__not_render_quality_or_birth_acceptance',
        checkpoint=str(a.checkpoint.resolve()),pending_cells=len(state['visual_hull_cells']),
        cells_with_spent_support=partial,cells_with_two_remaining_cameras=eligible_partial,
        sampled_cells=len(selected),previous_feasible=int(old_valid.sum()),repaired_feasible=int(new_valid.sum()),
        newly_feasible=int((new_valid&~old_valid).sum()),newly_infeasible=int((old_valid&~new_valid).sum()),
        available_camera_depth_width_histogram=support_width_histogram)
    if support_width_histogram:
        result['padded_constraint_slots']=max(support_width_histogram)*sum(support_width_histogram.values())
        result['grouped_constraint_slots']=sum(width*count for width,count in support_width_histogram.items())
    if selected:result['maximum_center_change']=float((new_center-old_center).norm(dim=1).max())
    (a.output/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
