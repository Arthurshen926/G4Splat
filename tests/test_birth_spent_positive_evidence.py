import torch
from outdoor.static_ray_birth import StaticRayBirthAccumulator, _Cell


def example():
    accumulator=StaticRayBirthAccumulator()
    token=(0,0,0,100000)
    cell=_Cell(torch.tensor([.15,.15,.15]),1.,torch.tensor([1.,0.,0.]),1.,
        {0,1,2},{'seq2'},
        {0:(torch.tensor([0.,0.,1.]),.1,.12),
         1:(torch.tensor([0.,0.,1.]),.18,.2),
         2:(torch.tensor([0.,0.,1.]),.18,.2)},
        {0:'seq2',1:'seq2',2:'seq2'}, {0:token})
    accumulator.visual_hull_cells[(0,0,0)]=cell
    accumulator.consumed_first_hit_witnesses.add(token)
    return accumulator,cell


def test_spent_first_hit_is_not_positive_depth_for_a_second_hidden_leaf():
    accumulator,cell=example()
    assert accumulator._available_support(cell)[0]=={1,2}
    centers,valid=accumulator._resolve_depth_centers([('visual_hull',(0,0,0),cell)])
    assert valid.tolist()==[True]
    assert .18-1.e-6<=float(centers[0,2])<=.2+1.e-6


def test_birth_does_not_claim_color_from_an_irreversibly_mixed_spent_witness():
    accumulator,_=example()
    centers,colors,support,_,audit=accumulator.drain(maximum_births=1,minimum_sequences=1)
    assert len(centers)==1 and set(support[0].tolist())=={1,2}
    assert torch.isnan(colors).all()
    assert audit['spent_witness_color_unknown_births']==1


def test_witness_spent_earlier_in_same_drain_forces_center_and_color_refresh():
    accumulator=StaticRayBirthAccumulator()
    token=(0,0,0,100000);normal=torch.tensor([0.,0.,1.])
    for key,cameras,z in [((0,0,0),{0,3,4,5},.15),((0,0,1),{0,1,2},.58)]:
        constraints={camera:(normal,.1,.6) if camera==0 else
                     (normal,.1,.2) if camera>=3 else (normal,.38,.59) for camera in cameras}
        accumulator.visual_hull_cells[key]=_Cell(torch.tensor([.15,.15,z]),1.,torch.ones(3),1.,
            cameras,{'seq2'},constraints,{camera:'seq2' for camera in cameras},{0:token})
    centers,colors,support,_,audit=accumulator.drain(maximum_births=2,minimum_sequences=1)
    assert len(centers)==2
    torch.testing.assert_close(centers[1],torch.tensor([.15,.15,.45]))
    assert set(support[1][support[1]>=0].tolist())=={1,2}
    assert torch.isfinite(colors[0]).all() and torch.isnan(colors[1]).all()
    assert audit['spent_witness_color_unknown_births']==1


def test_nearly_parallel_available_slabs_preserve_a_valid_warm_start():
    accumulator,cell=example()
    point=torch.tensor([.28,.28,.15]);cell.weighted_center=point.clone()
    normal=torch.tensor([1.,.01,0.]);normal/=normal.norm()
    depth=float(point.dot(normal))
    cell.depth_constraints[1]=(torch.tensor([1.,0.,0.]),.28-1e-6,.28+1e-6)
    cell.depth_constraints[2]=(normal,depth-1e-6,depth+1e-6)
    centers,valid=accumulator._resolve_depth_centers([('visual_hull',(0,0,0),cell)])
    assert valid.tolist()==[True]
    torch.testing.assert_close(centers[0],point,atol=1e-5,rtol=0)
