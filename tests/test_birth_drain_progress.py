import torch
from outdoor.static_ray_birth import StaticRayBirthAccumulator,_Cell


def accumulator():
    a=StaticRayBirthAccumulator(voxel_size=.1,visual_hull_voxel_size=.2)
    a.cells[(0,0,0)]=_Cell(torch.tensor([.05,.05,.05]),1.,torch.ones(3),1.,{1,2},{'a','b'})
    return a


def test_zero_capacity_never_scans_or_solves_and_does_not_consume_evidence(monkeypatch):
    a=accumulator();events=[];cell=a.cells[(0,0,0)]
    def forbidden(*args):raise AssertionError('No eligibility/geometry work at zero capacity')
    monkeypatch.setattr(a,'_available_support',forbidden)
    monkeypatch.setattr(a,'_resolve_depth_centers',forbidden)
    xyz,rgb,cameras,sequences,audit=a.drain(maximum_births=0,progress_callback=events.append)
    assert xyz.shape==rgb.shape==(0,3) and cameras.shape==(0,0) and sequences.shape==(0,)
    assert not audit['eligibility_evaluated'] and audit['skipped_reason']=='zero_birth_capacity'
    assert a.cells[(0,0,0)] is cell and not a.consumed_midpoint_cells and a.total_births==0
    assert [e['stage'] for e in events]==['filtering','skipped_zero_capacity']


def test_progress_reporting_does_not_change_selected_material():
    a=accumulator();b=accumulator();events=[]
    left=a.drain(maximum_births=1);right=b.drain(maximum_births=1,progress_callback=events.append)
    for x,y in zip(left[:4],right[:4]):torch.testing.assert_close(x,y,atol=0,rtol=0)
    assert left[4]==right[4]
    assert [e['stage'] for e in events]==['filtering','depth_solve','priority','selection','complete']
    assert all(e['elapsed_sec']>=0 for e in events)
