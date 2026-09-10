import torch
from outdoor.static_ray_birth import StaticRayBirthAccumulator, _Cell


def reference_depth_solver(accumulator, entries):
    centers=torch.stack([c.weighted_center/max(c.weight,1e-8) for _,_,c in entries])
    count=len(entries);width=max(len(c.depth_constraints) for _,_,c in entries)
    normals=torch.zeros(count,width,3)
    lower=torch.full((count,width),-float('inf'));upper=torch.full((count,width),float('inf'))
    complete=torch.zeros(count,dtype=torch.bool)
    for i,(_,_,cell) in enumerate(entries):
        complete[i]=cell.cameras.issubset(cell.depth_constraints)
        for j,camera in enumerate(sorted(cell.depth_constraints)):
            normal,lo,hi=cell.depth_constraints[camera]
            normals[i,j]=normal;lower[i,j],upper[i,j]=lo,hi
    box_lower=torch.tensor([key for _,key,_ in entries]).float()*accumulator.visual_hull_voxel_size
    box_upper=box_lower+accumulator.visual_hull_voxel_size
    for _ in range(24):
        for j in range(width):
            depth=(centers*normals[:,j]).sum(dim=1)
            delta=depth.clamp(min=lower[:,j],max=upper[:,j])-depth
            centers+=delta[:,None]*normals[:,j]
        centers=torch.maximum(box_lower,torch.minimum(box_upper,centers))
    depth=(centers[:,None,:]*normals).sum(dim=2)
    feasible=(complete & torch.isfinite(centers).all(dim=1)
        & (depth>=lower-1e-5).all(dim=1) & (depth<=upper+1e-5).all(dim=1))
    return centers,feasible


def make_entries(count=300):
    generator=torch.Generator().manual_seed(731)
    entries=[]
    for i in range(count):
        point=.10+.10*torch.rand(3,generator=generator)
        width=2+i%7
        normals=torch.randn(width,3,generator=generator)
        normals=normals/normals.norm(dim=1,keepdim=True)
        projected=(normals*point).sum(dim=1)
        constraints={j:(normals[j],float(projected[j]-.02),float(projected[j]+.02)) for j in range(width)}
        if i%10==0:
            constraints[0]=(torch.tensor([0.,0.,1.]),.25,.26)
        cell=_Cell(point.clone(),1.,torch.zeros(3),1.,set(range(width)),{'seq2'},constraints)
        entries.append(('visual_hull',(0,0,0),cell))
    return entries


def test_accelerated_depth_solver_matches_original_projected_centers_exactly():
    accumulator=StaticRayBirthAccumulator()
    entries=make_entries()
    expected=reference_depth_solver(accumulator,entries)
    actual=accumulator._resolve_depth_centers(entries)
    assert torch.equal(actual[0],expected[0])
    assert torch.equal(actual[1],expected[1])
    assert actual[1].any() and (~actual[1]).any()


def test_grouped_solver_matches_padded_solver_with_spent_witnesses_exactly():
    accumulator=StaticRayBirthAccumulator()
    entries=make_entries(1100)
    # One unusually well-observed cell must not pad every other candidate.
    cell=entries[-1][2]
    for camera in range(64):
        cell.cameras.add(camera)
        cell.depth_constraints[camera]=(torch.tensor([0.,0.,1.]),.1,.2)
    for _,_,cell in entries[::3]:
        cell.first_hit_witnesses[0]=(0,1,2,3)
        cell.camera_sequences={camera:'seq2' for camera in cell.cameras}
    accumulator.consumed_first_hit_witnesses.add((0,1,2,3))
    expected=accumulator._resolve_depth_centers_padded(entries)
    actual=accumulator._resolve_depth_centers(entries)
    assert torch.equal(actual[0],expected[0])
    assert torch.equal(actual[1],expected[1])
