import numpy as np
import pytest
import torch
from outdoor.moge3_evidence import sha256_file
from scripts.canopy_batch_prefix_constraints import select_prefix_samples,load_batch_prefix_records

R=dict(target=[.1]*3,wall_rgb=[1.]*3)

def test_empty_and_no_deficit():
    assert select_prefix_samples([],np.empty((0,3)),R)==[]
    assert select_prefix_samples([1],[[.1]*3],R)==[]

def test_ties_use_complete_prefix_and_endpoints():
    samples=select_prefix_samples([1,1,2,3],[[.2]*3,[.3]*3,[.4]*3,[.5]*3],R,maximum=2)
    assert samples==[dict(depth=1.,surface_rgb=[.3]*3),dict(depth=3.,surface_rgb=[.5]*3)]

@pytest.mark.parametrize('depth,rgb', [([2,1],[[.2]*3,[.3]*3]),([0],[[.2]*3]),([1,2],[[.3]*3,[.2]*3]),([1],[[float('nan')]*3])])
def test_reject_bad_prefix(depth,rgb):
    with pytest.raises(ValueError):select_prefix_samples(depth,rgb,R)


def test_payload_identity_and_completeness(tmp_path):
    ray=dict(R,view=1,x=2,y=3,kind='occlusion')
    entry=dict(ray=ray,depths=torch.tensor([1.]),cumulative_rgb=torch.tensor([[.3]*3]))
    payload=tmp_path/'surface_prefixes.pth'
    torch.save(dict(prefixes=[entry],source_checkpoint_sha256='source'),payload)
    manifest=dict(source_checkpoint_sha256='source',excluded_views=[2])
    old=dict(training_views=[1],rays=[ray])
    report=dict(source_checkpoint_sha256='source',training_views=[1],excluded_views=[2],
                maximum_native_rgb_error=1e-7,prefixes_sha256=sha256_file(payload))
    path=tmp_path/'surface_prefix_batch.json'
    assert len(load_batch_prefix_records(path,report,old,manifest))==1
    with pytest.raises(ValueError,match='provenance'):
        load_batch_prefix_records(path,{**report,'prefixes_sha256':'wrong'},old,manifest)
    with pytest.raises(ValueError,match='provenance'):
        load_batch_prefix_records(path,{**report,'excluded_views':[3]},old,manifest)
    for entries in ([],[entry,entry],[{**entry,'ray':{**ray,'x':9}}]):
        torch.save(dict(prefixes=entries,source_checkpoint_sha256='source'),payload)
        report['prefixes_sha256']=sha256_file(payload)
        with pytest.raises(ValueError):load_batch_prefix_records(path,report,old,manifest)
