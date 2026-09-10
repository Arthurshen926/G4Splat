import torch
from scripts.extract_canopy_render_state import extract_state


def test_deferred_extraction_preserves_learned_values_views_and_parameters(tmp_path):
    root=torch.arange(30,dtype=torch.float32).reshape(10,3)
    source=tmp_path/"checkpoint.pth"
    payload={"iteration":3,"surface":(torch.nn.Parameter(root),root[2:8:2]),
        "foliage":{"xyz":root.clone(),"ids":torch.arange(10,dtype=torch.int64)},
        "sky":{"color":torch.tensor([.1,.2,.3]),"missing_uv":torch.tensor([float('nan'),float('inf'),-0.])},
        "static_ray_birth_state":{"cells":[{"center":torch.rand(3)} for _ in range(100)]},
        "volume_optimizer":{"exp_avg":torch.ones(10,3)}}
    torch.save(payload,source)
    original=source.read_bytes();actual=extract_state(source)
    assert "static_ray_birth_state" not in actual and "volume_optimizer" not in actual
    assert torch.equal(actual['surface'][0],payload['surface'][0])
    assert isinstance(actual['surface'][0],torch.nn.Parameter)
    assert actual['surface'][0].requires_grad
    assert torch.equal(actual['surface'][1],payload['surface'][1])
    assert actual['surface'][1].stride()==payload['surface'][1].stride()
    assert torch.equal(actual['foliage']['xyz'],root)
    assert torch.equal(actual['foliage']['ids'],torch.arange(10))
    assert torch.equal(actual['sky']['color'],payload['sky']['color'])
    assert actual['sky']['missing_uv'].numpy().tobytes()==payload['sky']['missing_uv'].numpy().tobytes()
    assert actual['render_state_extraction']['loaded_storage_records']==5
    assert source.read_bytes()==original
    output=tmp_path/'render_only.pth';torch.save(actual,output)
    restored=torch.load(output)
    assert torch.equal(restored['surface'][1],payload['surface'][1])
