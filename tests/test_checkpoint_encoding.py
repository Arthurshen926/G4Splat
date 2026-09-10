import io

import torch

from outdoor.checkpoint_encoding import compact_static_birth_vectors
from outdoor.static_ray_birth import StaticRayBirthAccumulator


def test_compact_birth_state_preserves_resume_drain_and_input():
    accumulator = StaticRayBirthAccumulator(voxel_size=.04,visual_hull_voxel_size=.08,
                                           minimum_birth_separation=.04)
    for camera in (1,2):
        accumulator.add({"centers":torch.tensor([[.2,.2,5.]]),
            "confidence":torch.tensor([.8]),"colors":torch.tensor([[.1,.2,.3]]),
            "camera_id":camera,"origins":torch.tensor([[0.,0.,0.]]),
            "directions":torch.tensor([[.2,.2,5.]]),
            "hit_start":torch.tensor([4.8]),"hit_end":torch.tensor([5.2]),
            "single_first_hit":True},sequence_id="seq2")
    original=accumulator.capture()
    encoded=compact_static_birth_vectors(original)
    assert torch.is_tensor(original["visual_hull_cells"][0]["weighted_center"])
    assert isinstance(encoded["visual_hull_cells"][0]["weighted_center"],list)
    for a,b in zip(original["visual_hull_cells"],encoded["visual_hull_cells"]):
        for name in ("weighted_center","weighted_color"):
            assert torch.equal(a[name],torch.tensor(b[name],dtype=a[name].dtype))
        assert a["first_hit_witnesses"]==b["first_hit_witnesses"]
        assert a["camera_sequences"]==b["camera_sequences"]
    buffer=io.BytesIO();torch.save(encoded,buffer);buffer.seek(0)
    restored=StaticRayBirthAccumulator(voxel_size=.04,visual_hull_voxel_size=.08,
                                      minimum_birth_separation=.04)
    restored.restore(torch.load(buffer))
    expected=accumulator.drain(maximum_births=20,minimum_sequences=1)
    actual=restored.drain(maximum_births=20,minimum_sequences=1)
    for a,b in zip(expected[:4],actual[:4]):
        assert torch.equal(a,b)
    assert expected[4]==actual[4]
    assert restored.consumed_first_hit_witnesses==accumulator.consumed_first_hit_witnesses


def test_compact_birth_vectors_preserve_float32_extremes_exactly():
    value=torch.tensor([torch.finfo(torch.float32).tiny,-1e30,1.0000001192092896])
    source={"cells":[{"weighted_center":value,"weighted_color":value.clone()}]}
    encoded=compact_static_birth_vectors(source)
    assert torch.equal(value,torch.tensor(encoded["cells"][0]["weighted_center"]))


def test_compact_checkpoint_cli_writes_real_atomic_file(tmp_path,monkeypatch):
    from scripts.compact_static_birth_checkpoint import main
    source=tmp_path/"input.pth";output=tmp_path/"output.pth"
    value=torch.arange(12).reshape(4,3).float()
    torch.save({"iteration":7,"model":value,"static_ray_birth_state":{
        "cells":[{"weighted_center":value[0],"weighted_color":value[1]}]}},source)
    original=source.read_bytes()
    monkeypatch.setattr("sys.argv",["compact","--input",str(source),"--output",str(output),"--expected-iteration","7"])
    main()
    result=torch.load(output)
    assert torch.equal(result["model"],value)
    assert source.read_bytes()==original
    assert isinstance(result["static_ray_birth_state"]["cells"][0]["weighted_center"],list)
    assert not list(tmp_path.glob(".compact-checkpoint-*"))


def test_production_atomic_checkpoint_uses_lossless_encoding_without_input_mutation(tmp_path):
    from scripts.train_unified_outdoor_teacher import _save_checkpoint
    vector=torch.tensor([.1,.2,.3])
    payload={"iteration":11,"model":torch.arange(4),"static_ray_birth_state":{
        "cells":[{"weighted_center":vector,"weighted_color":vector.clone()}]}}
    path=tmp_path/"checkpoint.pth"
    _save_checkpoint(path,payload)
    result=torch.load(path)
    assert torch.equal(result["model"],payload["model"])
    assert torch.equal(torch.tensor(result["static_ray_birth_state"]["cells"][0]["weighted_center"]),vector)
    assert torch.is_tensor(payload["static_ray_birth_state"]["cells"][0]["weighted_center"])
    assert result["checkpoint_io_encoding"]["model_optimizer_rng_schedule_and_training_contract_unchanged"]
