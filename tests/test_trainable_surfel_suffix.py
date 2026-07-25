import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from outdoor.trainable_surfel_suffix import TrainableSurfelSuffix


class _Structural:
    max_sh_degree = 1
    active_sh_degree = 1
    use_mip_filter = False

    def __init__(self):
        self._xyz = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        self._features_dc = torch.zeros(5, 1, 3)
        self._features_rest = torch.zeros(5, 3, 3)
        self._opacity = torch.zeros(5, 1)
        self._scaling = torch.zeros(5, 2)
        self._rotation = torch.tensor([[1.0, 0, 0, 0]]).repeat(5, 1)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation, dim=-1)

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)


def test_only_suffix_owns_gradient_and_writes_back():
    structural = _Structural()
    adapter = TrainableSurfelSuffix(structural, 3)
    adapter.get_xyz.sum().backward()
    assert adapter._xyz.grad.shape == (2, 3)
    adapter._xyz.data.add_(2)
    adapter.write_back()
    assert torch.equal(structural._xyz[:3], torch.arange(9).reshape(3, 3))
    assert torch.equal(
        structural._xyz[3:], torch.arange(9, 15).reshape(2, 3) + 2
    )


def test_projection_bounds_suffix():
    adapter = TrainableSurfelSuffix(_Structural(), 3)
    adapter._xyz.data.add_(10)
    adapter._opacity.data.fill_(10)
    adapter._scaling.data.fill_(10)
    adapter._features_rest.data.fill_(1)
    adapter.project(
        maximum_opacity=0.3,
        maximum_scale=0.2,
        maximum_position_delta=0.1,
    )
    assert torch.all(torch.sigmoid(adapter._opacity) <= 0.300001)
    assert torch.all(torch.exp(adapter._scaling) <= 0.200001)
    assert torch.all(
        torch.linalg.vector_norm(
            adapter._xyz - adapter.initial_xyz, dim=-1
        )
        <= 0.100001
    )
    assert torch.count_nonzero(adapter._features_rest) == 0
