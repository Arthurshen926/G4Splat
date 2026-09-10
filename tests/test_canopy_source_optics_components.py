from types import SimpleNamespace
import pytest
import torch
from scripts.canopy_source_optics_components import source_optics_components


def test_component_swaps_preserve_original_and_restore_refined_even_after_error():
    original = SimpleNamespace(features=torch.zeros(1, 2, 3), opacity_logits=torch.zeros(1, 1))
    refined = SimpleNamespace(features=torch.ones(1, 2, 3), opacity_logits=torch.ones(1, 1))
    snapshots = []
    def render():
        snapshots.append((refined.features.clone(), refined.opacity_logits.clone()))
        return torch.zeros(3, 1, 1)
    target = torch.zeros(3, 1, 1); regions = dict(tree=torch.ones(1, 1, dtype=torch.bool))
    result = source_optics_components(original, refined, render, target, regions)
    assert set(result) == {'dc_only', 'directional_sh_only', 'opacity_only'}
    assert snapshots[0][0][:, 0].sum() == 3 and snapshots[0][0][:, 1:].sum() == 0
    assert snapshots[1][0][:, 0].sum() == 0 and snapshots[1][0][:, 1:].sum() == 3
    assert snapshots[2][0].sum() == 0 and snapshots[2][1].sum() == 1
    def fail(): raise RuntimeError('interrupted probe')
    with pytest.raises(RuntimeError): source_optics_components(original, refined, fail, target, regions)
    assert refined.features.sum() == 6 and refined.opacity_logits.sum() == 1
    assert original.features.sum() == 0 and original.opacity_logits.sum() == 0
