import torch
from scripts.canopy_source_opacity_projection import project_source_opacity_


def test_frozen_opacity_is_not_changed_by_post_step_projection():
    value = torch.nn.Parameter(torch.tensor([[-30.], [30.], [0.]]))
    original = value.detach().clone(); eligible = torch.tensor([True, True, False])
    optimizer = torch.optim.Adam([value], lr=0., eps=1e-15)
    for _ in range(3):
        value.grad = torch.ones_like(value); optimizer.step()
        project_source_opacity_(value, eligible, enabled=False)
        assert torch.equal(value, original)


def test_enabled_projection_preserves_unauthorized_rows():
    value = torch.tensor([[-30.], [30.], [30.]])
    project_source_opacity_(value, torch.tensor([True, True, False]), enabled=True)
    assert value[0] > -30 and value[1] < 30 and value[2] == 30
