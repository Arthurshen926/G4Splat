import pytest
import torch
from scripts.canopy_gradient_accumulation import accumulation_window


def test_partial_final_window_uses_actual_count_and_consumes_every_image_once():
    assert [accumulation_window(i, 6, 4) for i in range(1, 7)] == [
        (True, False, 4), (False, False, 4), (False, False, 4), (False, True, 4),
        (True, False, 2), (False, True, 2)]


@pytest.mark.parametrize('batch', [1, 4])
def test_sequential_backward_matches_mean_loss_without_retaining_graphs(batch):
    images = torch.tensor([1., -2., 3., .5, -1., 4.])
    p = torch.nn.Parameter(torch.tensor(.2)); reference = p.detach().clone().requires_grad_()
    optimizer = torch.optim.Adam([p], lr=.01)
    expected_optimizer = torch.optim.Adam([reference], lr=.01)
    window = []
    for step, value in enumerate(images, 1):
        first, last, count = accumulation_window(step, len(images), batch)
        if first:
            optimizer.zero_grad(set_to_none=True); window = []
        ((p-value).square()/count).backward(); window.append(value)
        if last:
            expected_optimizer.zero_grad(set_to_none=True)
            (reference-torch.stack(window)).square().mean().backward()
            torch.testing.assert_close(p.grad, reference.grad)
            optimizer.step(); expected_optimizer.step()
            torch.testing.assert_close(p, reference)


def test_invalid_window_rejected():
    with pytest.raises(ValueError): accumulation_window(0, 8, 4)
    with pytest.raises(ValueError): accumulation_window(1, 8, 9)
