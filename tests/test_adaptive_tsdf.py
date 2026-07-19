import torch

from matcha.dm_extractors.adaptive_tsdf import (
    get_interpolated_value_from_pixel_coordinates,
)


def test_empty_projection_returns_empty_values():
    image = torch.zeros((4, 6, 3))
    pixels = torch.empty((0, 2))

    sampled = get_interpolated_value_from_pixel_coordinates(image, pixels)

    assert sampled.shape == (0, 3)
    assert sampled.dtype == image.dtype
    assert sampled.device == image.device
