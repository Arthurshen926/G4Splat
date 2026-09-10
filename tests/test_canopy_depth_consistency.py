from types import SimpleNamespace
import torch
from scripts.audit_canopy_depth_consistency import reproject_depth_pixels


def camera(tx=0.):
    transform = torch.eye(4)
    transform[3, 0] = tx
    return SimpleNamespace(cx=19.2, cy=7.3, focal_x=100., focal_y=120.,
                           world_view_transform=transform)


def test_exact_k_identity_projection():
    uv = torch.tensor([[0.,0.], [19.2,7.3], [50.,40.]])
    depth = torch.tensor([1., 5., 20.])
    actual, z = reproject_depth_pixels(uv, depth, camera(), camera())
    torch.testing.assert_close(actual, uv, atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(z, depth)


def test_translation_disparity_and_roundtrip():
    uv = torch.tensor([[30.,20.], [40.,10.]])
    depth = torch.tensor([2.,10.])
    projected, z = reproject_depth_pixels(uv, depth, camera(), camera(-1.))
    expected = uv.clone(); expected[:,0] -= 100./depth
    torch.testing.assert_close(projected, expected)
    recovered, original_z = reproject_depth_pixels(projected, z, camera(-1.), camera())
    torch.testing.assert_close(recovered, uv)
    torch.testing.assert_close(original_z, depth)
