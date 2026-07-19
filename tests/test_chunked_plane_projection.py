from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "2d-gaussian-splatting"))

from planes import merge_global_3Dplane as plane_merge  # noqa: E402


class FakeCamera:
    image_height = 2
    image_width = 2


def test_chunked_plane_projection_keeps_only_front_surface_points(monkeypatch):
    # Columns encode depth, projected x, and projected y for the fake projector.
    points = torch.tensor(
        [
            [1.000, 0.0, 0.0],
            [1.005, 0.0, 0.0],
            [1.020, 0.0, 0.0],
            [2.000, 1.0, 0.0],
            [-1.00, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    plane_mask = torch.tensor([[1, 2], [0, 3]], dtype=torch.int32)

    def fake_project(_camera, point_chunk):
        depth = point_chunk[:, 0]
        coords = point_chunk[:, 1:3]
        in_image = (
            (coords[:, 0] >= 0)
            & (coords[:, 0] < 2)
            & (coords[:, 1] >= 0)
            & (coords[:, 1] < 2)
        )
        return depth, coords, in_image

    monkeypatch.setattr(plane_merge, "project_points_to_image", fake_project)

    result = plane_merge.get_plane_point_indices_chunked(
        FakeCamera(),
        points,
        plane_mask,
        depth_thresh=0.01,
        chunk_size=2,
    )
    assert result[1].tolist() == [0, 1]
    assert result[2].tolist() == [3]
    assert 3 not in result

    requested = plane_merge.get_plane_point_indices_chunked(
        FakeCamera(),
        points,
        plane_mask,
        plane_ids=[2],
        depth_thresh=0.01,
        chunk_size=2,
    )
    assert set(requested) == {2}
    assert requested[2].tolist() == [3]
