import numpy as np
import torch

from outdoor.foliage_geometry import semantic_tree_tracks
from outdoor.foliage_view_graph import (
    greedy_diverse_views,
    sequence_id,
    support_geometry,
)


def test_sequence_id_is_not_frame_identity():
    assert sequence_id("seq7__frame00042.png") == "seq7"
    assert sequence_id("seq9/frame00001.png") == "seq9"


def test_support_geometry_requires_real_sequence_and_baseline_diversity():
    centers = {
        1: np.array([0.0, 0.0, 0.0]),
        2: np.array([1.0, 0.0, 0.0]),
        3: np.array([0.0, 2.0, 0.0]),
    }
    sequences = {1: "seq1", 2: "seq1", 3: "seq2"}

    result = support_geometry(
        np.array([0.0, 0.0, 5.0]), [1, 2, 3], centers, sequences
    )

    assert result["camera_count"] == 3
    assert result["sequence_count"] == 2
    assert result["max_baseline"] > 2
    assert result["max_triangulation_angle_degrees"] > 20


def test_diverse_view_selection_penalizes_repeated_sequences():
    records = [
        {
            "image_name": "seq1__a",
            "sequence_id": "seq1",
            "canopy_fraction": 0.9,
            "camera_center": np.array([0.0, 0.0, 0.0]),
        },
        {
            "image_name": "seq1__b",
            "sequence_id": "seq1",
            "canopy_fraction": 0.89,
            "camera_center": np.array([0.01, 0.0, 0.0]),
        },
        {
            "image_name": "seq2__a",
            "sequence_id": "seq2",
            "canopy_fraction": 0.8,
            "camera_center": np.array([2.0, 0.0, 0.0]),
        },
    ]

    selected = greedy_diverse_views(records, limit=2, minimum_center_distance=0.5)

    assert {row["sequence_id"] for row in selected} == {"seq1", "seq2"}


def test_empty_cambridge_point2d_rows_use_tracked_calibrated_reprojection():
    mask = torch.ones((10, 10), dtype=torch.bool)
    mask[5, 5] = False

    class MaskLookup:
        masks = {"seq1/a.png": (mask, mask, mask, mask), "seq2/a.png": (mask, mask, mask, mask)}

        @staticmethod
        def source_name_for(name):
            return name

    images = {
        image_id: {
            "id": image_id,
            "name": name,
            "qvec": np.array([1.0, 0.0, 0.0, 0.0]),
            "tvec": np.zeros(3),
            "camera_id": 1,
            "xys": np.empty((0, 2)),
        }
        for image_id, name in [(1, "seq1/a.png"), (2, "seq1/a.png"), (3, "seq2/a.png")]
    }
    cameras = {
        1: {
            "model": "PINHOLE",
            "width": 10,
            "height": 10,
            "params": np.array([10.0, 10.0, 5.0, 5.0]),
        }
    }
    points = [
        {
            "id": 7,
            "xyz": np.array([0.0, 0.0, 5.0]),
            "rgb": np.array([20, 80, 30], dtype=np.uint8),
            "error": 0.2,
            "image_ids": np.array([1, 2, 3]),
            "point2d_indices": np.array([0, 0, 0]),
        }
    ]

    accepted = semantic_tree_tracks(
        points, images, cameras, MaskLookup(), minimum_tree_fraction=1.0
    )

    assert len(accepted) == 1
    assert accepted[0]["tree_sequence_count"] == 2
    assert accepted[0]["tree_image_ids"].tolist() == [1, 2, 3]
