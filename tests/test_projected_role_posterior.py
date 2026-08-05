import json
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from outdoor.projected_role_posterior import (
    PROJECTED_RIGID_POSTERIOR_VERSION,
    build_projected_rigid_conflict_posterior,
)


def test_all_camera_projection_records_mask_conflict_support_and_visibility(
    tmp_path,
):
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    image_name = "seq1__frame00001.png"
    source_name = "seq1/frame00001.png"
    (dataset / "name_mapping.json").write_text(
        json.dumps({image_name: source_name})
    )
    rgb = np.full((2, 2, 3), 100, dtype=np.uint8)
    Image.fromarray(rgb).save(images / image_name)
    masks = (
        torch.tensor([[False, True], [True, True]]),
        torch.ones((2, 2), dtype=torch.bool),
        torch.ones((2, 2), dtype=torch.bool),
        torch.ones((2, 2), dtype=torch.bool),
    )
    mask_pickle = tmp_path / "masks.pkl"
    with mask_pickle.open("wb") as handle:
        pickle.dump({source_name: masks}, handle)
    scene = tmp_path / "scene_contract.json"
    scene.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "image_name": image_name,
                        "T_world_to_camera": np.eye(4).tolist(),
                        "camera": {
                            "width": 2,
                            "height": 2,
                            "fx": 1.0,
                            "fy": 1.0,
                            "cx": 0.0,
                            "cy": 0.0,
                        },
                    }
                ]
            }
        )
    )
    tracks = tmp_path / "tracks.npz"
    role = np.zeros((1, 5), dtype=np.float32)
    role[:, 0] = 1.0
    np.savez_compressed(
        tracks,
        xyz=np.asarray([[0.1, 0.1, 1.0]], dtype=np.float32),
        rgb=np.asarray([[100, 100, 100]], dtype=np.uint8),
        track_id=np.asarray([17], dtype=np.int64),
        role_probabilities=role,
        valid_observation_count=np.asarray([3], dtype=np.int16),
        sequence_count=np.asarray([2], dtype=np.int16),
        reprojection_error=np.asarray([0.1], dtype=np.float32),
        triangulation_angle_median=np.asarray([10.0], dtype=np.float32),
        cycle_consistency=np.asarray([0.85], dtype=np.float32),
    )
    output = tmp_path / "projected.npz"
    summary = build_projected_rigid_conflict_posterior(
        tracks,
        scene,
        dataset,
        mask_pickle,
        output,
    )

    assert summary["schema_version"] == PROJECTED_RIGID_POSTERIOR_VERSION
    assert summary["camera_count"] == 1
    assert summary["projected_conflict_observation_count"] == 1
    with np.load(output, allow_pickle=False) as archive:
        assert archive["offsets"].tolist() == [0, 1]
        assert archive["pixels"].tolist() == [[0, 0]]
        assert float(archive["geometry_support_probability"][0]) > 0.5
        assert np.isclose(
            archive["visible_observation_probability"][0],
            archive["geometry_support_probability"][0],
            atol=2e-3,
        )


def test_tree_occluded_projection_keeps_geometry_without_inventing_visibility(
    tmp_path,
):
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    image_name = "seq1__frame00001.png"
    source_name = "seq1/frame00001.png"
    (dataset / "name_mapping.json").write_text(
        json.dumps({image_name: source_name})
    )
    Image.fromarray(np.full((2, 2, 3), 100, dtype=np.uint8)).save(
        images / image_name
    )
    mask_pickle = tmp_path / "masks.pkl"
    with mask_pickle.open("wb") as handle:
        pickle.dump(
            {
                source_name: (
                    torch.ones((2, 2), dtype=torch.bool),
                    torch.ones((2, 2), dtype=torch.bool),
                    torch.ones((2, 2), dtype=torch.bool),
                    torch.tensor([[False, True], [True, True]]),
                )
            },
            handle,
        )
    scene = tmp_path / "scene_contract.json"
    scene.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "image_name": image_name,
                        "T_world_to_camera": np.eye(4).tolist(),
                        "camera": {
                            "width": 2,
                            "height": 2,
                            "fx": 1.0,
                            "fy": 1.0,
                            "cx": 0.0,
                            "cy": 0.0,
                        },
                    }
                ]
            }
        )
    )
    tracks = tmp_path / "tracks.npz"
    role = np.zeros((1, 5), dtype=np.float32)
    role[:, 0] = 1.0
    np.savez_compressed(
        tracks,
        xyz=np.asarray([[0.1, 0.1, 1.0]], dtype=np.float32),
        rgb=np.asarray([[100, 100, 100]], dtype=np.uint8),
        track_id=np.asarray([17], dtype=np.int64),
        role_probabilities=role,
        valid_observation_count=np.asarray([3], dtype=np.int16),
        sequence_count=np.asarray([2], dtype=np.int16),
        reprojection_error=np.asarray([0.1], dtype=np.float32),
        triangulation_angle_median=np.asarray([10.0], dtype=np.float32),
        cycle_consistency=np.asarray([0.85], dtype=np.float32),
    )
    output = tmp_path / "projected.npz"
    summary = build_projected_rigid_conflict_posterior(
        tracks, scene, dataset, mask_pickle, output
    )

    assert summary["tree_occlusion_observation_count"] == 1
    assert summary["tree_occlusion_exact_observation_count"] == 0
    assert summary["tree_occlusion_geometry_support_mass"] > 0.5
    assert summary["tree_occlusion_visible_observation_mass"] == 0.0
    with np.load(output, allow_pickle=False) as archive:
        assert float(archive["geometry_support_probability"][0]) > 0.5
        assert float(archive["visible_observation_probability"][0]) == 0.0

    # The same projected row may use current-view RGB only after the source
    # archive proves that this exact track was really observed by this exact
    # camera (rather than merely projecting behind its coarse tree mask).
    exact_tracks = tmp_path / "exact_tracks.npz"
    np.savez_compressed(
        exact_tracks,
        xyz=np.asarray([[0.1, 0.1, 1.0]], dtype=np.float32),
        rgb=np.asarray([[100, 100, 100]], dtype=np.uint8),
        track_id=np.asarray([17], dtype=np.int64),
        role_probabilities=role,
        valid_observation_count=np.asarray([3], dtype=np.int16),
        sequence_count=np.asarray([2], dtype=np.int16),
        reprojection_error=np.asarray([0.1], dtype=np.float32),
        triangulation_angle_median=np.asarray([10.0], dtype=np.float32),
        cycle_consistency=np.asarray([0.85], dtype=np.float32),
        observation_offsets=np.asarray([0, 1], dtype=np.int64),
        observation_camera_indices=np.asarray([0], dtype=np.int32),
        camera_names=np.asarray([Path(image_name).stem]),
    )
    exact_output = tmp_path / "exact_projected.npz"
    exact_summary = build_projected_rigid_conflict_posterior(
        exact_tracks, scene, dataset, mask_pickle, exact_output
    )
    assert exact_summary["tree_occlusion_exact_observation_count"] == 1
    assert exact_summary["tree_occlusion_visible_observation_mass"] > 0.5
