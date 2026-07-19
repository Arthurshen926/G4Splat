from pathlib import Path

from scripts.propose_chart_reselection import build_reselection


def test_reselection_preserves_good_charts_and_replaces_rejected(tmp_path: Path):
    scene = tmp_path / "scene"
    (scene / "images").mkdir(parents=True)
    sparse = scene / "sparse" / "0"
    sparse.mkdir(parents=True)
    names = [f"frame{i:03d}.png" for i in range(8)]
    for name in names:
        (scene / "images" / name).touch()
    lines = ["# Image list"]
    for index, name in enumerate(names, start=1):
        # Identity rotation; t=-center gives a trajectory along x.
        lines.extend([f"{index} 1 0 0 0 {-float(index)} 0 0 1 {name}", ""])
    (sparse / "images.txt").write_text("\n".join(lines))

    selection = {
        "scene_path": str(scene),
        "n_images": 3,
        "image_idx": [0, 3, 6],
        "image_names": [names[0], names[3], names[6]],
        "candidate_pool_names": names,
        "coverage": {
            "coverage_objective": "target_kcenter",
            "view_clusters": 1,
            "min_views_per_cluster": 1,
            "min_baseline_ratio": 0.0,
            "min_global_pose_distance": 0.01,
            "max_borrowed_support_distance": 1.0,
        },
    }
    gate = {
        "records": [
            {"image_name": name, "rejected": name == names[3]} for name in selection["image_names"]
        ]
    }

    result = build_reselection(selection, gate, scene)

    assert names[0] in result["image_names"]
    assert names[6] in result["image_names"]
    assert names[3] not in result["image_names"]
    assert len(result["image_names"]) == 3
