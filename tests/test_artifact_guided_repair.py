import cv2
import numpy as np
import struct

from artifact_guided_repair import (
    ArtifactROI3D,
    ArtifactROI3D,
    CameraGeometry,
    backproject_depth,
    blend_projective_reference_colors,
    build_anomaly_rays,
    build_artifact_components,
    cluster_attributed_primitives,
    counterfactual_causal_rays,
    fit_support_plane,
    interpolate_c2w,
    multiview_clean_visible_points,
    multiview_depth_supported_points,
    project_points,
    rasterize_projected_component,
    render_support_plane_depth,
    robust_affine_color,
    sample_matcha_chart_surface_points,
    sample_colmap_track_surface_points,
    select_conservative_edit_trial,
    select_diverse_full_train_anomaly_targets,
    select_full_train_anomaly_targets,
    select_wrong_geometry_interpolation,
    see3d_gate,
    virtual_reprojection_consistency,
)
from artifact_guided_repair.causal import PrimitiveEvidence


def camera(name="camera", tx=0.0):
    w2c = np.eye(4)
    w2c[0, 3] = -tx
    return CameraGeometry(
        name=name,
        width=80,
        height=60,
        w2c=w2c,
        fx=60.0,
        fy=60.0,
        cx=40.0,
        cy=30.0,
    )


def test_affine_color_removes_exposure_difference():
    yy, xx = np.mgrid[:40, :60]
    render = np.stack([xx / 60.0, yy / 40.0, (xx + yy) / 100.0], axis=2).astype(np.float32)
    target = np.clip(render * np.asarray([1.2, 0.8, 1.1]) + 0.05, 0.0, 1.0)
    corrected, params = robust_affine_color(render, target, np.ones((40, 60), dtype=bool))

    assert np.mean(np.abs(corrected - target)) < 0.01
    assert params.shape == (3, 2)


def test_backprojection_round_trips_through_camera():
    cam = camera()
    depth = np.full((60, 80), 4.0, dtype=np.float32)
    mask = np.zeros_like(depth, dtype=bool)
    mask[20:40:4, 30:50:4] = True

    points, pixels = backproject_depth(depth, mask, cam, stride=1)
    projected, z, inside = project_points(points, cam)

    assert inside.all()
    assert np.allclose(projected, pixels)
    assert np.allclose(z, 4.0)


def test_roi_persists_numpy_evidence_metadata(tmp_path):
    roi = ArtifactROI3D(
        roi_id="numpy_evidence",
        points=np.asarray([[0.0, 0.0, 4.0], [0.2, 0.1, 4.0]]),
        source_view_ids=(),
        classification="surface_misalignment_or_appearance",
        plane_world=np.asarray([0.0, 0.0, 1.0, -4.0]),
        evidence={
            "support_indices": np.asarray([2, 5], dtype=np.int64),
            "nested": {"residual": np.float32(0.125)},
        },
    )

    restored = ArtifactROI3D.load(roi.save(tmp_path / "roi.npz"))

    assert restored.evidence["support_indices"] == [2, 5]
    assert restored.evidence["nested"]["residual"] == 0.125


def test_matcha_chart_evidence_requires_coordinate_and_multichart_support(tmp_path):
    """A 2-D anomaly mask becomes surface evidence only after two real Charts agree."""
    chart_height, chart_width = 10, 20
    yy, xx = np.mgrid[:chart_height, :chart_width]
    # The two Chart point maps represent the same z=4 plane in the target
    # camera frame.  Their native raster is 20x10 while the real images are
    # 80x60, exercising the coordinate-audit scale conversion.
    projected_x = (xx + 0.5) * 80.0 / chart_width - 0.5
    projected_y = (yy + 0.5) * 60.0 / chart_height - 0.5
    point_map = np.stack(
        [
            (projected_x - 40.0) * 4.0 / 60.0,
            (projected_y - 30.0) * 4.0 / 60.0,
            np.full_like(projected_x, 4.0),
        ],
        axis=2,
    ).astype(np.float32)
    np.savez_compressed(
        tmp_path / "charts_data.npz",
        pts=np.stack([point_map, point_map]),
        depths=np.full((2, chart_height, chart_width), 4.0, dtype=np.float32),
        confs=np.full((2, chart_height, chart_width), 2.0, dtype=np.float32),
        scale_factor=np.asarray(1.0),
        alignment_gate_valid=np.asarray([True, True]),
    )
    (tmp_path / "cameras.json").write_text(
        '{"filepaths": ["chart_a.png", "chart_b.png"]}', encoding="utf-8"
    )
    target = camera("target")
    mask = np.ones((target.height, target.width), dtype=bool)
    points, report = sample_matcha_chart_surface_points(
        tmp_path,
        target,
        mask,
        {"chart_a": camera("chart_a"), "chart_b": camera("chart_b")},
        source_static_masks={
            "chart_a": np.ones((60, 80), dtype=bool),
            "chart_b": np.ones((60, 80), dtype=bool),
        },
        chart_stride=1,
        coordinate_audit_stride=1,
        target_cell_pixels=8,
        max_points=500,
    )

    assert report["accepted"]
    assert report["source_semantic_mask_applied"]
    assert report["coordinate_audit_passing_chart_count"] == 2
    assert report["returned_source_chart_count"] == 2
    assert len(points) > 20


def test_original_colmap_tracks_are_coordinate_audited_before_surface_use(tmp_path):
    """A target ray needs one target + two static real track observations."""
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    target = camera("target", tx=0.0)
    support_a = camera("support_a", tx=0.5)
    support_b = camera("support_b", tx=-0.5)
    cameras = {item.name: item for item in (target, support_a, support_b)}
    points_world = np.asarray(
        [[x, y, 4.0] for x in (-0.2, 0.0, 0.2) for y in (-0.2, 0.0, 0.2)]
    )
    observed = {
        name: project_points(points_world, value)[0]
        for name, value in cameras.items()
    }

    with (sparse / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<iiQQ", 1, 1, 80, 60))  # PINHOLE
        handle.write(struct.pack("<dddd", 60.0, 60.0, 40.0, 30.0))
    with (sparse / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 3))
        for image_id, name in enumerate(("target", "support_a", "support_b"), start=1):
            tvec = cameras[name].w2c[:3, 3]
            handle.write(struct.pack("<idddddddi", image_id, 1.0, 0.0, 0.0, 0.0, *tvec, 1))
            handle.write(f"{name}.png".encode("utf-8") + b"\x00")
            handle.write(struct.pack("<Q", len(points_world)))
            for point_id, pixel in enumerate(observed[name], start=1):
                handle.write(struct.pack("<ddq", *pixel, point_id))
    with (sparse / "points3D.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(points_world)))
        for point_id, point in enumerate(points_world, start=1):
            handle.write(struct.pack("<QdddBBBd", point_id, *point, 100, 100, 100, 0.1))
            handle.write(struct.pack("<Q", 3))
            handle.write(struct.pack("<iiiiii", 1, point_id - 1, 2, point_id - 1, 3, point_id - 1))

    rays = np.zeros((60, 80), dtype=bool)
    xy = np.rint(observed["target"]).astype(int)
    rays[xy[:, 1], xy[:, 0]] = True
    points, report = sample_colmap_track_surface_points(
        sparse,
        target,
        rays,
        cameras,
        source_static_masks={name: np.ones((60, 80), dtype=bool) for name in cameras},
    )

    assert report["accepted"]
    assert report["support_count_p50"] == 2.0
    assert report["coordinate_audit_reprojection_p90_pixels"] < 1e-6
    assert np.allclose(np.sort(points, axis=0), np.sort(points_world, axis=0))


def test_colmap_track_projection_fallback_handles_exports_without_xys(tmp_path):
    """Some Cambridge sparse exports retain tracks but strip image xys arrays."""
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    target = camera("target", tx=0.0)
    support_a = camera("support_a", tx=0.5)
    support_b = camera("support_b", tx=-0.5)
    cameras = {item.name: item for item in (target, support_a, support_b)}
    points_world = np.asarray(
        [[x, y, 4.0] for x in (-0.2, 0.0, 0.2) for y in (-0.2, 0.0, 0.2)]
    )

    with (sparse / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<iiQQ", 1, 1, 80, 60))
        handle.write(struct.pack("<dddd", 60.0, 60.0, 40.0, 30.0))
    with (sparse / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 3))
        for image_id, name in enumerate(("target", "support_a", "support_b"), start=1):
            tvec = cameras[name].w2c[:3, 3]
            handle.write(struct.pack("<idddddddi", image_id, 1.0, 0.0, 0.0, 0.0, *tvec, 1))
            handle.write(f"{name}.png".encode("utf-8") + b"\x00")
            handle.write(struct.pack("<Q", 0))
    with (sparse / "points3D.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(points_world)))
        for point_id, point in enumerate(points_world, start=1):
            handle.write(struct.pack("<QdddBBBd", point_id, *point, 100, 100, 100, 0.1))
            handle.write(struct.pack("<Q", 3))
            handle.write(struct.pack("<iiiiii", 1, point_id - 1, 2, point_id - 1, 3, point_id - 1))

    points, report = sample_colmap_track_surface_points(
        sparse,
        target,
        np.ones((60, 80), dtype=bool),
        cameras,
        source_static_masks={name: np.ones((60, 80), dtype=bool) for name in cameras},
    )

    assert report["accepted"]
    assert report["target_track_selection_mode"] == "project_all_target_visible_tracks_to_anomaly_rays"
    assert report["coordinate_audit_reprojection_p90_pixels"] is None
    assert np.allclose(np.sort(points, axis=0), np.sort(points_world, axis=0))


def test_counterfactual_causal_rays_keep_a_tight_surface_core():
    baseline = np.zeros((20, 30, 3), dtype=np.float32)
    intervention = baseline.copy()
    # A cluster affects the full left patch, with a higher-confidence core.
    intervention[5:15, 3:13] = 0.04
    intervention[8:12, 6:10] = 0.30
    anomaly = np.zeros((20, 30), dtype=bool)
    anomaly[4:16, 2:14] = True

    result = counterfactual_causal_rays(
        baseline,
        intervention,
        anomaly,
        minimum_causal_pixels=8,
        minimum_surface_pixels=8,
    )

    assert result["causal_mask"][5:15, 3:13].all()
    assert result["surface_core_mask"].sum() < result["causal_mask"].sum()
    assert result["surface_core_mask"][8:12, 6:10].all()
    assert not result["causal_fallback_to_full_anomaly"]


def test_full_train_prescan_selection_is_deterministic_and_excludes_clean_views():
    records = [
        {"name": "seq10__frame00001", "anomaly_ray_pixel_count": 0, "priority": 0.99},
        {"name": "seq10__frame00002", "anomaly_ray_pixel_count": 48, "priority": 0.31},
        {"name": "seq11__frame00003", "anomaly_ray_pixel_count": 96, "priority": 0.31},
        {"name": "seq12__frame00004", "anomaly_ray_pixel_count": 80, "priority": 0.72},
    ]

    selected = select_full_train_anomaly_targets(records, count=3)

    assert selected == [
        "seq12__frame00004",
        "seq11__frame00003",
        "seq10__frame00002",
    ]


def test_diverse_full_train_prescan_selection_preserves_priority_but_avoids_adjacent_duplicates():
    records = [
        {"name": "seq10__frame00001", "anomaly_ray_pixel_count": 120, "priority": 0.98},
        {"name": "seq10__frame00002", "anomaly_ray_pixel_count": 118, "priority": 0.97},
        {"name": "seq10__frame00003", "anomaly_ray_pixel_count": 116, "priority": 0.96},
        {"name": "seq11__frame00001", "anomaly_ray_pixel_count": 110, "priority": 0.91},
        {"name": "seq12__frame00001", "anomaly_ray_pixel_count": 105, "priority": 0.89},
    ]
    # seq10 frames are nearly identical in real-camera pose/direction space;
    # the other two views preserve residual strength while adding coverage.
    features = {
        "seq10__frame00001": np.asarray([0.00, 0.00, 0.0]),
        "seq10__frame00002": np.asarray([0.01, 0.00, 0.0]),
        "seq10__frame00003": np.asarray([0.02, 0.00, 0.0]),
        "seq11__frame00001": np.asarray([1.00, 0.00, 0.0]),
        "seq12__frame00001": np.asarray([0.00, 1.00, 0.0]),
    }

    selected, audit = select_diverse_full_train_anomaly_targets(
        records, features, count=3, candidate_multiplier=4
    )

    assert selected[0] == "seq10__frame00001"
    assert {"seq11__frame00001", "seq12__frame00001"}.issubset(selected)
    assert [row["selection_step"] for row in audit] == [1, 2, 3]
    assert audit[0]["minimum_real_camera_feature_distance_to_previous_selection"] is None
    assert audit[1]["marginal_weighted_real_camera_coverage_gain"] > 0.5


def test_candidate_cluster_splits_a_long_same_view_single_linkage_chain():
    """One bad image must not join a facade-length candidate chain into one edit."""
    evidence = {}
    # Dense local links over a long extent emulate the failure mode seen on a
    # facade: every adjacent pair is close, while the transitive component is
    # much wider than one local intervention.
    for primitive_id in range(65):
        item = PrimitiveEvidence(primitive_id)
        item.add("bad_view", score=1.0, opacity_benefit=0.5, position_gradient=0.1)
        evidence[primitive_id] = item
    xyz = np.stack(
        [np.arange(65, dtype=np.float64) * 0.4, np.zeros(65), np.full(65, 4.0)], axis=1
    )
    normals = np.tile(np.asarray([[0.0, 0.0, 1.0]]), (65, 1))
    scales = np.full((65, 2), 0.1, dtype=np.float64)

    clusters, diagnostics = cluster_attributed_primitives(
        evidence,
        xyz,
        normals,
        scales,
        max_component_diameter_factor=1.5,
    )

    assert diagnostics["pre_compaction_component_count"] == 1.0
    assert diagnostics["spatial_chain_split_count"] >= 1.0
    assert len(clusters) > 1
    assert sum(item["primitive_count"] for item in clusters) == len(evidence)
    assert all(
        item["diameter_world"] <= diagnostics["max_component_diameter"] + 1e-8
        for item in clusters
    )


def test_depth_layer_ambiguity_is_a_weak_anomaly_audit_signal():
    """Renderer depth disagreement may rank a residual ray, never create one alone."""
    height, width = 60, 80
    target = np.full((height, width, 3), 0.55, dtype=np.float32)
    render = target.copy()
    alpha = np.ones((height, width), dtype=np.float32)
    depth = np.full((height, width), 4.0, dtype=np.float32)
    expected = depth.copy()
    median = depth.copy()
    median[18:44, 26:58] = 8.0
    keep = np.ones((height, width), dtype=bool)

    # Self-disagreement from the current model cannot by itself certify a bad
    # ray or a 3-D surface.
    neutral = build_anomaly_rays(
        render, target, alpha, depth, keep,
        expected_depth=expected, median_depth=median, min_component_pixels=16,
    )
    assert not neutral["mask"].any()

    render[20:42, 28:56] = 0.05
    residual = build_anomaly_rays(
        render, target, alpha, depth, keep,
        expected_depth=expected, median_depth=median, min_component_pixels=16,
    )
    assert residual["mask"][30, 40]
    assert residual["depth_layer_ambiguity"][30, 40] > 0.45
    assert residual["summary"]["mean_anomaly_depth_layer_ambiguity"] > 0.2


def test_see3d_gate_requires_known_surface_and_absent_real_or_model_appearance():
    """Observed training surfaces must never be escalated to generative RGB."""
    observed = see3d_gate(
        surface_known=True,
        real_rgb_support_fraction=1.0,
        reliable_model_fraction=0.0,
    )
    unknown_geometry = see3d_gate(
        surface_known=False,
        real_rgb_support_fraction=0.0,
        reliable_model_fraction=0.0,
    )
    residual = see3d_gate(
        surface_known=True,
        real_rgb_support_fraction=0.0,
        reliable_model_fraction=0.0,
    )

    assert not observed["eligible"]
    assert "real_rgb_support_available" in observed["reasons_not_eligible"]
    assert not unknown_geometry["eligible"]
    assert residual["eligible"]
    assert residual["geometry_frozen"]
    assert residual["allowed_parameters_if_generated"] == ["appearance", "opacity"]


def test_roi3d_roundtrip_and_depth_consistent_projection(tmp_path):
    points = np.asarray([[x, y, 4.0] for x in np.linspace(-0.5, 0.5, 5) for y in np.linspace(-0.5, 0.5, 5)])
    roi = ArtifactROI3D(
        roi_id="facade",
        points=points,
        source_view_ids=(1,),
        evidence_view_ids=(2, 3),
        classification="wrong_geometry",
        plane_world=np.asarray([0.0, 0.0, 1.0, -4.0]),
        slab_thickness=0.02,
    )
    path = roi.save(tmp_path / "roi.npz")
    loaded = ArtifactROI3D.load(path)
    depth = np.full((60, 80), 4.0, dtype=np.float32)

    mask = loaded.project_mask(camera(), reference_depth=depth, point_radius=2, close_radius=2)

    assert loaded.summary()["evidence_view_ids"] == [2, 3]
    assert mask.any()
    assert mask.mean() < 0.5


def test_component_detector_finds_coherent_bad_region_after_color_fit():
    target = np.full((100, 140, 3), 0.45, dtype=np.float32)
    target[25:75, 40:100] = 0.8
    render = np.clip(target * 0.8 + 0.05, 0.0, 1.0)
    render[30:70, 45:95] = 0.05
    alpha = np.ones((100, 140), dtype=np.float32)
    depth = np.full((100, 140), 3.0, dtype=np.float32)

    result = build_artifact_components(
        render,
        target,
        alpha,
        depth,
        np.ones((100, 140), dtype=bool),
        min_area_fraction=0.01,
    )

    assert result["components"]
    assert result["labels"][50, 70] > 0
    assert result["summary"]["largest_component_fraction"] > 0.05


def test_component_detector_reports_global_failure_but_disables_local_repair():
    target = np.zeros((100, 100, 3), dtype=np.float32)
    render = np.ones_like(target)
    alpha = np.ones((100, 100), dtype=np.float32)
    depth = np.full((100, 100), 3.0, dtype=np.float32)

    result = build_artifact_components(
        render,
        target,
        alpha,
        depth,
        np.ones((100, 100), dtype=bool),
        residual_threshold=0.1,
        min_area_fraction=0.01,
        max_area_fraction=0.4,
    )

    assert len(result["components"]) == 1
    assert result["components"][0]["oversized"]
    assert not result["components"][0]["repair_eligible"]
    assert result["summary"]["artifact_fraction"] > 0.95
    assert result["summary"]["repair_eligible_artifact_fraction"] == 0.0


def test_pose_interpolation_and_projected_mask_are_bounded():
    first = np.eye(4)
    second = np.eye(4)
    second[0, 3] = 2.0
    midpoint = interpolate_c2w(first, second, 0.5)
    assert np.allclose(midpoint[:3, 3], [1.0, 0.0, 0.0])

    cam = camera(tx=1.0)
    points = np.asarray([[x, y, 5.0] for x in np.linspace(0.5, 1.5, 6) for y in np.linspace(-0.5, 0.5, 6)])
    mask = rasterize_projected_component(points, cam, point_radius=2, close_radius=2)
    assert mask.any()
    assert mask.mean() < 0.35


def test_multiview_depth_support_rejects_inconsistent_points():
    first = camera("first", tx=0.0)
    second = camera("second", tx=0.5)
    points = np.asarray([[0.0, 0.0, 4.0], [0.5, 0.0, 8.0]])
    depth_first = np.full((60, 80), 4.0, dtype=np.float32)
    depth_second = np.full((60, 80), 4.0, dtype=np.float32)
    alpha = np.ones((60, 80), dtype=np.float32)
    clean = np.ones((60, 80), dtype=bool)

    supported, counts = multiview_depth_supported_points(
        points,
        [(first, depth_first, alpha, clean), (second, depth_second, alpha, clean)],
        min_views=2,
    )

    assert supported.shape == (1, 3)
    assert np.allclose(supported[0], points[0])
    assert counts.tolist() == [2, 0]


def test_clean_visibility_support_does_not_depend_on_rendered_depth():
    first = camera("first", tx=0.0)
    second = camera("second", tx=0.5)
    points = np.asarray([[0.0, 0.0, 4.0], [0.5, 0.0, 8.0]])
    first_clean = np.ones((60, 80), dtype=bool)
    second_clean = np.ones((60, 80), dtype=bool)
    projected, _, inside = project_points(points[1:2], second)
    assert inside[0]
    x, y = np.rint(projected[0]).astype(int)
    second_clean[y, x] = False

    supported, counts = multiview_clean_visible_points(
        points,
        [(first, first_clean), (second, second_clean)],
        min_views=2,
    )

    assert supported.shape == (1, 3)
    assert np.allclose(supported[0], points[0])
    assert counts.tolist() == [2, 1]


def test_wrong_geometry_scan_keeps_failure_near_clean_support():
    scan = [
        {"fraction": 0.05, "depth_metrics": {"median_relative_depth_error": 0.62, "overlap_fraction": 1.0, "projected_count": 18}},
        {"fraction": 0.10, "depth_metrics": {"median_relative_depth_error": 0.45, "overlap_fraction": 1.0, "projected_count": 18}},
        {"fraction": 0.15, "depth_metrics": {"median_relative_depth_error": 0.26, "overlap_fraction": 1.0, "projected_count": 18}},
    ]

    selected = select_wrong_geometry_interpolation(scan, min_median_relative_depth_error=0.30)

    assert selected["fraction"] == 0.10


def test_support_plane_produces_camera_z_depth_and_rejects_outlier():
    yy, xx = np.mgrid[-1.0:1.01:0.5, -1.0:1.01:0.5]
    points = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, 4.0)], axis=1)
    points = np.concatenate([points, np.asarray([[0.0, 0.0, 9.0]])], axis=0)

    plane, inliers, metrics = fit_support_plane(points)
    mask = np.zeros((60, 80), dtype=bool)
    mask[20:40, 30:50] = True
    depth, valid = render_support_plane_depth(camera(), plane, mask)

    assert not inliers[-1]
    assert metrics["inlier_fraction"] > 0.8
    assert valid[mask].all()
    assert np.allclose(depth[mask], 4.0, atol=1e-4)


def test_conservative_edit_prefers_small_near_best_trial():
    trials = [
        {"edited_points": 1, "depth_improvement": 0.01, "outside_rgb_mae": 0.005, "min_validation_psnr_delta": 0.0},
        {"edited_points": 3, "depth_improvement": 0.092, "outside_rgb_mae": 0.014, "min_validation_psnr_delta": 0.0},
        {"edited_points": 10, "depth_improvement": 0.094, "outside_rgb_mae": 0.014, "min_validation_psnr_delta": -0.02},
        {"edited_points": 20, "depth_improvement": 0.18, "outside_rgb_mae": 0.04, "min_validation_psnr_delta": -0.01},
    ]

    selected = select_conservative_edit_trial(
        trials,
        min_depth_improvement=0.05,
        max_outside_rgb_mae=0.02,
        max_validation_psnr_drop=0.05,
    )

    assert selected["edited_points"] == 3


def test_projective_color_fusion_prefers_depth_support_and_rejects_disagreement():
    colors = np.asarray(
        [
            [[0.2, 0.3, 0.4], [0.1, 0.1, 0.1], [0.0, 0.0, 0.0]],
            [[0.9, 0.8, 0.7], [0.12, 0.10, 0.11], [1.0, 1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    semantic = np.ones((2, 3), dtype=bool)
    depth = np.asarray([[True, False, False], [False, False, False]])
    scores = np.zeros((2, 3), dtype=np.float32)

    fused, accepted, diagnostics = blend_projective_reference_colors(
        colors,
        semantic,
        depth,
        scores,
        max_fallback_rgb_disagreement=0.15,
    )

    assert np.allclose(fused[0], colors[0, 0])
    assert accepted.tolist() == [True, True, False]
    assert diagnostics["depth_supported_count"] == 1
    assert diagnostics["consistent_multiview_fallback_count"] == 1


def test_virtual_reprojection_consistency_distinguishes_stable_surface_from_depth_switch():
    height, width = 60, 80
    rgb = np.stack(
        [
            np.tile(np.linspace(0.0, 1.0, width), (height, 1)),
            np.tile(np.linspace(0.0, 1.0, height)[:, None], (1, width)),
            np.full((height, width), 0.5),
        ],
        axis=2,
    ).astype(np.float32)
    depth = np.full((height, width), 4.0, dtype=np.float32)
    alpha = np.ones((height, width), dtype=np.float32)
    normals = np.zeros((height, width, 3), dtype=np.float32)
    normals[..., 2] = 1.0

    stable = virtual_reprojection_consistency(
        rgb, depth, alpha, normals, camera(), rgb, depth, alpha, normals, camera()
    )
    assert stable["depth_consistent_fraction"] > 0.99
    assert stable["depth_relative_p90"] < 1e-6
    assert stable["rgb_mae"] < 1e-6
    assert stable["normal_angle_p90_degrees"] < 1e-4

    switched = virtual_reprojection_consistency(
        rgb,
        depth,
        alpha,
        normals,
        camera(),
        rgb,
        depth * 2.0,
        alpha,
        normals,
        camera(),
    )
    assert switched["depth_consistent_fraction"] == 0.0
    assert switched["depth_relative_p90"] > 0.4
