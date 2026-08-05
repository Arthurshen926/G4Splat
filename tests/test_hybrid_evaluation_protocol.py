import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.evaluate_hybrid_teacher import (
    _assert_camera_geometry_contract,
    _assert_localization_query_contract,
    _assert_query_rgb_compatibility,
    _assert_resolution_contract,
    _assert_semantic_contract,
    _conditioned_visit_counts,
    _database_view_set_contract,
    _evaluation_indices,
    _evaluation_render_modes,
    _historical_uint8_raster,
    _high_frequency_metrics,
    _historical_static_comparison_is_valid,
    _protocol_metrics,
    _projected_radius_diagnostics,
    _query_representation_contract,
    _resolve_evaluation_mode,
    _route_evaluation_scene_artifacts,
    _tree_boundary_masks,
)
from scripts.evaluate_render_dir import metrics as historical_metrics


def test_auto_evaluation_keeps_static_canonical_canopy():
    validity = {
        "canonical_surface": True,
        "canonical_canopy": True,
        "conditioned": False,
        "reason": "single_static_localization_map",
    }
    assert _resolve_evaluation_mode(
        "auto", localization_query=False, branch_validity=validity
    ) == "canonical"
    assert _resolve_evaluation_mode(
        "auto", localization_query=True, branch_validity=validity
    ) == "canonical"


def test_auto_evaluation_uses_rigid_only_when_canopy_is_untrained():
    validity = {
        "canonical_surface": True,
        "canonical_canopy": False,
        "conditioned": False,
        "reason": "rigid_stage_profile",
    }
    assert _resolve_evaluation_mode(
        "auto", localization_query=False, branch_validity=validity
    ) == "rigid"


def test_rigid_evaluation_is_not_reported_as_canonical():
    assert _evaluation_render_modes("rigid") == ("rigid", ("rigid",))
    assert _evaluation_render_modes("canonical") == (
        "canonical",
        ("canonical",),
    )
    assert _evaluation_render_modes("hybrid") == (
        "canonical",
        ("canonical", "conditioned"),
    )


@pytest.mark.parametrize("masked", [False, True])
def test_native_teacher_static_metrics_match_historical_protocol(masked):
    generator = torch.Generator().manual_seed(17)
    prediction = torch.rand(3, 18, 32, generator=generator)
    target = torch.rand(3, 18, 32, generator=generator)
    mask = None
    if masked:
        mask = torch.rand(18, 32, generator=generator) > 0.35
    expected = historical_metrics(prediction, target, mask)
    actual = _protocol_metrics(prediction, target, mask)
    for name in ("psnr", "ssim", "mae", "rmse"):
        assert actual[name] == pytest.approx(expected[name], abs=1e-6)


def test_historical_uint8_raster_matches_renderer_png_write_semantics():
    image = torch.tensor(
        [[[-0.1, 0.0, 0.5, 1.0, 1.2, float("nan")]]]
    )
    actual = _historical_uint8_raster(image)
    expected = torch.tensor(
        [[[0.0, 0.0, 127.0 / 255.0, 1.0, 1.0, 0.0]]]
    )
    torch.testing.assert_close(actual, expected)


def test_historical_uint8_metric_is_not_labelled_by_float_equivalence():
    prediction = torch.tensor([[[1.02, 0.501]]]).expand(3, -1, -1)
    target = torch.tensor([[[1.0, 0.5]]]).expand(3, -1, -1)
    float_metric = _protocol_metrics(prediction, target)
    png_metric = _protocol_metrics(
        _historical_uint8_raster(prediction),
        _historical_uint8_raster(target),
    )
    assert float_metric["mae"] != pytest.approx(png_metric["mae"])


def test_high_frequency_metric_detects_blurred_edge():
    target = torch.zeros(3, 32, 32)
    target[:, :, 16:] = 1.0
    exact = _high_frequency_metrics(target, target)
    blurred = torch.nn.functional.avg_pool2d(
        target[None], kernel_size=5, stride=1, padding=2
    )[0]
    degraded = _high_frequency_metrics(blurred, target)
    assert exact["gradient_mae"] == pytest.approx(0.0)
    assert exact["laplacian_mae"] == pytest.approx(0.0)
    assert exact["gradient_cosine"] == pytest.approx(1.0)
    assert degraded["gradient_mae"] > exact["gradient_mae"]
    assert degraded["laplacian_mae"] > exact["laplacian_mae"]
    assert degraded["gradient_cosine"] < exact["gradient_cosine"]


def test_tree_boundary_masks_separate_inside_and_outside_bands():
    tree_keep = torch.ones(11, 11, dtype=torch.bool)
    tree_keep[4:7, 4:7] = False
    inside, outside, radius = _tree_boundary_masks(
        tree_keep, radius_pixels=1
    )
    assert radius == 1
    assert bool(inside[4, 4])
    assert not bool(inside[5, 5])
    assert bool(outside[3, 4])
    assert not bool(outside[2, 4])
    assert not bool((inside & outside).any())
    assert bool((inside <= ~tree_keep).all())
    assert bool((outside <= tree_keep).all())


def test_tree_boundary_radius_scales_with_render_resolution():
    tree_keep = torch.ones(360, 640, dtype=torch.bool)
    _, _, radius = _tree_boundary_masks(tree_keep)
    assert radius == 5


def test_historical_database_comparison_requires_exact_all_view_set():
    subset = _database_view_set_contract([408, 409, 410], 1487)
    assert not subset["complete_current_split"]
    assert subset["historical_view_count_match"]
    assert not subset["exact_historical_database_view_set"]

    complete = _database_view_set_contract(range(1487), 1487)
    assert complete["complete_current_split"]
    assert complete["historical_view_count_match"]
    assert complete["exact_historical_database_view_set"]

    wrong_scene_size = _database_view_set_contract(range(64), 64)
    assert wrong_scene_size["complete_current_split"]
    assert not wrong_scene_size["historical_view_count_match"]
    assert not wrong_scene_size["exact_historical_database_view_set"]


def test_canonical_full_database_is_historical_static_comparable():
    complete = _database_view_set_contract(range(1487), 1487)
    subset = _database_view_set_contract([408, 409, 410], 1487)
    assert _historical_static_comparison_is_valid(
        raster_contract_match=True,
        database_view_set_contract=complete,
        evaluation_mode="canonical",
        semantic_conditioning_source="rendered",
    )
    assert _historical_static_comparison_is_valid(
        raster_contract_match=True,
        database_view_set_contract=complete,
        evaluation_mode="hybrid",
        semantic_conditioning_source="rendered",
    )
    assert not _historical_static_comparison_is_valid(
        raster_contract_match=True,
        database_view_set_contract=complete,
        evaluation_mode="hybrid",
        semantic_conditioning_source="oracle",
    )
    assert not _historical_static_comparison_is_valid(
        raster_contract_match=True,
        database_view_set_contract=subset,
        evaluation_mode="canonical",
        semantic_conditioning_source="rendered",
    )


def test_projected_radius_diagnostics_ignore_invisible_rows():
    stats = _projected_radius_diagnostics(
        torch.tensor([0.0, -1.0, float("nan"), 2.0, 10.0, 30.0])
    )
    assert stats["visible_count"] == 3
    assert stats["median_pixels"] == pytest.approx(10.0)
    assert stats["maximum_pixels"] == pytest.approx(30.0)
    assert stats["large_count"] == 1
    assert stats["large_fraction"] == pytest.approx(1.0 / 3.0)
    assert stats["largest_rows"][0] == {
        "row": 5,
        "radius_pixels": 30.0,
    }


def test_projected_radius_diagnostics_separate_opaque_risk_by_source():
    stats = _projected_radius_diagnostics(
        torch.tensor([100.0, 40.0, 30.0, 20.0]),
        opacity=torch.tensor([1.0e-6, 0.2, 0.8, 0.9]),
        source_type=torch.tensor([2, 2, 1, 1]),
    )

    assert stats["large_count"] == 3
    assert stats["opacity_ge_0_1"]["large_count"] == 2
    assert stats["opacity_ge_0_5"]["large_count"] == 1
    assert stats["large_opaque_count_by_source_type"] == {
        "1": 1,
        "2": 1,
    }
    assert stats["optical_radius_pixels"]["maximum"] == pytest.approx(24.0)


def test_projected_radius_diagnostics_report_role_specific_quantiles():
    stats = _projected_radius_diagnostics(
        torch.tensor([4.0, 8.0, 16.0, 32.0]),
        opacity=torch.tensor([0.2, 0.4, 0.6, 0.8]),
        source_type=torch.tensor([0, 0, 1, 1]),
        group_label="layer_role",
    )

    crown = stats["visible_by_layer_role"]["0"]
    skeleton = stats["visible_by_layer_role"]["1"]
    assert crown["visible_count"] == 2
    assert crown["radius_median_pixels"] == pytest.approx(6.0)
    assert skeleton["radius_median_pixels"] == pytest.approx(24.0)
    assert skeleton["large_opaque_count"] == 1


def test_evaluation_scene_artifacts_are_isolated_from_teacher_model(tmp_path):
    teacher = tmp_path / "teacher"
    evaluation = tmp_path / "evaluation"
    dataset = SimpleNamespace(model_path=str(teacher))
    original = _route_evaluation_scene_artifacts(dataset, evaluation)
    assert original == teacher.resolve()
    assert Path(dataset.model_path) == evaluation.resolve()
    assert evaluation.is_dir()
    assert not teacher.exists()


def test_evaluation_refuses_training_resolution_mismatch():
    state = {
        "training_contract": {
            "render_resolution": {"width": 640, "height": 360}
        }
    }
    view = type(
        "View",
        (),
        {"image_width": 1920, "image_height": 1080},
    )()
    with pytest.raises(RuntimeError, match="trained="):
        _assert_resolution_contract(
            state, [view], allow_override=False
        )
    _assert_resolution_contract(state, [view], allow_override=True)


def test_evaluation_refuses_camera_geometry_identity_mismatch(tmp_path):
    contract = tmp_path / "camera_intrinsics_contract.json"
    contract.write_text(
        json.dumps({"camera_geometry_sha256": "actual"}),
        encoding="utf-8",
    )
    state = {
        "training_contract": {
            "camera_geometry_sha256": "trained",
        }
    }
    with pytest.raises(RuntimeError, match="camera geometry differs"):
        _assert_camera_geometry_contract(
            state,
            contract,
            allow_override=False,
        )
    actual, matched = _assert_camera_geometry_contract(
        state,
        contract,
        allow_override=True,
    )
    assert actual == "actual"
    assert not matched


def test_evaluation_accepts_exact_camera_geometry_identity(tmp_path):
    contract = tmp_path / "camera_intrinsics_contract.json"
    contract.write_text(
        json.dumps({"camera_geometry_sha256": "same"}),
        encoding="utf-8",
    )
    state = {
        "training_contract": {
            "camera_geometry_sha256": "same",
        }
    }
    actual, matched = _assert_camera_geometry_contract(
        state,
        contract,
        allow_override=False,
    )
    assert actual == "same"
    assert matched


def test_evaluation_binds_tree_mask_to_semantic_contract(tmp_path):
    tree_mask = tmp_path / "masks_with_tree.pkl"
    tree_mask.write_bytes(b"four-channel-tree-mask")
    from outdoor.evidence_store import sha256_file

    contract = tmp_path / "task_semantics.json"
    contract.write_text(
        json.dumps(
            {
                "tree_mask_pickle": str(tree_mask),
                "input_hashes": {
                    "tree_mask_pickle": sha256_file(tree_mask)
                },
                "class_availability": {
                    "canopy": "tree_mask_index_3"
                },
            }
        ),
        encoding="utf-8",
    )
    parsed, digest = _assert_semantic_contract(contract, tree_mask)
    assert parsed["class_availability"]["canopy"] == "tree_mask_index_3"
    assert digest == sha256_file(tree_mask)
    parsed_default, default_digest = _assert_semantic_contract(
        contract, None
    )
    assert parsed_default == parsed
    assert default_digest == digest

    wrong = tmp_path / "base_masks.pkl"
    wrong.write_bytes(b"three-channel-base-mask")
    with pytest.raises(RuntimeError, match="tree mask differs"):
        _assert_semantic_contract(contract, wrong)


def test_conditioned_visit_counts_exclude_pre_activation_schedule():
    state = {
        "iteration": 8,
        "training_contract": {
            "iterations": 10,
            "branch_activation": {"dynamic": 0.2},
        },
        "schedules": {
            # Iterations 1 and 2 are inactive.  Active entries are therefore
            # schedule[2:8] = [2, 2, 3, 4, 2, 4].
            "conditioned": [0, 1, 2, 2, 3, 4, 2, 4, 0, 1],
        },
    }
    counts, audit = _conditioned_visit_counts(state, 5)
    assert counts.tolist() == [0, 0, 3, 1, 2]
    assert audit["available"]
    assert audit["active_conditioned_steps"] == 6
    assert audit["trained_view_count"] == 3
    assert audit["untrained_view_count"] == 2
    assert audit["trained_view_fraction"] == pytest.approx(0.6)


def test_conditioned_visit_counts_use_absolute_schedule_horizon():
    state = {
        "iteration": 6,
        "training_contract": {
            "schedule_horizon": 12,
            "branch_activation": {
                "dynamic": 0.75,
                "dynamic_iteration": 3,
            },
        },
        "schedules": {
            "conditioned": [0, 1, 2, 2, 3, 4, 0, 1, 2, 3, 4, 0],
        },
    }
    counts, audit = _conditioned_visit_counts(state, 5)
    assert counts.tolist() == [0, 0, 2, 1, 1]
    assert audit["activation_source"] == "absolute_iteration"
    assert audit["dynamic_iteration"] == 3
    assert audit["schedule_horizon"] == 12
    assert audit["active_schedule_begin"] == 2
    assert audit["active_schedule_end"] == 6


def test_conditioned_visit_counts_prefers_explicit_resume_ledger():
    state = {
        "iteration": 8,
        "training_contract": {
            "schedule_horizon": 10,
            "branch_activation": {"dynamic_iteration": 1},
        },
        # This repaired schedule was never consumed before a step-6 resume
        # and must not be treated as historical visitation.
        "schedules": {"conditioned": [2] * 10},
        "conditioned_visit_counts": np.asarray([1, 3, 0]),
        "conditioned_visit_provenance": {
            "source": "legacy_saved_schedule_exact_migration"
        },
    }
    counts, audit = _conditioned_visit_counts(state, 3)
    assert counts.tolist() == [1, 3, 0]
    assert audit["source"] == "explicit_runtime_ledger"
    assert audit["active_conditioned_steps"] == 4
    assert audit["trained_view_count"] == 2


def test_conditioned_visit_counts_excludes_canonical_polish_suffix():
    state = {
        "iteration": 10,
        "training_contract": {
            "schedule_horizon": 10,
            "branch_activation": {"dynamic_iteration": 1},
            "resolved_phase_schedule": (
                ("dynamic_appearance", 7),
                ("canonical_polish", 10),
            ),
        },
        "schedules": {
            "conditioned": [0, 1, 0, 1, 0, 1, 0, 2, 2, 2],
        },
    }
    counts, audit = _conditioned_visit_counts(state, 3)
    assert counts.tolist() == [4, 3, 0]
    assert audit["active_schedule_end"] == 7
    assert audit["active_conditioned_steps"] == 7


def test_localization_query_defaults_to_complete_split():
    assert _evaluation_indices(
        indices=None,
        evaluate_all=False,
        localization_query=True,
        view_count=4,
    ) == [0, 1, 2, 3]
    assert _evaluation_indices(
        indices="1,3",
        evaluate_all=False,
        localization_query=True,
        view_count=4,
    ) == [1, 3]


def test_query_representation_allows_labelled_rigid_stage_diagnostic():
    contract = _query_representation_contract(
        localization_query=True,
        evaluation_mode="rigid",
    )
    assert contract["scope"] == "rigid_surface_stage_diagnostic"
    assert not contract["complete_localization_map"]
    assert not contract["conditioned_database_code_used"]
    assert "canonical_canopy_volume" in contract["omitted_components"]


def test_query_representation_rejects_conditioned_database_code():
    with pytest.raises(RuntimeError, match="fitted per-database-image"):
        _query_representation_contract(
            localization_query=True,
            evaluation_mode="hybrid",
        )
    contract = _query_representation_contract(
        localization_query=True,
        evaluation_mode="canonical",
    )
    assert contract["complete_localization_map"]
    assert not contract["conditioned_database_code_used"]


def test_database_diagnostic_retains_historical_default_indices():
    assert _evaluation_indices(
        indices=None,
        evaluate_all=False,
        localization_query=False,
        view_count=500,
    ) == [408, 409, 410]
    with pytest.raises(IndexError, match="outside the selected split"):
        _evaluation_indices(
            indices="4",
            evaluate_all=False,
            localization_query=False,
            view_count=4,
        )


def test_conditioned_visit_counts_marks_legacy_state_unknown():
    counts, audit = _conditioned_visit_counts(
        {"iteration": 100}, view_count=4
    )
    assert counts.tolist() == [0, 0, 0, 0]
    assert not audit["available"]
    assert audit["reason"] == (
        "checkpoint_has_no_conditioned_schedule_contract"
    )


def test_conditioned_visit_counts_rejects_invalid_camera_index():
    state = {
        "iteration": 2,
        "training_contract": {
            "iterations": 2,
            "branch_activation": {"dynamic": 0.0},
        },
        "schedules": {"conditioned": [0, 4]},
    }
    with pytest.raises(RuntimeError, match="out-of-range"):
        _conditioned_visit_counts(state, view_count=4)


def test_query_rgb_requires_training_raster_convention():
    state = {
        "training_contract": {
            "rgb_source": {
                "target_storage": "shared_uint8",
                "canonical_image_size_wh": [640, 360],
            }
        }
    }
    _assert_query_rgb_compatibility(
        state,
        {
            "target_storage": "shared_uint8",
            "canonical_image_size_wh": [640, 360],
        },
    )
    with pytest.raises(RuntimeError, match="raster convention"):
        _assert_query_rgb_compatibility(
            state,
            {
                "target_storage": "shared_uint8",
                "canonical_image_size_wh": [1920, 1080],
            },
        )


def test_query_contract_requires_disjoint_exact_camera_names(tmp_path):
    dataset = tmp_path / "query"
    dataset.mkdir()
    database_contract = tmp_path / "database.json"
    query_contract = tmp_path / "query.json"
    database_contract.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "source_image_name": "seq1/frame00001.png",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    query_contract.write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "split": "localization_query",
                "camera_policy": "calibrated_fixed",
                "image_count": 1,
                "records": [
                    {
                        "image_name": "seq13__frame00001.png",
                        "source_image_name": "seq13/frame00001.png",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    audit = _assert_localization_query_contract(
        database_contract,
        query_contract,
        dataset_source=dataset,
        # Scene strips the raster suffix from image_name.
        view_names=["seq13__frame00001"],
    )
    assert audit["disjointness"]["passed"]
    assert audit["query_image_count"] == 1
    assert audit["camera_name_comparison"] == (
        "path_without_raster_suffix"
    )

    with pytest.raises(RuntimeError, match="camera names differ"):
        _assert_localization_query_contract(
            database_contract,
            query_contract,
            dataset_source=dataset,
            view_names=["seq13__frame00002.png"],
        )
