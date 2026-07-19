from scripts.audit_causal_repair import audit


def test_contract_audit_accepts_current_relative_removal_threshold_name():
    """A valid current-format opacity intervention must not be falsely rejected."""
    report = {
        "full_train_view_count": 1487,
        "counterfactual_clusters": [
            {
                "cluster_id": 3,
                "accepted": True,
                "visible_clean_control_count": 1,
                "median_bad_relative_improvement": 0.08,
                "worst_clean_mae_increase": 0.001,
                "acceptance_thresholds": {
                    "min_bad_relative_improvement": 0.04,
                    "max_clean_mae_increase": 0.01,
                    "min_visible_clean_controls": 1,
                },
            }
        ],
        "geometry_projection_counterfactuals": {},
        "edit_plan": {"audit_groups": []},
        "selected_update_primitive_count": 0,
        "applied_edit": {"mode": "not_requested"},
    }

    result = audit(report, expected_full_train_views=1487)

    assert result["passed"]
    assert result["checks"]["accepted_removals"][0]["passed"]


def test_contract_audit_rejects_missing_removal_threshold():
    """An incomplete report is unsafe, rather than implicitly permissive."""
    report = {
        "full_train_view_count": 1487,
        "counterfactual_clusters": [
            {
                "cluster_id": 4,
                "accepted": True,
                "visible_clean_control_count": 1,
                "median_bad_relative_improvement": 1.0,
                "worst_clean_mae_increase": 0.0,
                "acceptance_thresholds": {"max_clean_mae_increase": 0.01},
            }
        ],
        "geometry_projection_counterfactuals": {},
        "edit_plan": {"audit_groups": []},
        "selected_update_primitive_count": 0,
        "applied_edit": {"mode": "not_requested"},
    }

    result = audit(report, expected_full_train_views=1487)

    assert not result["passed"]
    assert "opacity-zero contract" in result["failures"][0]


def test_version5_contract_requires_complete_all_train_anomaly_prescan():
    report = {
        "version": 5,
        "full_train_view_count": 1487,
        "full_train_anomaly_prescan": {
            "status": "completed",
            "processed_view_count": 1487,
            "report_path": "/tmp/full_train_anomaly_prescan.json",
        },
        "counterfactual_clusters": [],
        "geometry_projection_counterfactuals": {},
        "edit_plan": {"audit_groups": []},
        "selected_update_primitive_count": 0,
        "applied_edit": {"mode": "not_requested"},
    }

    passed = audit(report, expected_full_train_views=1487)
    report["full_train_anomaly_prescan"]["processed_view_count"] = 1486
    failed = audit(report, expected_full_train_views=1487)

    assert passed["passed"]
    assert passed["checks"]["full_train_anomaly_prescan"]["passed"]
    assert not failed["passed"]
    assert "all-real-training-view anomaly prescan" in failed["failures"][0]


def test_version6_contract_requires_real_camera_counterfactual_envelope():
    report = {
        "version": 6,
        "full_train_view_count": 1487,
        "full_train_anomaly_prescan": {
            "status": "completed",
            "processed_view_count": 1487,
            "report_path": "/tmp/full_train_anomaly_prescan.json",
        },
        "counterfactual_clusters": [
            {
                "cluster_id": 3,
                "accepted": True,
                "visible_clean_control_count": 12,
                "median_bad_relative_improvement": 0.08,
                "worst_clean_mae_increase": 0.001,
                "acceptance_thresholds": {
                    "min_bad_relative_improvement": 0.04,
                    "max_clean_mae_increase": 0.01,
                    "min_visible_clean_controls": 12,
                },
                "control_envelope": {
                    "policy": "real_camera_envelope_v1",
                    "quality_is_not_a_filter": True,
                    "temporal_radius_frames": 16,
                    "minimum_visible_control_count": 12,
                },
            }
        ],
        "geometry_projection_counterfactuals": {},
        "edit_plan": {"audit_groups": []},
        "selected_update_primitive_count": 0,
        "applied_edit": {"mode": "not_requested"},
    }

    passed = audit(report, expected_full_train_views=1487)
    report["counterfactual_clusters"][0]["control_envelope"]["quality_is_not_a_filter"] = False
    failed = audit(report, expected_full_train_views=1487)

    assert passed["passed"]
    assert not failed["passed"]
    assert not failed["checks"]["accepted_removals"][0]["real_camera_envelope_passed"]


def test_version7_written_edit_requires_accepted_causal_target_scope():
    report = {
        "version": 7,
        "full_train_view_count": 1487,
        "full_train_anomaly_prescan": {
            "status": "completed",
            "processed_view_count": 1487,
            "report_path": "/tmp/full_train_anomaly_prescan.json",
        },
        "counterfactual_clusters": [],
        "geometry_projection_counterfactuals": {},
        "edit_plan": {"audit_groups": []},
        "selected_update_primitive_count": 0,
        "applied_edit": {
            "output_model": "/tmp/edited_model",
            "validation_gate": {
                "passed": True,
                "target_scope": {
                    "policy": "accepted_counterfactual_causal_targets_only",
                    "names": ["seq4__frame00148"],
                },
            },
        },
    }

    passed = audit(report, expected_full_train_views=1487)
    report["applied_edit"]["validation_gate"]["target_scope"]["names"] = []
    failed = audit(report, expected_full_train_views=1487)

    assert passed["passed"]
    assert not failed["passed"]
    assert not failed["checks"]["post_edit_causal_target_scope"]["passed"]
