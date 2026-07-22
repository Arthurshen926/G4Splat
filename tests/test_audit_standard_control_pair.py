import json

from scripts.audit_standard_control_pair import build_pair_audit, parse_cfg_args


def _write_run(
    path,
    *,
    threshold,
    source_hash="source",
    implementation="implementation-a",
    training_state=None,
):
    path.mkdir()
    (path / "cfg_args").write_text(
        "Namespace(model_path={!r}, densify_grad_threshold={}, iterations=7000, "
        "source_path='dataset')\n".format(str(path), threshold)
    )
    manifest = {
        "source_path": "dataset",
        "image_count": 1487,
        "colmap_camera_count": 1487,
        "loaded_camera_count": 1487,
        "camera_set_sha256_input": source_hash,
        "name_mapping_sha256": "names",
        "effective_initialization_ply_sha256": "ply",
        "cambridge_mask_pickle_sha256": "masks",
        "supervision_profile": "ulfloc_masked",
        "ulfloc_native_pixel_protocol": True,
        "opacity_reset_iterations": [500, 3000, 6000],
        "cudnn": {"deterministic": True, "benchmark": False},
        "camera_schedule": {"sequence_sha256": "schedule"},
        "mask_resolution_audit": {"native_pixel_geometry": True},
        "implementation": {"files": {"trainer": implementation}},
        "training_state": training_state,
    }
    (path / "input_manifest.json").write_text(json.dumps(manifest))


def test_parse_cfg_args_rejects_non_literal_namespace(tmp_path):
    path = tmp_path / "cfg_args"
    path.write_text("not_a_namespace()")

    try:
        parse_cfg_args(path)
    except RuntimeError as error:
        assert "Namespace" in str(error)
    else:
        raise AssertionError("Only literal Namespace cfg_args should be accepted")


def test_pair_audit_accepts_only_declared_threshold_change(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, threshold=0.0004)
    _write_run(candidate, threshold=0.0002)

    audit = build_pair_audit(
        baseline,
        candidate,
        allowed_config_differences={"model_path", "densify_grad_threshold"},
    )

    assert audit["passed"]
    assert audit["immutable_manifest_differences"] == []
    assert [row["field"] for row in audit["config_differences"]] == [
        "densify_grad_threshold",
        "model_path",
    ]


def test_pair_audit_flags_data_identity_change(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, threshold=0.0004)
    _write_run(candidate, threshold=0.0002, source_hash="different")

    audit = build_pair_audit(
        baseline,
        candidate,
        allowed_config_differences={"model_path", "densify_grad_threshold"},
    )

    assert not audit["passed"]
    assert audit["immutable_manifest_differences"][0]["field"] == "camera_set_sha256_input"


def test_pair_audit_flags_implementation_change(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, threshold=0.0004)
    _write_run(candidate, threshold=0.0002, implementation="implementation-b")

    audit = build_pair_audit(
        baseline,
        candidate,
        allowed_config_differences={"model_path", "densify_grad_threshold"},
    )

    assert not audit["passed"]
    assert audit["immutable_manifest_differences"] == [
        {
            "field": "implementation",
            "baseline": {"files": {"trainer": "implementation-a"}},
            "candidate": {"files": {"trainer": "implementation-b"}},
            "allowed": False,
        }
    ]


def test_pair_audit_requires_common_exact_training_state_for_causal_branches(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(
        baseline,
        threshold=0.0004,
        training_state={"resume_training_state_sha256": "parent-a", "start_iteration": 30001},
    )
    _write_run(
        candidate,
        threshold=0.0004,
        training_state={"resume_training_state_sha256": "parent-b", "start_iteration": 30001},
    )

    audit = build_pair_audit(baseline, candidate, allowed_config_differences={"model_path"})

    assert not audit["passed"]
    assert audit["immutable_manifest_differences"] == [
        {
            "field": "training_state",
            "baseline": {"resume_training_state_sha256": "parent-a", "start_iteration": 30001},
            "candidate": {"resume_training_state_sha256": "parent-b", "start_iteration": 30001},
            "allowed": False,
        }
    ]


def test_pair_audit_allows_only_declared_implementation_change(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, threshold=0.0004)
    _write_run(candidate, threshold=0.0004, implementation="implementation-b")

    audit = build_pair_audit(
        baseline,
        candidate,
        allowed_config_differences={"model_path"},
        allowed_manifest_differences={"implementation"},
    )

    assert audit["passed"]
    assert audit["unexpected_immutable_manifest_differences"] == []
    assert audit["immutable_manifest_differences"][0]["allowed"] is True
