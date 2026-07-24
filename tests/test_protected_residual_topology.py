"""Regression guards for the protected residual-continuation contract."""

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GAUSSIAN_MODEL = REPO_ROOT / "2d-gaussian-splatting" / "scene" / "gaussian_model.py"
REFINER = REPO_ROOT / "scripts" / "refine_free_gaussians.py"
RESIDUAL_CONFIG = (
    REPO_ROOT
    / "configs"
    / "free_gaussians_refinement"
    / "outdoor_protected_residual48k.yaml"
)
SPLIT_RESIDUAL_CONFIG = (
    REPO_ROOT
    / "configs"
    / "free_gaussians_refinement"
    / "outdoor_protected_residual56k_split.yaml"
)
PREFIX_AUDIT = REPO_ROOT / "scripts" / "audit_checkpoint_prefix_integrity.py"
SEMANTIC_RESIDUAL = REPO_ROOT / "scripts" / "train_semantic_residual_2dgs.py"


def _method_calls(path: Path, method_name: str) -> set[str]:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            calls = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Attribute):
                        calls.add(child.func.attr)
                    elif isinstance(child.func, ast.Name):
                        calls.add(child.func.id)
            return calls
    raise AssertionError(f"Method not found: {method_name}")


def test_protected_residual_primitives_are_additive_and_never_prune():
    for method_name in ("densify_and_clone_limited", "densify_and_split_limited"):
        calls = _method_calls(GAUSSIAN_MODEL, method_name)

        assert "densification_postfix" in calls
        assert "prune_points" not in calls
        assert "densify_and_split" not in calls
        assert "reset_opacity" not in calls


def test_residual_launcher_and_config_expose_an_explicit_checkpoint_stage():
    launcher = REFINER.read_text()
    config = RESIDUAL_CONFIG.read_text()
    split_config = SPLIT_RESIDUAL_CONFIG.read_text()

    assert "--warmstart-allow-residual-densification" in launcher
    assert "--warmstart-residual-densification-mode" in launcher
    assert "--warmstart-freeze-baseline" in launcher
    assert "--warmstart-residual-lr-restart" in launcher
    assert "--warmstart-residual-densify-max-per-event" in launcher
    assert "iterations: 48000" in config
    assert "densify_until_iter: 47800" in config
    assert "historical all-real-RGB initialization" in config
    assert "iterations: 56000" in split_config
    assert "historical all-real-RGB PLY" in split_config


def test_frozen_checkpoint_prefix_also_clears_adam_momentum():
    freeze_calls = _method_calls(GAUSSIAN_MODEL, "freeze_prefix_gradients")
    clear_calls = _method_calls(GAUSSIAN_MODEL, "clear_prefix_optimizer_state")

    assert "clear_prefix_optimizer_state" in freeze_calls
    assert "zero_" in clear_calls
    source = GAUSSIAN_MODEL.read_text()
    assert "Adam will still apply a non-zero update" in source


def test_checkpoint_continuation_reads_its_own_appearance_sidecar():
    source = (
        REPO_ROOT / "2d-gaussian-splatting" / "train_with_refine_depth.py"
    ).read_text()

    assert "checkpoint_model_dir = os.path.dirname(checkpoint)" in source
    assert "candidate_paths = [" in source
    assert "for candidate in dict.fromkeys(candidate_paths):" in source


def test_checkpoint_prefix_audit_requires_exact_parameter_rows():
    source = PREFIX_AUDIT.read_text()

    assert "MODEL_FIELDS" in source
    assert "torch.equal(baseline, candidate[:count])" in source
    assert "Protected prefix changed during continuation" in source


def test_semantic_residual_fork_has_no_global_prune_or_opacity_reset():
    source = SEMANTIC_RESIDUAL.read_text()
    main_calls = _method_calls(SEMANTIC_RESIDUAL, "main")

    assert "densify_and_clone_limited" in main_calls
    assert "densify_and_split_limited" in main_calls
    assert "prune_points" not in main_calls
    assert "densify_and_prune" not in main_calls
    assert "reset_opacity" not in main_calls
    assert "exact_full_state_restore_then_declared_schedule_fork" in source
    assert '"true_volumetric_foliage": False' in source
    assert "normalized_norm = norm / projected_area" in source
    assert "torch.where(" in source
    assert "sequence_count >= int(args.minimum_support_sequences)" in source
    assert "maximum_baseline >= float(args.minimum_support_baseline)" in source


def test_gaussian_metadata_is_serialized_with_ply_and_training_capture():
    source = GAUSSIAN_MODEL.read_text()

    for field in (
        "primitive_class",
        "source_type",
        "geometry_confidence",
        "protected_flag",
        "block_id",
    ):
        assert field in source
    assert "if len(model_args) not in {12, 13}" in source
