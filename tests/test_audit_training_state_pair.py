import torch

from scripts.audit_training_state_pair import build_training_state_pair_audit


def _write_state(path, *, value=1.0, scalar=7):
    torch.save(
        {
            "version": 2,
            "iteration": 3,
            "model_capture": (
                torch.tensor([value, 2.0]),
                {"exp_avg": torch.tensor([0.25, 0.5])},
            ),
            "rng_state": {"torch_cpu": torch.tensor([1, 2], dtype=torch.uint8)},
            "training_loop_state": {"ema": 0.5, "scalar": scalar},
        },
        path,
    )


def test_training_state_pair_audit_accepts_identical_state(tmp_path):
    baseline = tmp_path / "baseline.pth"
    candidate = tmp_path / "candidate.pth"
    _write_state(baseline)
    _write_state(candidate)

    audit = build_training_state_pair_audit(baseline, candidate)

    assert audit["passed"]
    assert audit["inexact_leaf_count"] == 0
    assert audit["structural_mismatches"] == []


def test_training_state_pair_audit_reports_tensor_and_scalar_difference(tmp_path):
    baseline = tmp_path / "baseline.pth"
    candidate = tmp_path / "candidate.pth"
    _write_state(baseline)
    _write_state(candidate, value=1.25, scalar=8)

    audit = build_training_state_pair_audit(baseline, candidate)

    assert not audit["passed"]
    assert audit["global_max_abs_difference"] == 0.25
    assert any("model_capture" in leaf["path"] for leaf in audit["inexact_leaves"])
    assert any("scalar" in leaf["path"] for leaf in audit["inexact_leaves"])
