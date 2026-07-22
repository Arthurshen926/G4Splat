from pathlib import Path
from types import SimpleNamespace
import sys

import yaml


CONFIG_ROOT = Path("configs/free_gaussians_refinement")


def _config(name: str) -> dict:
    return yaml.safe_load((CONFIG_ROOT / f"{name}.yaml").read_text())


def _diff(left: dict, right: dict) -> dict:
    return {
        key: (left.get(key), right.get(key))
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }


def test_strict_capacity_and_standard_protocol_schedule_chain_is_single_variable():
    """Prevent later config edits from invalidating the causal ablation chain."""

    base = _config("default")
    densify = _config("default_densify15k")
    opacity = _config("default_densify15k_opacity3k")
    no_mip = _config("default_densify15k_opacity3k_no_mip")

    assert _diff(base, densify) == {"densify_until_iter": (3500, 15000)}
    assert _diff(densify, opacity) == {"opacity_reset_interval": (1000, 3000)}
    assert _diff(opacity, no_mip) == {"use_mip_filter": (True, False)}


def test_standard_normal_schedule_control_changes_only_activation_time():
    assert _diff(_config("default"), _config("default_normal7k")) == {
        "normal_consistency_from": (3500, 7000)
    }
    assert _diff(
        _config("default_densify15k_opacity3k"),
        _config("default_densify15k_opacity3k_normal7k"),
    ) == {"normal_consistency_from": (3500, 7000)}


def test_refinement_wrapper_can_make_opacity_culling_an_explicit_variable():
    source = Path("scripts/refine_free_gaussians.py").read_text()
    assert "'--opacity-cull', '--opacity_cull'" in source
    # A command-line override remains available, while a named refinement
    # profile can now declare its own safe default for an outdoor run.
    assert "effective_opacity_cull =" in source
    assert 'command.extend(["--opacity_cull", str(effective_opacity_cull)])' in source


def test_refinement_explicitly_clears_inherited_mip_state_for_no_mip_controls():
    source = Path("2d-gaussian-splatting/train_with_refine_depth.py").read_text()
    assert "gaussians.set_mip_filter(bool(use_mip_filter))" in source


def test_refinement_exposes_a_reset_timeline_control_for_true_topology_ablations():
    trainer = Path("2d-gaussian-splatting/train_with_refine_depth.py").read_text()
    wrapper = Path("scripts/refine_free_gaussians.py").read_text()
    assert '"--continue-opacity-resets-after-densify"' in trainer
    assert "continue_after_densify=bool(continue_opacity_resets_after_densify)" in trainer
    assert "'--continue-opacity-resets-after-densify'" in wrapper


def test_post30k_non_position_decay_leaves_xyz_on_its_own_schedule():
    """The long-horizon control must not accidentally slow geometry updates."""
    surfel_root = str(Path("2d-gaussian-splatting").resolve())
    if surfel_root not in sys.path:
        sys.path.insert(0, surfel_root)
    from scene.gaussian_model import GaussianModel

    class Probe:
        non_position_lr_decay_from = 30_000
        non_position_lr_decay_until = 40_000
        non_position_lr_final_mult = 0.1
        non_position_base_lrs = {
            "f_dc": 0.0025,
            "f_rest": 0.000125,
            "opacity": 0.05,
            "scaling": 0.005,
            "rotation": 0.001,
        }
        xyz_scheduler_args = staticmethod(lambda iteration: float(iteration))

        def __init__(self):
            self.optimizer = SimpleNamespace(
                param_groups=[
                    {"name": "xyz", "lr": -1.0},
                    {"name": "f_dc", "lr": 0.0025},
                    {"name": "f_rest", "lr": 0.000125},
                    {"name": "opacity", "lr": 0.05},
                    {"name": "scaling", "lr": 0.005},
                    {"name": "rotation", "lr": 0.001},
                ]
            )

    midpoint = Probe()
    GaussianModel.update_learning_rate(midpoint, 35_000)
    lrs = {group["name"]: group["lr"] for group in midpoint.optimizer.param_groups}

    # Geometric interpolation in log-LR space gives sqrt(0.1) at the midpoint.
    assert abs(lrs["f_dc"] - 0.0025 * (0.1 ** 0.5)) < 1e-12
    assert abs(lrs["rotation"] - 0.001 * (0.1 ** 0.5)) < 1e-12
    assert lrs["xyz"] == 35_000.0

    endpoint = Probe()
    GaussianModel.update_learning_rate(endpoint, 40_000)
    endpoint_lrs = {group["name"]: group["lr"] for group in endpoint.optimizer.param_groups}
    assert abs(endpoint_lrs["opacity"] - 0.005) < 1e-12
    assert abs(endpoint_lrs["scaling"] - 0.0005) < 1e-12
    assert endpoint_lrs["xyz"] == 40_000.0
