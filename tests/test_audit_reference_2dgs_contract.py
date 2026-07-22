from pathlib import Path

from scripts.audit_reference_2dgs_contract import build_reference_audit


def _write_reference(root: Path, *, scene_args: str, model_args: str) -> None:
    (root / "scene").mkdir(parents=True)
    (root / "arguments").mkdir()
    (root / "train.py").write_text(
        "def train(gaussians, opt, scene):\n"
        "    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, None)\n"
    )
    (root / "scene" / "__init__.py").write_text(
        "class Scene:\n"
        "    def initialize(self, scene_info, args):\n"
        f"        self.gaussians.create_from_pcd({scene_args})\n"
    )
    (root / "scene" / "gaussian_model.py").write_text(
        "class GaussianModel_2dgs:\n"
        f"    def create_from_pcd(self, {model_args}):\n"
        "        pass\n"
    )
    (root / "arguments" / "__init__.py").write_text(
        "class OptimizationParams:\n"
        "    def __init__(self):\n"
        "        self.opacity_cull = 0.05\n"
    )


def test_reference_audit_exposes_ulfloc_initializer_arity_bug_and_effective_cull(tmp_path):
    ulfloc = tmp_path / "ulfloc"
    stdloc = tmp_path / "stdloc"
    _write_reference(
        ulfloc,
        scene_args="scene_info.point_cloud, 1.0, args.speedup",
        model_args="pcd, spatial_lr_scale, loc_feature_size, speedup",
    )
    _write_reference(
        stdloc,
        scene_args="scene_info.point_cloud, 1.0, scene_info.loc_feature_dim, args.speedup",
        model_args="pcd, spatial_lr_scale, loc_feature_size, speedup",
    )

    report = build_reference_audit(ulfloc, stdloc)

    assert not report["comparisons"]["ulfloc_2d_entrypoint_callable"]
    assert report["comparisons"]["stdloc_2d_entrypoint_callable"]
    assert report["comparisons"]["effective_opacity_cull_matches"]
    assert report["references"]["ulfloc"]["opacity_cull"] == {
        "declared_argument_default": 0.05,
        "effective_train_loop_value": 0.005,
        "declared_matches_effective": False,
    }
