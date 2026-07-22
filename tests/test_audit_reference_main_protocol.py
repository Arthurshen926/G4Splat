from scripts.audit_reference_main_protocol import audit_sources


def _sources(*, scene_args: str, two_d_args: str, three_d_args: str, launcher_type: str):
    return {
        "train.py": (
            "from renderer import render_gsplat\n"
            "def training(gaussians, opt, scene):\n"
            "    render_gsplat(rgb_only=True)\n"
            "    Ll1_feature = 0\n"
            "    loss = Ll1_feature\n"
            "    scene.save(7)\n"
            "    gaussians.densify_and_prune(0.1, 0.005, 1.0, None)\n"
            "    gaussians.reset_opacity()\n"
            "    gaussians.optimizer.step()\n"
        ),
        "scene/__init__.py": (
            "class Scene:\n"
            "    def init(self, info, args):\n"
            f"        self.gaussians.create_from_pcd({scene_args})\n"
        ),
        "scene/gaussian_model.py": (
            "class GaussianModel:\n"
            f"    def create_from_pcd(self, {three_d_args}):\n"
            "        pass\n"
            "class GaussianModel_2dgs:\n"
            f"    def create_from_pcd(self, {two_d_args}):\n"
            "        pass\n"
        ),
        "gaussian_renderer/__init__.py": "import gsplat\n",
        "arguments/__init__.py": (
            "class OptimizationParams:\n"
            "    def __init__(self):\n"
            "        self.opacity_cull = 0.05\n"
        ),
        "scripts/train_cambridge.sh": (
            f"python train.py -g {launcher_type} --iterations 30000 "
            "--densify_grad_threshold 0.0004 --position_lr_init 0.000016 "
            "--scaling_lr 0.001 --images processed\n"
        ),
    }


def test_main_protocol_audit_reports_representation_and_constructor_contracts(tmp_path):
    report = audit_sources(
        root=tmp_path,
        revision="deadbeef",
        sources=_sources(
            scene_args="pcd, extent, speedup",
            two_d_args="pcd, extent, loc_dim, speedup",
            three_d_args="pcd, extent, speedup",
            launcher_type="3dgs",
        ),
    )

    assert report["trainer"]["checkpoint_order"]["save_precedes_densify"]
    assert report["cambridge_launcher"]["all_commands_use_3dgs"]
    assert report["two_dgs_constructor"] == {
        "scene_call_argument_count": 3,
        "model_method_argument_count": 4,
        "callable": False,
    }
    assert report["three_dgs_constructor"]["callable"]
    assert report["trainer"]["effective_opacity_cull"] == 0.005
    assert report["trainer"]["contains_feature_loss"]
    assert report["trainer"]["checkpoint_order"] == {
        "scene_save_lines": [6],
        "densify_and_prune_lines": [7],
        "reset_opacity_lines": [8],
        "optimizer_step_lines": [9],
        "save_precedes_densify": True,
        "save_precedes_opacity_reset": True,
        "save_precedes_optimizer_step": True,
    }
    assert report["renderer"]["uses_gsplat"]
