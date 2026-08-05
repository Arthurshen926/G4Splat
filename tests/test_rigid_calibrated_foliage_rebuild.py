from pathlib import Path

from outdoor.scene_contract import sha256_file
from scripts.rebuild_foliage_with_rigid_depth import (
    _full_foliage_initialization_contract,
)


def test_full_foliage_rebuild_retires_rigid_placeholder(tmp_path: Path):
    foliage = tmp_path / "foliage_seed_gaussians.pth"
    foliage.write_bytes(b"real visual hull and ray posterior")

    contract = _full_foliage_initialization_contract(
        {
            "rigid_stage_placeholder_foliage": True,
            "foliage_reuse": {
                "policy": "inert_rigid_stage_schema_placeholder",
                "mixed_training_eligible": False,
            },
        },
        foliage_path=foliage,
    )

    assert contract["rigid_stage_placeholder_foliage"] is False
    assert contract["foliage_reuse"] == {
        "policy": (
            "full_rigid_depth_calibrated_visual_hull_ray_posterior_rebuild"
        ),
        "foliage_seed_sha256": sha256_file(foliage),
        "mixed_training_eligible": True,
        "source_placeholder_retired": True,
    }
