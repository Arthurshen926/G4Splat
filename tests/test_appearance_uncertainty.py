from types import SimpleNamespace

import numpy as np
import torch

from outdoor.appearance_uncertainty import OutdoorAppearanceUncertainty


def _camera(name):
    return SimpleNamespace(
        image_name=name,
        image_height=5,
        image_width=7,
        cx=3.0,
        cy=2.0,
        focal_x=5.0,
        focal_y=4.0,
        R=np.eye(3),
    )


def _task(name):
    zero = torch.zeros(5, 7)
    one = torch.ones(5, 7)
    return {
        "image_name": name,
        "p_canopy": one,
        "p_sky": zero,
        "p_transient": zero,
        "p_rigid": zero,
    }


def test_appearance_is_exact_identity_at_initialization_and_has_gradients():
    model = OutdoorAppearanceUncertainty(["seq__000"], rank=2, device="cpu")
    rgb = torch.full((3, 5, 7), 0.4)
    target = torch.full_like(rgb, 0.5)
    conditioned = model(rgb, _camera("seq__000"), _task("seq__000"))
    assert torch.equal(conditioned, rgb)
    loss = model.heteroscedastic_loss(
        conditioned, target, _task("seq__000")
    )
    loss.backward()
    assert model.canopy_decoder.grad is not None
    assert torch.isfinite(model.canopy_decoder.grad).all()


def test_unknown_query_must_use_canonical_render():
    model = OutdoorAppearanceUncertainty(["database__000"], device="cpu")
    try:
        model(
            torch.zeros(3, 5, 7),
            _camera("query__000"),
            _task("query__000"),
        )
    except KeyError as error:
        assert "canonical render" in str(error)
    else:
        raise AssertionError("Unknown query unexpectedly received a train code")


def test_temporal_pairs_do_not_cross_sequence_boundaries():
    model = OutdoorAppearanceUncertainty(
        ["seq1__frame00002", "seq2__frame00001", "seq1__frame00001"],
        rank=2,
        spatial_grid_size=4,
        device="cpu",
    )
    pairs = {tuple(pair) for pair in model.temporal_pairs.cpu().tolist()}
    assert pairs == {(2, 0)}


def test_effective_temporal_code_has_non_collapsing_norm():
    model = OutdoorAppearanceUncertainty(
        ["seq1__frame00001"],
        rank=2,
        temporal_code_norm=0.25,
        device="cpu",
    )
    with torch.no_grad():
        model.codes[0] = torch.tensor([0.003, -0.004])
    code = model.temporal_code("seq1__frame00001")
    assert torch.isclose(code.norm(), torch.tensor(0.25), atol=1e-6)


def test_uncertainty_is_smooth_spatial_and_legacy_camera_grid_is_inert():
    model = OutdoorAppearanceUncertainty(
        ["seq1__frame00001"],
        rank=2,
        spatial_grid_size=4,
        device="cpu",
    )
    with torch.no_grad():
        model.uncertainty_codes[0, 0] = 1.0
        model.uncertainty_codes[0, 1] = 0.0
        model.uncertainty_decoder[0, 1] = 3.0
        model.uncertainty_decoder[0, 2] = -2.0
        model.spatial_uncertainty_basis[0, 0, 0, 0] = 2.0
    local_rgb, sigma = model._spatial_fields(
        "seq1__frame00001", (12, 16)
    )
    assert sigma.shape == (2, 12, 16)
    assert sigma[0].std().item() > 0
    assert torch.equal(local_rgb, torch.zeros_like(local_rgb))
    # The trainable field is continuous and low order: there are no grid-cell
    # jumps capable of imprinting a fixed vertical stripe frequency.
    horizontal_step = (sigma[0, :, 1:] - sigma[0, :, :-1]).abs()
    vertical_step = (sigma[0, 1:, :] - sigma[0, :-1, :]).abs()
    assert horizontal_step.max() < 0.05
    assert vertical_step.max() < 0.05
    assert model.spatial_uncertainty_basis.requires_grad is False
    assert model.local_canopy_basis.requires_grad is False


def test_legacy_shared_code_checkpoint_migrates_to_split_latents():
    model = OutdoorAppearanceUncertainty(
        ["seq1__frame00001"], rank=2, device="cpu"
    )
    legacy = model.capture()
    legacy["version"] = (
        "outdoor_spatial_sequence_appearance_uncertainty_v2"
    )
    for key in (
        "foliage_codes",
        "sky_codes",
        "uncertainty_codes",
        "uncertainty_decoder",
    ):
        legacy["state_dict"].pop(key)
    restored = OutdoorAppearanceUncertainty(
        ["seq1__frame00001"], rank=2, device="cpu"
    )
    restored.restore(legacy)
    assert torch.equal(restored.foliage_codes, restored.codes)
    assert torch.equal(restored.sky_codes, restored.codes)
    assert torch.equal(restored.uncertainty_codes, restored.codes)
