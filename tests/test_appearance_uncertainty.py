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
