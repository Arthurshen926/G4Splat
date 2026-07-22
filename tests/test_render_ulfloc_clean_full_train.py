from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from scripts.render_ulfloc_clean_full_train import (
    _external_dataset_args,
    _gsplat_backend_audit,
)


def test_external_export_recreates_training_model_parameter_contract():
    args = _external_dataset_args(
        source_path=Path("/data/cambridge"),
        model_path=Path("/outputs/ulfloc"),
        images="images",
        feature_type="sp",
        resolution=1,
        gaussian_type="2dgs",
    )

    assert args == [
        "-s",
        "/data/cambridge",
        "-m",
        "/outputs/ulfloc",
        "--images",
        "images",
        "-f",
        "sp",
        "-r",
        "1",
        "--data_device",
        "cpu",
        "-g",
        "2dgs",
    ]


def test_external_export_records_and_can_pin_gsplat_backend(monkeypatch):
    fake_gsplat = SimpleNamespace(
        __version__="1.4.0",
        __file__="/vendor/gsplat/__init__.py",
    )
    monkeypatch.setitem(sys.modules, "gsplat", fake_gsplat)

    audit = _gsplat_backend_audit(required_version="1.4.0")

    assert audit == {
        "version": "1.4.0",
        "module_file": "/vendor/gsplat/__init__.py",
    }
    with pytest.raises(RuntimeError, match="required=1.5.3, resolved=1.4.0"):
        _gsplat_backend_audit(required_version="1.5.3")
