import numpy as np
import torch

from matcha.dm_scene.charts import load_charts_data


def test_chart_loader_keeps_string_provenance_on_cpu(tmp_path):
    """Strict Chart archives mix numeric geometry with a string gate basis."""
    archive = tmp_path / "charts_data.npz"
    np.savez(
        archive,
        depths=np.ones((2, 3, 4), dtype=np.float32),
        alignment_gate_valid=np.asarray([True, False]),
        alignment_rejection_basis=np.asarray("mast3r_reference_depths"),
    )

    charts = load_charts_data(str(archive), device="cpu")

    assert isinstance(charts["depths"], torch.Tensor)
    assert charts["depths"].device.type == "cpu"
    assert isinstance(charts["alignment_gate_valid"], torch.Tensor)
    assert isinstance(charts["alignment_rejection_basis"], np.ndarray)
    assert charts["alignment_rejection_basis"].item() == "mast3r_reference_depths"
