from pathlib import Path

import torch


def test_shared_optimizer_parameters_are_deduplicated_by_identity(monkeypatch):
    mast3r_root = Path(__file__).resolve().parents[1] / "mast3r"
    monkeypatch.syspath_prepend(str(mast3r_root))
    from mast3r.cloud_opt.sparse_ga import unique_optimizer_parameters

    shared = torch.nn.Parameter(torch.tensor([1.0]))
    independent = torch.nn.Parameter(torch.tensor([2.0]))

    result = unique_optimizer_parameters(
        [shared, shared, independent, shared, independent]
    )

    assert result == [shared, independent]
