import numpy as np
from plyfile import PlyData, PlyElement

from scripts.audit_gaussian_ply_pair import build_ply_pair_audit


def _write_ply(path, values):
    table = np.empty(len(values), dtype=[("x", "f4"), ("opacity", "f4")])
    table["x"] = [value[0] for value in values]
    table["opacity"] = [value[1] for value in values]
    PlyData([PlyElement.describe(table, "vertex")], text=False).write(str(path))


def test_ordered_ply_audit_accepts_last_bit_noise_within_tolerance(tmp_path):
    baseline = tmp_path / "baseline.ply"
    candidate = tmp_path / "candidate.ply"
    _write_ply(baseline, [(1.0, 0.1), (2.0, 0.2)])
    _write_ply(candidate, [(1.0 + 4e-7, 0.1), (2.0, 0.2)])

    audit = build_ply_pair_audit(baseline, candidate, atol=1e-6)

    assert audit["passed"]
    assert audit["schema_match"]
    assert audit["baseline_vertex_count"] == 2
    assert audit["fields"]["x"]["nonzero_count"] == 1


def test_ordered_ply_audit_rejects_topology_or_numeric_change(tmp_path):
    baseline = tmp_path / "baseline.ply"
    candidate = tmp_path / "candidate.ply"
    _write_ply(baseline, [(1.0, 0.1), (2.0, 0.2)])
    _write_ply(candidate, [(1.0, 0.1), (2.0, 0.25)])

    audit = build_ply_pair_audit(baseline, candidate, atol=1e-4)

    assert not audit["passed"]
    assert audit["fields"]["opacity"]["max_abs_difference"] > 1e-4
