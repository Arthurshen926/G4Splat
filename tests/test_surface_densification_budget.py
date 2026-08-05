"""Regression tests for bounded native-2DGS topology growth."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn


SURFEL_ROOT = str(
    (Path(__file__).resolve().parents[1] / "2d-gaussian-splatting").resolve()
)
if SURFEL_ROOT not in sys.path:
    sys.path.insert(0, SURFEL_ROOT)

from scene.gaussian_model import GaussianModel


class _ReplacingSplitProbe:
    """Minimal model implementing the bounded-topology method's contract."""

    SOURCE_CHART_RESIDUAL = GaussianModel.SOURCE_CHART_RESIDUAL
    SOURCE_FREE_RESIDUAL = GaussianModel.SOURCE_FREE_RESIDUAL
    SOURCE_DAV2_RIGID_HOLE = GaussianModel.SOURCE_DAV2_RIGID_HOLE

    def __init__(self, count: int = 10):
        self._xyz = torch.zeros(count, 3)
        # Large enough for the split path (> percent_dense * extent), while
        # remaining below the routine world-space culling threshold.
        self._scaling = torch.full((count, 3), -3.0)
        self._opacity = torch.zeros(count, 1)
        self._source_type = torch.zeros(count, dtype=torch.int16)
        self._track_id = torch.full((count,), -1, dtype=torch.int64)
        self._protected_flag = torch.zeros(count, dtype=torch.bool)
        self._geometry_confidence = torch.ones(count)
        self._observation_mass = torch.zeros(count)
        self.max_radii2D = torch.full((count,), 32.0)
        self.xyz_gradient_accum = torch.ones(count, 1)
        self.denom = torch.ones(count, 1)
        self.percent_dense = 0.01
        self.use_mip_filter = False
        self.requested_split_children = None

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    def prune_points(self, mask):
        keep = ~mask
        self._xyz = self._xyz[keep]
        self._scaling = self._scaling[keep]
        self._opacity = self._opacity[keep]
        self._source_type = self._source_type[keep]
        self._track_id = self._track_id[keep]
        self._protected_flag = self._protected_flag[keep]
        self._geometry_confidence = self._geometry_confidence[keep]
        self._observation_mass = self._observation_mass[keep]
        self.xyz_gradient_accum = self.xyz_gradient_accum[keep]
        self.denom = self.denom[keep]
        self.max_radii2D = self.max_radii2D[keep]

    def densify_and_clone_limited(self, *args, **kwargs):
        eligibility = kwargs.get("eligibility_mask")
        if eligibility is not None and not bool(eligibility.any()):
            return 0
        raise AssertionError("Large projected probes must spend capacity on splits")

    def densify_and_split_limited(
        self,
        _grads,
        _max_grad,
        _extent,
        max_new_points,
        *,
        children_per_parent,
        replace_parent,
        opacity_ceiling,
        **_kwargs,
    ):
        assert children_per_parent == 2
        assert replace_parent is True
        assert opacity_ceiling > 0.99
        assert _kwargs.get("source_type") is None
        children = int(max_new_points)
        parents = children // children_per_parent
        self.requested_split_children = children
        self._xyz = torch.zeros(len(self._xyz) + children - parents, 3)
        return children


@pytest.mark.parametrize(
    ("max_points", "max_growth", "expected_growth"),
    ((100, 4, 4), (13, 4, 3)),
)
def test_replacing_splits_charge_net_growth_not_child_count(
    max_points, max_growth, expected_growth
):
    probe = _ReplacingSplitProbe()

    report = GaussianModel.densify_and_prune_bounded(
        probe,
        max_grad=0.1,
        min_opacity=0.001,
        extent=1.0,
        max_screen_size=64,
        max_points=max_points,
        max_growth=max_growth,
    )

    assert report["split_parents"] == expected_growth
    assert probe.requested_split_children == 2 * expected_growth
    assert report["net_growth"] == expected_growth
    assert report["after"] == 10 + expected_growth
    assert report["after"] <= max_points
    assert report["net_growth"] <= max_growth


def test_saturated_budget_never_evicts_unrelated_weak_points():
    probe = _ReplacingSplitProbe()
    probe.xyz_gradient_accum[4:] = 0
    probe._opacity[4:] = torch.logit(torch.full((6, 1), 0.01))
    probe._geometry_confidence[4:] = 0.05

    report = GaussianModel.densify_and_prune_bounded(
        probe,
        max_grad=0.1,
        min_opacity=0.005,
        extent=1.0,
        max_screen_size=64,
        max_points=10,
        max_growth=4,
    )

    assert report["routine_pruned"] == 0
    assert report["reallocated_pruned"] == 0
    assert report["split_parents"] == 0
    assert report["after"] == 10
    assert report["model_delta"] == 0
    assert report["capacity_saturated_before"] is True
    assert report["candidates_deferred_for_capacity"] == 4
    assert report["global_capacity_eviction_disabled"] is True


def test_residual_topology_cannot_prune_or_split_immutable_prefix():
    probe = _ReplacingSplitProbe()
    probe.xyz_gradient_accum.zero_()
    probe.max_radii2D.zero_()
    probe._opacity[:6] = torch.logit(torch.full((6, 1), 0.001))
    probe._opacity[6:] = torch.logit(torch.full((4, 1), 0.5))
    probe._observation_mass.fill_(100.0)

    report = GaussianModel.densify_and_prune_bounded(
        probe,
        max_grad=0.1,
        min_opacity=0.005,
        extent=1.0,
        max_screen_size=64,
        max_points=20,
        max_growth=4,
        mutable_start=6,
    )

    assert report["pruned"] == 0
    assert report["split_parents"] == 0
    assert report["after"] == 10
    assert report["mutable_start"] == 6
    assert report["mutable_rows_before"] == 4
    assert report["immutable_prefix_preserved"] is True


def test_densification_statistics_accept_continuous_responsibility():
    probe = _ReplacingSplitProbe(count=3)
    probe.xyz_gradient_accum.zero_()
    probe.denom.zero_()
    means2d = torch.zeros(3, 3, requires_grad=True)
    means2d.grad = torch.tensor(
        [[3.0, 4.0, 0.0], [0.0, 2.0, 0.0], [9.0, 0.0, 0.0]]
    )

    GaussianModel.add_densification_stats(
        probe, means2d, torch.tensor([1.0, 0.5, 0.0])
    )

    torch.testing.assert_close(
        probe.xyz_gradient_accum[:, 0],
        torch.tensor([5.0, 1.0, 0.0]),
    )
    torch.testing.assert_close(
        probe.denom[:, 0], torch.tensor([1.0, 0.5, 0.0])
    )
    torch.testing.assert_close(
        probe._observation_mass, torch.tensor([1.0, 0.5, 0.0])
    )


def test_dav2_opacity_cull_matures_continuously_with_observation_mass():
    unseen = _ReplacingSplitProbe(count=1)
    unseen.xyz_gradient_accum.zero_()
    unseen.max_radii2D.zero_()
    unseen._source_type[:] = GaussianModel.SOURCE_DAV2_RIGID_HOLE
    unseen._opacity[:] = torch.logit(torch.tensor(0.001))

    unseen_report = GaussianModel.densify_and_prune_bounded(
        unseen,
        max_grad=1.0,
        min_opacity=0.005,
        extent=1.0,
        max_screen_size=0,
        max_points=10,
        max_growth=1,
    )
    assert unseen_report["routine_pruned"] == 0
    assert unseen_report["dav2_lineage"]["before"] == 1
    assert unseen_report["dav2_lineage"]["after"] == 1
    assert (
        unseen_report["dav2_lineage"][
            "continuous_cull_maturity_mean"
        ]
        == 0.0
    )

    observed = _ReplacingSplitProbe(count=1)
    observed.xyz_gradient_accum.zero_()
    observed.max_radii2D.zero_()
    observed._source_type[:] = GaussianModel.SOURCE_DAV2_RIGID_HOLE
    observed._opacity[:] = torch.logit(torch.tensor(0.001))
    observed._observation_mass[:] = 64.0

    observed_report = GaussianModel.densify_and_prune_bounded(
        observed,
        max_grad=1.0,
        min_opacity=0.005,
        extent=1.0,
        max_screen_size=0,
        max_points=10,
        max_growth=1,
    )
    assert observed_report["routine_pruned"] == 1
    assert observed_report["dav2_lineage"]["opacity_pruned"] == 1
    assert observed_report["dav2_lineage"]["after"] == 0


def test_metric_evidence_opacity_cull_matures_continuously():
    unseen = _ReplacingSplitProbe(count=1)
    unseen.xyz_gradient_accum.zero_()
    unseen.max_radii2D.zero_()
    unseen._source_type[:] = GaussianModel.SOURCE_MAST3R_TRACK
    unseen._opacity[:] = torch.logit(torch.tensor(0.001))

    report = GaussianModel.densify_and_prune_bounded(
        unseen,
        max_grad=1.0,
        min_opacity=0.005,
        extent=1.0,
        max_screen_size=64,
        max_points=10,
        max_growth=1,
        observation_reference_count=1487,
    )

    assert report["routine_pruned"] == 0
    assert report["opacity_pruned"] == 0
    assert report["after"] == 1


def test_bound_chart_atlas_cell_requires_explicit_retirement_receiver():
    probe = _ReplacingSplitProbe(count=1)
    probe.xyz_gradient_accum.zero_()
    probe.max_radii2D.zero_()
    probe._source_type[:] = GaussianModel.SOURCE_CHART_RESIDUAL
    probe._track_id[:] = -2
    probe._opacity[:] = torch.logit(torch.tensor(1.0e-6))
    probe._observation_mass[:] = 100_000.0

    report = GaussianModel.densify_and_prune_bounded(
        probe,
        max_grad=1.0,
        min_opacity=0.005,
        extent=1.0,
        max_screen_size=0,
        max_points=10,
        max_growth=1,
        observation_reference_count=1,
    )

    assert report["routine_pruned"] == 0
    assert report["after"] == 1
    assert report["chart_atlas_lifecycle"] == {
        "bound_before": 1,
        "ordinary_opacity_retirement_blocked": 1,
        "ordinary_opacity_is_retirement_authority": False,
        "allowed_retirement": (
            "uv_quadtree_replace_and_retire_or_independent_"
            "geometric_contradiction"
        ),
    }


def test_unbound_chart_provenance_can_follow_ordinary_opacity_lifecycle():
    probe = _ReplacingSplitProbe(count=1)
    probe.xyz_gradient_accum.zero_()
    probe.max_radii2D.zero_()
    probe._source_type[:] = GaussianModel.SOURCE_CHART_RESIDUAL
    probe._track_id[:] = -1
    probe._opacity[:] = torch.logit(torch.tensor(1.0e-6))
    probe._observation_mass[:] = 100_000.0

    report = GaussianModel.densify_and_prune_bounded(
        probe,
        max_grad=1.0,
        min_opacity=0.005,
        extent=1.0,
        max_screen_size=0,
        max_points=10,
        max_growth=1,
        observation_reference_count=1,
    )

    assert report["routine_pruned"] == 1
    assert report["after"] == 0
    assert report["chart_atlas_lifecycle"][
        "ordinary_opacity_retirement_blocked"
    ] == 0


def test_oversized_surface_is_replacing_split_not_pruned():
    probe = _ReplacingSplitProbe(count=1)
    probe.max_radii2D[:] = 128.0
    probe._opacity[:] = torch.logit(torch.tensor(0.5))

    report = GaussianModel.densify_and_prune_bounded(
        probe,
        max_grad=0.1,
        min_opacity=0.005,
        extent=1.0,
        max_screen_size=64,
        max_points=10,
        max_growth=1,
    )

    assert report["routine_pruned"] == 0
    assert report["oversized_screen_retained_for_split"] == 1
    assert report["split_parents"] == 1
    assert report["after"] == 2


def test_screen_footprint_alone_cannot_manufacture_split_evidence():
    probe = _ReplacingSplitProbe(count=1)
    probe.xyz_gradient_accum.zero_()
    probe.max_radii2D[:] = 128.0
    probe._opacity[:] = torch.logit(torch.tensor(0.5))

    report = GaussianModel.densify_and_prune_bounded(
        probe,
        max_grad=0.1,
        min_opacity=0.005,
        extent=1.0,
        max_screen_size=64,
        max_points=10,
        max_growth=1,
    )

    assert report["screen_evidence_candidates_before"] == 0
    assert report["split_parents"] == 0
    assert report["topology_priority"].find(
        "no_independent_radius_only_split_evidence"
    ) >= 0


def test_screen_footprint_promotes_observed_near_threshold_residual():
    probe = _ReplacingSplitProbe(count=1)
    probe.xyz_gradient_accum[:] = 0.09
    probe.denom[:] = 1.0
    probe.max_radii2D[:] = 128.0
    probe._opacity[:] = torch.logit(torch.tensor(0.5))

    report = GaussianModel.densify_and_prune_bounded(
        probe,
        max_grad=0.1,
        min_opacity=0.005,
        extent=1.0,
        max_screen_size=64,
        max_points=10,
        max_growth=1,
    )

    assert report["screen_evidence_candidates_before"] == 1
    assert report["split_parents"] == 1


def test_append_reseed_preserves_complete_evidence_metadata():
    model = GaussianModel(sh_degree=1)
    model._xyz = nn.Parameter(torch.zeros(1, 3))
    model._features_dc = nn.Parameter(torch.zeros(1, 1, 3))
    model._features_rest = nn.Parameter(torch.zeros(1, 3, 3))
    model._scaling = nn.Parameter(torch.zeros(1, 2))
    model._rotation = nn.Parameter(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    )
    model._opacity = nn.Parameter(torch.zeros(1, 1))
    model._initialize_point_metadata(1, device="cpu")

    appended = model.append_from_parameters(
        torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
        torch.full((2, 2), 0.04),
        torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        ),
        torch.tensor([[0.2, 0.3, 0.4], [0.4, 0.3, 0.2]]),
        initial_opacity=torch.tensor([0.03, 0.06]),
        source_type=torch.tensor([1, 2]),
        track_id=torch.tensor([17, -23]),
        geometry_confidence=torch.tensor([0.8, 0.6]),
        protected_flag=True,
        block_id=torch.tensor([4, 5]),
    )

    assert appended == 2
    assert len(model._xyz) == len(model._track_id) == 3
    assert model._source_type.tolist() == [0, 1, 2]
    assert model._track_id.tolist() == [-1, 17, -23]
    torch.testing.assert_close(
        model._geometry_confidence,
        torch.tensor([1.0, 0.8, 0.6]),
    )
    assert model._protected_flag.tolist() == [False, True, True]
    assert model._block_id.tolist() == [-1, 4, 5]
    assert model._observation_mass.tolist() == [0.0, 0.0, 0.0]
    torch.testing.assert_close(
        model.get_opacity[-2:, 0],
        torch.tensor([0.03, 0.06]),
    )
