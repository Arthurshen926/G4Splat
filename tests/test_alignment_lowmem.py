import inspect
from pathlib import Path

import torch
import yaml

from matcha.dm_deformation.encodings import ChartsEncoding
from matcha.dm_scene.parallel_aligner import (
    ParallelAligner,
    _backward_chunked_chart_encoding_norm,
    _masked_mean,
)
from matcha.dm_trainers.charts_alignment import (
    align_charts_in_parallel,
    reject_catastrophic_alignment_confidences,
    restore_unaligned_geometry,
)


def test_masked_strong_alignment_keeps_matching_and_bounds_projection_memory():
    config = yaml.safe_load(
        Path("configs/charts_alignment/semantic_masked_strong_lowmem.yaml").read_text()
    )

    assert config["alignment"]["use_matching_loss"] is True
    assert config["alignment"]["projection_chunk_size"] <= 65536
    assert config["alignment"]["matching_checkpoint_chunks"] is True
    assert config["alignment"]["use_normal_loss"] is True
    assert config["alignment"]["use_curvature_loss"] is True
    assert config["masking"]["use_masks_for_alignment"] is True
    assert "projection_chunk_size" in inspect.signature(ParallelAligner.optimize).parameters
    assert "projection_chunk_size" in inspect.signature(align_charts_in_parallel).parameters
    assert "matching_checkpoint_chunks" in inspect.signature(ParallelAligner.optimize).parameters
    assert "matching_checkpoint_chunks" in inspect.signature(align_charts_in_parallel).parameters
    assert config["alignment"]["chart_encoding_norm_chunk_rows"] > 0
    assert "chart_encoding_norm_chunk_rows" in inspect.signature(ParallelAligner.optimize).parameters
    assert "chart_encoding_norm_chunk_rows" in inspect.signature(align_charts_in_parallel).parameters


def test_fullmatch_control_changes_only_matching_source_stride():
    lowmem = yaml.safe_load(
        Path("configs/charts_alignment/semantic_masked_strong_lowmem.yaml").read_text()
    )
    fullmatch = yaml.safe_load(
        Path("configs/charts_alignment/semantic_masked_strong_fullmatch_lowmem.yaml").read_text()
    )

    lowmem_stride = lowmem["alignment"].pop("matching_pixel_stride")
    fullmatch_stride = fullmatch["alignment"].pop("matching_pixel_stride")
    assert lowmem_stride == 2
    assert fullmatch_stride == 1
    assert fullmatch == lowmem


def test_nomatching_control_changes_only_crossview_matching_term():
    baseline = yaml.safe_load(
        Path("configs/charts_alignment/semantic_masked_strong_lowmem.yaml").read_text()
    )
    control = yaml.safe_load(
        Path(
            "configs/charts_alignment/semantic_masked_strong_nomatching_lowmem.yaml"
        ).read_text()
    )

    assert baseline["alignment"]["use_matching_loss"] is True
    assert control["alignment"]["use_matching_loss"] is False
    baseline["alignment"].pop("use_matching_loss")
    control["alignment"].pop("use_matching_loss")
    assert control == baseline


def test_chunked_chart_encoding_norm_has_dense_objective_gradient():
    torch.manual_seed(3)
    charts = ChartsEncoding(num_charts=2, encoding_h=4, encoding_w=5, encoding_dim=3)
    uv = torch.stack(
        torch.meshgrid(torch.linspace(-1.0, 1.0, 6), torch.linspace(-1.0, 1.0, 7)),
        dim=-1,
    ).repeat(2, 1, 1, 1)

    dense_loss = charts(uv).norm(dim=-1).mean()
    dense_loss.backward()
    dense_gradient = charts.encodings.grad.detach().clone()
    charts.encodings.grad = None

    chunked_value = _backward_chunked_chart_encoding_norm(
        charts, uv, chunk_rows=2, weight=1.0
    )

    assert abs(chunked_value - float(dense_loss.detach())) < 1e-7
    assert torch.allclose(charts.encodings.grad, dense_gradient, atol=1e-7, rtol=1e-6)


def test_alignment_masked_mean_does_not_dilute_loss_with_excluded_sky():
    values = torch.tensor([[2.0, 1000.0], [4.0, 1000.0]])
    mask = torch.tensor([[True, False], [True, False]])

    assert torch.isclose(_masked_mean(values, mask), torch.tensor(3.0))


def test_alignment_masked_mean_broadcasts_over_curvature_channels():
    values = torch.tensor([[[[1.0, 2.0, 3.0], [100.0, 100.0, 100.0]]]])
    mask = torch.tensor([[[True, False]]])

    assert torch.isclose(_masked_mean(values, mask), torch.tensor(2.0))


def test_alignment_mask_restores_prior_geometry_without_deleting_confidence():
    confidence = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]])
    mask = torch.tensor([[[True, False], [False, True]]])
    aligned_depth = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])
    prior_depth = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    aligned_verts = torch.arange(12.0).reshape(1, 2, 2, 3) + 100.0
    prior_verts = torch.arange(12.0).reshape(1, 2, 2, 3)

    verts, depth, restored_confidence = restore_unaligned_geometry(
        aligned_verts,
        aligned_depth,
        confidence,
        prior_verts,
        prior_depth,
        mask,
    )

    assert torch.equal(depth, torch.tensor([[[10.0, 2.0], [3.0, 40.0]]]))
    assert torch.equal(verts[mask], aligned_verts[mask])
    assert torch.equal(verts[~mask], prior_verts[~mask])
    assert torch.equal(restored_confidence, confidence)


def test_catastrophic_whole_chart_depth_collapse_is_disabled():
    prior = torch.ones((2, 2, 2))
    aligned = torch.stack((prior[0] * 1.1, prior[1] * 4.0))
    confidence = torch.ones_like(aligned)

    filtered, rejected, medians, bad_fractions = reject_catastrophic_alignment_confidences(
        confidence,
        aligned,
        prior,
        minimum_valid_pixels=1,
    )

    assert rejected.tolist() == [False, True]
    assert torch.equal(filtered[0], confidence[0])
    assert torch.count_nonzero(filtered[1]) == 0
    assert medians[0] < 0.2
    assert medians[1] > 2.0
    assert bad_fractions[1] == 1.0


def test_reference_target_prevents_rejecting_a_large_but_correct_dav2_correction():
    prior = torch.ones((1, 2, 2))
    reference = torch.full_like(prior, 4.0)
    aligned = reference.clone()
    confidence = torch.ones_like(aligned)

    (
        filtered,
        rejected,
        prior_medians,
        _,
        reference_medians,
        _,
    ) = reject_catastrophic_alignment_confidences(
        confidence,
        aligned,
        prior,
        reference_depths=reference,
        minimum_valid_pixels=1,
        return_reference_metrics=True,
    )

    assert not rejected.item()
    assert prior_medians.item() > 2.0
    assert reference_medians.item() == 0.0
    assert torch.equal(filtered, confidence)


def test_reference_target_rejects_a_global_departure_from_mast3r_observation():
    prior = torch.ones((1, 2, 2))
    reference = torch.ones_like(prior)
    aligned = torch.full_like(prior, 4.0)
    confidence = torch.ones_like(aligned)

    filtered, rejected, *_ = reject_catastrophic_alignment_confidences(
        confidence,
        aligned,
        prior,
        reference_depths=reference,
        minimum_valid_pixels=1,
        return_reference_metrics=True,
    )

    assert rejected.item()
    assert torch.count_nonzero(filtered) == 0


def test_reference_safety_check_does_not_depend_on_dav2_prior_validity():
    prior = torch.zeros((1, 2, 2))
    reference = torch.ones_like(prior)
    aligned = torch.full_like(prior, 4.0)
    confidence = torch.ones_like(aligned)

    filtered, rejected, prior_medians, _, reference_medians, _ = (
        reject_catastrophic_alignment_confidences(
            confidence,
            aligned,
            prior,
            reference_depths=reference,
            minimum_valid_pixels=1,
            return_reference_metrics=True,
        )
    )

    assert rejected.item()
    assert torch.count_nonzero(filtered) == 0
    assert torch.isnan(prior_medians).item()
    assert reference_medians.item() > 2.0
