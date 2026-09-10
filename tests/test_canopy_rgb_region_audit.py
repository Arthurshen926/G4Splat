import torch
from scripts.canopy_rgb_region_audit import rgb_region_audit


def test_rigid_interface_partition_does_not_hide_interior_regression():
    source = torch.zeros(3, 2, 3)
    updated = source.clone(); updated[:, 0, 0] = .5
    rigid = torch.tensor([[True, True, False], [True, True, False]])
    interface = torch.tensor([[False, True, True], [False, True, True]])
    rows = rgb_region_audit(source, updated, source, rigid, interface)
    assert rows['rigid_all']['pixels'] == 4
    assert rows['rigid_tree_interface']['pixels'] == 2
    assert rows['rigid_away_from_tree_interface']['pixels'] == 2
    assert rows['rigid_tree_interface']['rgb_change_l1'] == 0
    assert rows['rigid_away_from_tree_interface']['worsened_fraction'] == .5
    assert rows['rigid_away_from_tree_interface']['updated_psnr'] < rows['rigid_all']['updated_psnr']


def test_empty_rigid_regions_are_not_perfect_psnr_observations():
    rgb = torch.zeros(3, 2, 2); mask = torch.zeros(2, 2, dtype=torch.bool)
    rows = rgb_region_audit(rgb, rgb, rgb, mask, ~mask)
    assert all(row['pixels'] == 0 and row['source_psnr'] is None for row in rows.values())
