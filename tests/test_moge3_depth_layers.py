import numpy as np
import pytest
from outdoor.moge3_depth_layers import reduce_depth_layers
from scripts.build_moge3_chart_base import _resize_scalar, _resize_normal
from outdoor.moge3_chart_base import chart_base_variants


def test_two_surfaces_keep_native_rays_and_depths():
    d = np.array([[2., 9., 9.], [2., 9., 9.], [2., 9., 9.]])
    r = reduce_depth_layers(d, np.ones_like(d, bool), (1, 1))
    assert r['mixed'].item() and not r['continuous_valid'].item()
    assert r['front_fraction'].item() == pytest.approx(1/3)
    assert r['back_fraction'].item() == pytest.approx(2/3)
    for name, expected in [('front', 2.), ('back', 9.)]:
        x, y = r[name+'_native_x'].item(), r[name+'_native_y'].item()
        assert d[y, x] == r[name+'_depth'].item() == expected


def test_multiple_modes_withhold_proposal_and_empty_cells():
    d = np.array([[1., 3., 9.], [np.nan, 0., np.inf]])
    r = reduce_depth_layers(d, np.ones_like(d, bool), (2, 1))
    assert r['unresolved'][0, 0]
    assert not r['continuous_valid'].any()
    assert not r['front_depth'].any() and not r['back_depth'].any()
    assert (r['front_native_x'] == -1).all()


def test_single_pixel_and_noninteger_reduction():
    r = reduce_depth_layers(np.ones((2, 2)), np.ones((2, 2), bool), (2, 2))
    assert r['continuous_valid'].all()
    assert not r['back_fraction'].any()
    d=np.arange(1,36,dtype=float).reshape(5,7)
    r=reduce_depth_layers(d,np.ones_like(d,bool),(3,4),separation_ratio=100)
    assert (r['valid_fraction']==1).all()
    np.testing.assert_array_equal(d[r['front_native_y'],r['front_native_x']],r['front_depth'])
    with pytest.raises(ValueError):
        reduce_depth_layers(np.ones((3, 3)), np.ones((3, 3), bool), (4, 4))


def test_masked_nan_does_not_poison_reduced_valid_observations():
    d = np.array([[3., np.nan], [3., 3.]], np.float32)
    out, valid = _resize_scalar(d, np.isfinite(d), (1, 1))
    assert valid.item() and out.item() == 3.
    normal = np.zeros((2, 2, 3), np.float32); normal[..., 2] = 1
    normal[0, 1] = np.nan
    out, valid = _resize_normal(normal, np.isfinite(d), (1, 1))
    assert valid.item()
    np.testing.assert_array_equal(out[0, 0], [0., 0., 1.])


def test_depth_without_normal_remains_explicit_unverified_depth():
    args = dict(rigid_valid=np.ones((2, 2), bool), refinement_sigma=np.zeros((2, 2)),
                normal_direct_camera=np.zeros((2, 2, 3)), normal_depth_camera=np.zeros((2, 2, 3)),
                depth_normal_valid=np.zeros((2, 2), bool))
    old = chart_base_variants(np.ones((2, 2))*3, np.ones((2, 2))*2, **args)
    new = chart_base_variants(np.ones((2, 2))*3, np.ones((2, 2))*2,
                              depth_valid_independent_of_normal=True, **args)
    assert (old['depth_moge3'] == 3).all()
    assert (new['depth_moge3'] == 2).all()
    assert new['moge3_depth_valid'].all() and not new['moge3_normal_valid'].any()
