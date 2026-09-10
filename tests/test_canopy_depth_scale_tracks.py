import numpy as np
from scripts.audit_canopy_depth_scale_tracks import depth_errors


def test_depth_scale_comparison_is_signed_and_rejects_invalid():
    out = depth_errors([2., 4., np.nan, -1.], [1., 2., 1., 1.])
    assert out['count'] == 2
    assert np.isclose(out['median_log_ratio'], np.log(2))
    assert np.isclose(out['median_absolute_log_error'], np.log(2))
    assert depth_errors([], [])['count'] == 0
