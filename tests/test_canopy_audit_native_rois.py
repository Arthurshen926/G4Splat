from scripts.audit_canopy_candidate_contribution import scaled_regression_roi


def test_regression_rois_scale_with_image_not_fit_target():
    box=(470,205,630,335)
    assert scaled_regression_roi(box,640,360)==box
    assert scaled_regression_roi(box,1920,1080)==(1410,615,1890,1005)
