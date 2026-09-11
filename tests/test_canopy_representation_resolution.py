import pytest
from scripts.canopy_representation_resolution import representation_recipe, validate_representation_scope


def test_nominal_area_and_motion_extent():
    coarse = representation_recipe('coarse')
    fine = representation_recipe('dense_fine')
    assert coarse.nominal_projected_area_ratio(10000) == 1.
    assert fine.nominal_projected_area_ratio(10000) == 1.
    assert coarse.pixel_sigma * coarse.position_radius_sigmas == fine.pixel_sigma * fine.position_radius_sigmas


def test_capped_population_is_not_claimed_optically_equivalent():
    fine = representation_recipe('dense_fine')
    assert fine.nominal_projected_area_ratio(512) == .25
    assert fine.nominal_projected_area_ratio(2048) == .5
    assert fine.nominal_projected_area_ratio(0) is None
    with pytest.raises(ValueError): fine.nominal_projected_area_ratio(-1)


@pytest.mark.parametrize('population',[1,512,1024,2048,4096,10000])
def test_capped_seed_initial_budget(population):
    fine = representation_recipe('dense_fine')
    opacity = fine.initial_peak_opacity(population)
    assert .001 <= opacity <= .004
    assert fine.nominal_projected_area_ratio(population) * opacity == pytest.approx(.001)


def test_no_resolution_or_cohort_confounds():
    validate_representation_scope(native_rgb=True, shape_cohort=None, kernel='native')
    for kwargs in [dict(native_rgb=False,shape_cohort=None,kernel='native'),
                   dict(native_rgb=True,shape_cohort='old.pth',kernel='native'),
                   dict(native_rgb=True,shape_cohort=None,kernel='projected_tau')]:
        with pytest.raises(ValueError): validate_representation_scope(**kwargs)
    with pytest.raises(ValueError): representation_recipe('unknown')
