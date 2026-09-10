import pytest
from scripts.seed_bound_moge_diagnostic import apply_seed_calibration


class Geometry:
    def configure_moge3_metric_scale(self,value):self.scale=value
    def configure_moge3_canopy_depth_scales(self,value):self.profiles=value


def test_diagnostic_applies_both_scales_not_global_only():
    geometry=Geometry()
    audit={'moge3_exact_k_front_hit':{'metric_to_cambridge_scale':.8},
           'fresh_canonical_leaves':{'depth_profiles_by_image':{'seq2__frame1':1.9}}}
    apply_seed_calibration(geometry,audit,.8)
    assert geometry.scale==.8 and geometry.profiles=={'seq2__frame1':1.9}
    with pytest.raises(ValueError,match='differs'):
        apply_seed_calibration(geometry,audit,1.)


def test_missing_canopy_profile_is_not_silently_identity():
    audit={'moge3_exact_k_front_hit':{'metric_to_cambridge_scale':.8},
           'fresh_canonical_leaves':{'depth_profiles_by_image':{}}}
    with pytest.raises(ValueError,match='Missing'):
        apply_seed_calibration(Geometry(),audit,.8)
