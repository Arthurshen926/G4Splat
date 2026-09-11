import copy
import pytest
from scripts.canopy_representation_comparison import validate_recipe_manifest


def manifest():
    return dict(args=dict(representation_policy='dense_fine',native_rgb_training=True,
        candidate_shape_cohort=None,leaf_optical_kernel='native',rays_per_view=4096,
        position_radius_sigmas=16.,depth_search_policy='photometric_volume'),
        seed_audit=[dict(index=4,rays=512)],candidate_count=1536,
        representation_seed_audit=[dict(index=4,population=512,rays=512,pixel_sigma=.6,
            initial_peak_opacity=.004,nominal_projected_area_ratio=.25)],
        initial_opacity=dict(policy='per_seed_nominal_projected_area_budget',minimum=.004,maximum=.004))


def test_declared_budget_is_validated():
    original=manifest()
    assert validate_recipe_manifest(original)==[(4,512)]
    for key,value in [('pixel_sigma',1.2),('initial_peak_opacity',.001),
                      ('nominal_projected_area_ratio',1.),('rays',511)]:
        data=copy.deepcopy(original);data['representation_seed_audit'][0][key]=value
        with pytest.raises(ValueError):validate_recipe_manifest(data)
    data=copy.deepcopy(original);data['args']['position_radius_sigmas']=8.
    with pytest.raises(ValueError):validate_recipe_manifest(data)
