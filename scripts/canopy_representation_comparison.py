"""Validate the declared coupled resolution intervention before comparing RGB."""
import math
from scripts.canopy_representation_resolution import representation_recipe, validate_representation_scope


def validate_recipe_manifest(manifest):
    args = manifest['args']
    recipe = representation_recipe(args['representation_policy'])
    validate_representation_scope(native_rgb=args['native_rgb_training'],
        shape_cohort=args['candidate_shape_cohort'], kernel=args['leaf_optical_kernel'])
    if (args['rays_per_view'] != recipe.rays_per_view
            or args['position_radius_sigmas'] != recipe.position_radius_sigmas
            or args['depth_search_policy'] != 'photometric_volume'):
        raise ValueError('Recipe-derived arguments do not match declared policy')
    records = manifest['representation_seed_audit']
    seeds = [row for row in manifest['seed_audit'] if row['rays']]
    if not records or len(records) != len(seeds):
        raise ValueError('Every emitted seed group must have a representation budget')
    for row, seed in zip(records, seeds):
        if (row['index'] != seed['index'] or row['rays'] != seed['rays']
                or row['rays'] != min(row['population'], recipe.rays_per_view)
                or row['pixel_sigma'] != recipe.pixel_sigma
                or not math.isclose(row['initial_peak_opacity'], recipe.initial_peak_opacity(row['population']), abs_tol=1e-10)
                or not math.isclose(row['nominal_projected_area_ratio'], recipe.nominal_projected_area_ratio(row['population']), abs_tol=1e-10)):
            raise ValueError('Invalid per-seed representation initialization')
    if manifest['candidate_count'] != 3 * sum(row['rays'] for row in records):
        raise ValueError('Candidate count disagrees with representation seeds')
    initial = manifest['initial_opacity']
    peaks = [row['initial_peak_opacity'] for row in records]
    if (initial['policy'] != 'per_seed_nominal_projected_area_budget'
            or not math.isclose(initial['minimum'], min(peaks), abs_tol=1e-7)
            or not math.isclose(initial['maximum'], max(peaks), abs_tol=1e-7)):
        raise ValueError('Initial opacity budget does not match seeds')
    return [(row['index'], row['population']) for row in records]
