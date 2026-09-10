"""Matched candidate-depth reporting, without claiming independent samples."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_canopy_field_controls import rows_at


def validate_matching_camera_inputs(directories):
    for name in ('cameras.json', 'camera_intrinsics_contract.json'):
        paths = [directory/name for directory in directories]
        present = [path.exists() for path in paths]
        if any(present) and (not all(present) or json.loads(paths[0].read_text()) != json.loads(paths[1].read_text())):
            raise ValueError(f'Unmatched camera input: {name}')


def compare(narrow, wide, step, changed_argument='ratios'):
    if changed_argument not in ('ratios', 'ray_log_radius', 'rigid_rgb_preservation_weight', 'search_depth_center',
                                'rays_per_view', 'footprint_log_radius', 'noncanopy_rgb_target', 'position_radius_sigmas', 'candidate_gain',
                                'surface_rgb_feasibility_weight', 'depth_search_policy', 'source_rest_step_multiplier',
                                'rigid_boundary_preservation_weight', 'spatial_footprint_log_radius', 'source_opacity_lr', 'gradient_accumulation',
                                'canopy_only_color_gradients', 'seed_ray_policy', 'source_position_radius_sigmas',
                                'surface_rgb_feasibility_domain'):
        raise ValueError('One explicit controlled argument required')
    manifests = [json.loads((d/'manifest.json').read_text()) for d in (narrow, wide)]
    validate_matching_camera_inputs((narrow, wide))
    if manifests[0].get('relative_radiance_helper_sha256') != manifests[1].get('relative_radiance_helper_sha256'):
        raise ValueError('Unmatched relative radiance helper')
    if changed_argument == 'surface_rgb_feasibility_domain':
        for key in ('relative_floor_initial_scales', 'relative_floor_normalization'):
            if manifests[0].get(key) != manifests[1].get(key):
                raise ValueError('Unmatched initial radiance-gradient normalization')
    if manifests[0].get('source_position_helper_sha256') != manifests[1].get('source_position_helper_sha256'):
        raise ValueError('Unmatched supported source-position helper')
    if manifests[0].get('source_position_authority') != manifests[1].get('source_position_authority'):
        raise ValueError('Unmatched supported source-position authority')
    if manifests[0].get('gradient_accumulation_helper_sha256') != manifests[1].get('gradient_accumulation_helper_sha256'):
        raise ValueError('Unmatched gradient accumulation helper')
    if manifests[0].get('color_gradient_ownership_helper_sha256') != manifests[1].get('color_gradient_ownership_helper_sha256'):
        raise ValueError('Unmatched color gradient ownership helper')
    if manifests[0].get('frozen_reference_cache_helper_sha256') != manifests[1].get('frozen_reference_cache_helper_sha256'):
        raise ValueError('Unmatched immutable reference cache helper')
    if manifests[0].get('seed_ray_sampling_helper_sha256') != manifests[1].get('seed_ray_sampling_helper_sha256'):
        raise ValueError('Unmatched seed ray sampling helper')
    if manifests[0].get('boundary_preservation_helper_sha256') != manifests[1].get('boundary_preservation_helper_sha256'):
        raise ValueError('Unmatched boundary preservation helper')
    if manifests[0].get('spatial_footprint_helper_sha256') != manifests[1].get('spatial_footprint_helper_sha256'):
        raise ValueError('Unmatched joint spatial footprint helper')
    if manifests[0].get('photometric_depth_helper_sha256') != manifests[1].get('photometric_depth_helper_sha256'):
        raise ValueError('Unmatched photometric depth helper')
    if manifests[0].get('source_opacity_projection_helper_sha256') != manifests[1].get('source_opacity_projection_helper_sha256'):
        raise ValueError('Unmatched source opacity projection helper')
    for key in ('scope', 'source_checkpoint_sha256', 'training_views', 'excluded_views',
                'seed_audit', 'candidate_count', 'initial_opacity', 'calibration',
                'helper_sha256', 'script_sha256', 'masks_sha256', 'runtime_index_sha256', 'depth_sha256'):
        if changed_argument == 'rays_per_view' and key in ('seed_audit', 'candidate_count'): continue
        if manifests[0][key] != manifests[1][key]: raise ValueError(f'Unmatched candidate contract: {key}')
    if changed_argument == 'rays_per_view':
        image_identity = lambda m: [(r['index'], r['image_sha256']) for r in m['seed_audit']]
        if image_identity(manifests[0]) != image_identity(manifests[1]): raise ValueError('Changed seed images')
        for m in manifests:
            if m['candidate_count'] != len(m['args']['ratios'])*sum(r['rays'] for r in m['seed_audit']):
                raise ValueError('Candidate capacity does not match sampled ray counts')
    if manifests[0].get('refinement_helper_sha256') != manifests[1].get('refinement_helper_sha256'):
        raise ValueError('Unmatched refinement helper')
    if manifests[0].get('rgb_ownership_helper_sha256') != manifests[1].get('rgb_ownership_helper_sha256'):
        raise ValueError('Unmatched RGB ownership helper')
    if manifests[0].get('joint_optics_helper_sha256') != manifests[1].get('joint_optics_helper_sha256'):
        raise ValueError('Unmatched source optical authority helper')
    if manifests[0].get('surface_rgb_feasibility_helper_sha256') != manifests[1].get('surface_rgb_feasibility_helper_sha256'):
        raise ValueError('Unmatched RGB feasibility helper')
    if manifests[0].get('stratified_depth_helper_sha256') != manifests[1].get('stratified_depth_helper_sha256'):
        raise ValueError('Unmatched depth search helper')
    if manifests[0].get('diagnostic_checkpoint_helper_sha256') != manifests[1].get('diagnostic_checkpoint_helper_sha256'):
        raise ValueError('Unmatched diagnostic checkpoint publisher')
    if manifests[0].get('directional_sh_step_helper_sha256') != manifests[1].get('directional_sh_step_helper_sha256'):
        raise ValueError('Unmatched directional SH step policy')
    allowed = {'output', changed_argument}
    if ({k: v for k, v in manifests[0]['args'].items() if k not in allowed}
            != {k: v for k, v in manifests[1]['args'].items() if k not in allowed}):
        raise ValueError('Only the explicit controlled argument may change between paired candidates')
    docs = [json.loads((d/'metrics.json').read_text()) for d in (narrow, wide)]
    sources = [rows_at(d, 0, 'source') for d in docs]
    rows = [rows_at(d, step, 'candidate') for d in docs]
    if not (sources[0].keys() == sources[1].keys() == rows[0].keys() == rows[1].keys()
            == set(manifests[0]['excluded_views'])):
        raise ValueError('Matched full evaluation cohort required')
    if set(rows[0]) & set(manifests[0]['training_views']): raise ValueError('Evaluation leakage')
    output = {}
    for key in ('tree', 'tree_interior', 'tree_boundary', 'rigid', 'hard'):
        values = []
        for index in rows[0]:
            a, b = [r[index][key] for r in rows]; s, t = [r[index][key] for r in sources]
            if all(v is None for v in (a, b, s, t)): continue
            if any(v is None or not math.isfinite(v) for v in (a, b, s, t)) or abs(s-t) > 1e-5:
                raise ValueError('Finite metrics and identical disabled-candidate source required')
            values.append(dict(view=index, wide_minus_narrow=b-a, narrow_minus_source=a-s,
                               wide_minus_source=b-s))
        output[key] = dict(mean_wide_minus_narrow=mean(v['wide_minus_narrow'] for v in values),
                           mean_narrow_minus_source=mean(v['narrow_minus_source'] for v in values),
                           mean_wide_minus_source=mean(v['wide_minus_source'] for v in values),
                           median_wide_minus_source=median(v['wide_minus_source'] for v in values),
                           wide_improved_views=sum(v['wide_minus_source'] > 1e-5 for v in values),
                           worst_wide_views=sorted(values, key=lambda v: v['wide_minus_source'])[:8],
                           per_view=values)
    frozen = []
    for directory in (narrow, wide):
        path = directory/'frozen_parameter_audit.json'
        frozen.append(json.loads(path.read_text()) if path.exists() else None)
    return dict(step=step, changed_argument=changed_argument, control=str(narrow), experiment=str(wide),
                training_images_per_arm=step,
                optimizer_updates_per_arm=[math.ceil(step/m['args'].get('gradient_accumulation', 1)) for m in manifests],
                descriptive_only=True, metrics=output, frozen_audits=frozen,
                warning='Frozen parameters do not guarantee preserved building renders; inspect per-view RGB.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('narrow', 'wide', 'output'): p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--changed-argument', choices=('ratios', 'ray_log_radius', 'rigid_rgb_preservation_weight',
                   'search_depth_center', 'rays_per_view', 'footprint_log_radius', 'noncanopy_rgb_target',
                   'position_radius_sigmas', 'candidate_gain', 'surface_rgb_feasibility_weight', 'depth_search_policy',
                   'source_rest_step_multiplier', 'rigid_boundary_preservation_weight',
                   'spatial_footprint_log_radius', 'source_opacity_lr', 'gradient_accumulation',
                   'canopy_only_color_gradients', 'seed_ray_policy', 'source_position_radius_sigmas',
                   'surface_rgb_feasibility_domain'), default='ratios')
    a = p.parse_args(); report = compare(a.narrow, a.wide, a.step, a.changed_argument)
    with a.output.open('x') as f: json.dump(report, f, indent=2)
    print(json.dumps({k: {q: v for q, v in row.items() if q not in ('per_view', 'worst_wide_views')}
                      for k, row in report['metrics'].items()}))


if __name__ == '__main__': main()
