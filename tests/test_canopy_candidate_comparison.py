import json
import pytest
from scripts.compare_canopy_candidate_controls import compare
from scripts.compare_canopy_candidate_controls import validate_matching_camera_inputs


def test_comparison_checks_training_camera_inputs_not_only_evaluation_rgb(tmp_path):
    dirs = [tmp_path/'a', tmp_path/'b']
    for d in dirs:
        d.mkdir(); (d/'cameras.json').write_text(json.dumps([dict(id=2, position=[0., 0., 0.])]))
    validate_matching_camera_inputs(dirs)
    (dirs[1]/'cameras.json').write_text(json.dumps([dict(id=2, position=[1., 0., 0.])]))
    with pytest.raises(ValueError, match='camera input'): validate_matching_camera_inputs(dirs)


def test_candidate_comparison_checks_source_not_different_initial_candidates(tmp_path):
    dirs = [tmp_path/'narrow', tmp_path/'wide']
    keys = ('tree', 'tree_interior', 'tree_boundary', 'rigid', 'hard')
    for i, d in enumerate(dirs):
        d.mkdir()
        manifest = {k: 'same' for k in ('scope', 'source_checkpoint_sha256', 'seed_audit',
                    'calibration', 'helper_sha256', 'script_sha256', 'masks_sha256',
                    'runtime_index_sha256', 'depth_sha256')}
        manifest.update(training_views=[2], excluded_views=[1], candidate_count=3, initial_opacity=.001,
                        args={'output': str(d), 'ratios': [1, 1, 1] if i == 0 else [.65, .8, 1], 'steps': 10})
        (d/'manifest.json').write_text(json.dumps(manifest))
        docs = dict(results=[dict(step=0, per_view=[dict(index=1, modes={
            'source': dict.fromkeys(keys, 15.), 'candidate': dict.fromkeys(keys, 15.+i*.01)})]),
            dict(step=10, per_view=[dict(index=1, modes={'candidate': dict.fromkeys(keys, 15.+i*.2)})])])
        (d/'metrics.json').write_text(json.dumps(docs))
    report = compare(*dirs, 10)
    assert report['metrics']['tree']['mean_wide_minus_narrow'] == pytest.approx(.2)
    assert report['frozen_audits'] == [None, None]
    assert report['training_images_per_arm'] == 10
    assert report['optimizer_updates_per_arm'] == [10, 10]
    manifest['args']['steps'] = 11
    (dirs[1]/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='explicit controlled argument'): compare(*dirs, 10)


def test_density_comparison_requires_same_seed_images_and_valid_capacity(tmp_path):
    dirs = [tmp_path/'narrow', tmp_path/'wide']
    keys = ('tree', 'tree_interior', 'tree_boundary', 'rigid', 'hard')
    for rays, d in zip((256, 1024), dirs):
        d.mkdir()
        manifest = {k: 'same' for k in ('scope', 'source_checkpoint_sha256', 'calibration',
                    'helper_sha256', 'script_sha256', 'masks_sha256', 'runtime_index_sha256', 'depth_sha256')}
        manifest.update(training_views=[2], excluded_views=[1], candidate_count=3*rays,
                        initial_opacity=.001, seed_audit=[dict(index=2, image_sha256='same', rays=rays)],
                        args=dict(output=str(d), ratios=[.65, .8, 1], rays_per_view=rays))
        (d/'manifest.json').write_text(json.dumps(manifest))
        docs = dict(results=[dict(step=0, per_view=[dict(index=1, modes={
            'source': dict.fromkeys(keys, 15.)})]),
            dict(step=10, per_view=[dict(index=1, modes={'candidate': dict.fromkeys(keys, 15.1)})])])
        (d/'metrics.json').write_text(json.dumps(docs))
    assert compare(*dirs, 10, 'rays_per_view')['metrics']['tree']['mean_wide_minus_narrow'] == 0
    manifest['candidate_count'] += 1
    (dirs[1]/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='capacity'): compare(*dirs, 10, 'rays_per_view')
    manifest['candidate_count'] -= 1
    manifest['seed_audit'][0]['image_sha256'] = 'different'
    (dirs[1]/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='seed images'): compare(*dirs, 10, 'rays_per_view')


def test_boundary_control_requires_matching_helper_and_one_argument(tmp_path):
    dirs = [tmp_path/'control', tmp_path/'boundary']
    keys = ('tree', 'tree_interior', 'tree_boundary', 'rigid', 'hard')
    for weight, d in enumerate(dirs):
        d.mkdir()
        manifest = {k: 'same' for k in ('scope', 'source_checkpoint_sha256', 'seed_audit',
            'calibration', 'helper_sha256', 'script_sha256', 'masks_sha256',
            'runtime_index_sha256', 'depth_sha256', 'boundary_preservation_helper_sha256')}
        manifest.update(training_views=[2], excluded_views=[1], candidate_count=3,
            initial_opacity=.001, args=dict(output=str(d), rigid_boundary_preservation_weight=weight))
        (d/'manifest.json').write_text(json.dumps(manifest))
        docs = dict(results=[dict(step=0, per_view=[dict(index=1, modes={
            'source': dict.fromkeys(keys, 15.)})]), dict(step=400, per_view=[dict(index=1,
            modes={'candidate': dict.fromkeys(keys, 15.1)})])])
        (d/'metrics.json').write_text(json.dumps(docs))
    assert compare(*dirs, 400, 'rigid_boundary_preservation_weight')['metrics']['tree']['mean_wide_minus_narrow'] == 0
    manifest['boundary_preservation_helper_sha256'] = 'changed'
    (dirs[1]/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='boundary preservation helper'):
        compare(*dirs, 400, 'rigid_boundary_preservation_weight')
