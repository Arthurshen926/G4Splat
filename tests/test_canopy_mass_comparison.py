import json
import pytest
from scripts.compare_canopy_mass_controls import compare


def test_mass_comparison_rejects_other_changes_and_missing_views(tmp_path):
    paths = [tmp_path/'control', tmp_path/'experiment']
    metrics = ('tree', 'tree_interior', 'tree_boundary', 'rigid', 'hard')
    for i, p in enumerate(paths):
        p.mkdir()
        manifest = {k: 'same' for k in ('scope', 'source_checkpoint_sha256', 'masks_sha256', 'script_sha256', 'helper_sha256')}
        manifest.update(training_views=[2], excluded_views=[1], eligible_rows=100,
                        args=dict(consistent=bool(i), output=str(p), steps=8))
        (p/'manifest.json').write_text(json.dumps(manifest))
        doc = dict(results=[dict(step=0, per_view=[dict(index=1, modes={'source': dict.fromkeys(metrics, 15.)})]),
            dict(step=8, per_view=[dict(index=1, modes={'refined': dict.fromkeys(metrics, 15.+.1*i)})])])
        (p/'metrics.json').write_text(json.dumps(doc))
    result = compare(*paths, 8)
    assert result['metrics']['tree']['mean_corrected_minus_legacy'] == pytest.approx(.1)
    assert result['frozen_audits'] == [None, None]
    manifest['args']['steps'] = 9
    (paths[1]/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Only the gradient'): compare(*paths, 8)
    manifest['args']['steps'] = 8
    (paths[1]/'manifest.json').write_text(json.dumps(manifest))
    doc['results'][-1]['per_view'] = []
    (paths[1]/'metrics.json').write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='evaluation views|cohort'): compare(*paths, 8)
