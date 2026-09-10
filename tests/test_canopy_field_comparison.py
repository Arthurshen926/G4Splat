from copy import deepcopy
import pytest
from scripts.compare_canopy_field_controls import KEYS, compare


def document(delta=0):
    return {'training_views': [1, 2], 'results': [
        {'step': step, 'per_view': [
            {'index': index, 'modes': {mode: {key: 10 + (delta if step else 0)
              for key in KEYS} for mode in ('static_canonical', 'interpolated_deformed')}}
            for index in (3, 4)]} for step in (0, 800)]}


def test_paired_comparison_reports_change_from_shared_initial():
    r = compare(document(1), document(1.5), 800, 'interpolated_deformed')['metrics']['tree']
    assert r['mean_delta'] == .5 and r['experiment_from_initial'] == 1.5
    assert r['positive_delta_views'] == 2


@pytest.mark.parametrize('change', ['baseline', 'duplicate', 'fitted', 'nonfinite'])
def test_comparison_rejects_unmatched_or_invalid_evidence(change):
    a = document(); b = deepcopy(a)
    if change == 'baseline':
        b['results'][0]['per_view'][0]['modes']['static_canonical']['tree'] = 11
    elif change == 'duplicate':
        b['results'][1]['per_view'].append(b['results'][1]['per_view'][0])
    elif change == 'fitted':
        a['training_views'] = b['training_views'] = [1, 3]
    else:
        b['results'][1]['per_view'][0]['modes']['static_canonical']['tree'] = float('nan')
    with pytest.raises(ValueError):
        compare(a, b, 800, 'static_canonical')
