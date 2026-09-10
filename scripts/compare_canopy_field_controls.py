"""Paired per-view reporting; no statistical independence or quality-pass claim."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, median

KEYS = ('tree', 'tree_interior', 'tree_boundary', 'rigid', 'hard', 'tree_alpha')


def rows_at(document, step, mode):
    matches = [row for row in document['results'] if row['step'] == step]
    if len(matches) != 1:
        raise ValueError('Exactly one completed evaluation at requested step required')
    rows = matches[0]['per_view']
    result = {row['index']: row['modes'][mode] for row in rows}
    if not result or len(result) != len(rows):
        raise ValueError('Nonempty unique evaluation views required')
    return result


def compare(control, experiment, step, mode):
    if control['training_views'] != experiment['training_views']:
        raise ValueError('Matched training cameras and order required')
    before_a = rows_at(control, 0, 'static_canonical')
    before_b = rows_at(experiment, 0, mode)
    a = rows_at(control, step, 'static_canonical')
    b = rows_at(experiment, step, mode)
    if not (a.keys() == b.keys() == before_a.keys() == before_b.keys()):
        raise ValueError('Matched evaluation cameras required')
    if set(a) & set(control['training_views']):
        raise ValueError('Evaluation camera was fitted')
    result = {}
    for key in KEYS:
        deltas = []
        for index in a:
            values = [row[index][key] for row in (before_a, before_b, a, b)]
            if all(value is None for value in values):
                continue
            if any(value is None or not math.isfinite(value) for value in values):
                raise ValueError('Aligned finite regional metrics required')
            if abs(values[0] - values[1]) > 1.e-5:
                raise ValueError('Initial native render metrics differ')
            deltas.append({'view': index, 'delta': values[3] - values[2],
                           'control_from_initial': values[2] - values[0],
                           'experiment_from_initial': values[3] - values[1]})
        if not deltas:
            result[key] = None
            continue
        ordered = sorted(deltas, key=lambda row: row['delta'])
        result[key] = dict(count=len(deltas), mean_delta=mean(row['delta'] for row in deltas),
            median_delta=median(row['delta'] for row in deltas),
            positive_delta_views=sum(row['delta'] > 1.e-6 for row in deltas),
            negative_delta_views=sum(row['delta'] < -1.e-6 for row in deltas),
            control_from_initial=mean(row['control_from_initial'] for row in deltas),
            experiment_from_initial=mean(row['experiment_from_initial'] for row in deltas),
            lowest_deltas=ordered[:8], highest_deltas=ordered[-8:], per_view=deltas)
    return {'step': step, 'experimental_mode': mode, 'metrics': result,
            'scope': 'paired_descriptive_only__alpha_increase_is_not_automatically_improvement'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--modes',nargs='+',choices=('static_canonical','interpolated_deformed'),
                   default=['static_canonical','interpolated_deformed'])
    a = p.parse_args()
    control = json.loads(a.control.read_text())
    experiment = json.loads(a.experiment.read_text())
    reports = [compare(control, experiment, a.step, mode)
               for mode in a.modes]
    with a.output.open('x') as stream:
        json.dump(reports, stream, indent=2)
    print(json.dumps([{**r, 'metrics': {k: {x: y for x, y in v.items()
          if x not in ('per_view', 'lowest_deltas', 'highest_deltas')}
          if v else None for k, v in r['metrics'].items()}} for r in reports]))


if __name__ == '__main__':
    main()
