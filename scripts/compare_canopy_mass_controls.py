"""Strict single-variable mass-coordinate calibration comparison."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_canopy_field_controls import rows_at


def compare(control, experiment, step):
    paths = [control, experiment]
    manifests = [json.loads((p/'manifest.json').read_text()) for p in paths]
    for key in ('scope', 'source_checkpoint_sha256', 'training_views', 'excluded_views',
                'eligible_rows', 'masks_sha256', 'script_sha256', 'helper_sha256'):
        if manifests[0][key] != manifests[1][key]: raise ValueError(f'Unmatched mass calibration: {key}')
    if [m['args']['consistent'] for m in manifests] != [False, True]:
        raise ValueError('Legacy then consistent Jacobian controls required')
    if ({k: v for k, v in manifests[0]['args'].items() if k not in ('output', 'consistent')}
            != {k: v for k, v in manifests[1]['args'].items() if k not in ('output', 'consistent')}):
        raise ValueError('Only the gradient Jacobian may differ')
    docs = [json.loads((p/'metrics.json').read_text()) for p in paths]
    source = [rows_at(d, 0, 'source') for d in docs]
    rows = [rows_at(d, step, 'refined') for d in docs]
    keys = set(manifests[0]['excluded_views'])
    if any(set(v) != keys for v in source+rows) or keys&set(manifests[0]['training_views']):
        raise ValueError('Complete excluded camera cohort required')
    metrics = {}
    for metric in ('tree', 'tree_interior', 'tree_boundary', 'rigid', 'hard'):
        values = []
        for index in keys:
            s, t = [v[index][metric] for v in source]; a, b = [v[index][metric] for v in rows]
            if all(x is None for x in (s, t, a, b)): continue
            if any(x is None or not math.isfinite(x) for x in (s, t, a, b)) or abs(s-t) > 1e-5:
                raise ValueError('Finite matched original source metrics required')
            values.append(dict(view=index, corrected_minus_legacy=b-a,
                               legacy_minus_source=a-s, corrected_minus_source=b-s))
        metrics[metric] = dict(mean_corrected_minus_legacy=mean(v['corrected_minus_legacy'] for v in values),
            mean_legacy_minus_source=mean(v['legacy_minus_source'] for v in values),
            mean_corrected_minus_source=mean(v['corrected_minus_source'] for v in values),
            improved_views=sum(v['corrected_minus_source'] > 1e-5 for v in values),
            worst_views=sorted(values, key=lambda v: v['corrected_minus_source'])[:8], per_view=values)
    audits = [json.loads((p/'frozen_parameter_audit.json').read_text())
              if (p/'frozen_parameter_audit.json').exists() else None for p in paths]
    return dict(step=step, control=str(control), experiment=str(experiment), metrics=metrics,
                frozen_audits=audits, descriptive_only=True,
                warning='Frozen buildings are not proof that their rendered pixels are preserved')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('control', 'experiment', 'output'): p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--step', type=int, required=True); a = p.parse_args()
    report = compare(a.control, a.experiment, a.step)
    with a.output.open('x') as f: json.dump(report, f, indent=2)
    print(json.dumps({k: {q: v for q, v in row.items() if q not in ('worst_views', 'per_view')}
                     for k, row in report['metrics'].items()}))


if __name__ == '__main__': main()
