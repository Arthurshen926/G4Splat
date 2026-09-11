"""Bounded conditional LP replay preparation; does not modify any checkpoint.

Keep every ray constraint, but vary only strong positive-ray contributors and
enough existing visible-ray extinction contributors to permit the upper bound.
Minimum change is only within this subset; compare slack with the full LP.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from scripts.canopy_joint_optical_feasibility import solve_optical_intervals


def select_columns(K, lower, upper, reference, caps, count):
    active = np.zeros(K.shape[1], dtype=bool)
    for i in np.flatnonzero(lower > 0):
        row = K.getrow(i)
        order = np.argsort(-(row.data * caps[row.indices]))[:count]
        active[row.indices[order]] = True
    for i in np.flatnonzero(np.isfinite(upper)):
        row = K.getrow(i)
        ids = row.indices[~active[row.indices]]
        values = row.data[~active[row.indices]] * reference[ids]
        if values.sum() > upper[i]:
            order = np.argsort(-values)
            n = np.searchsorted(np.cumsum(values[order]), values.sum()-upper[i]) + 1
            active[ids[order[:n]]] = True
    return active


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--columns-per-positive-ray', type=int, default=32)
    parser.add_argument('--source-count', type=int, required=True)
    a = parser.parse_args()
    if a.columns_per_positive_ray < 1 or a.source_count < 1:
        raise ValueError('Positive bounds required')
    data = np.load(a.source/'joint_solution.npz')
    K = sparse.load_npz(a.source/'joint_kernel.npz').tocsr()
    source = json.loads((a.source/'joint_feasibility.json').read_text())
    caps = np.where(data['primitive_ids'] < a.source_count, -np.log(.005), -np.log(1e-6))
    active = select_columns(K, data['lower'], data['upper'], data['current_tau'], caps, a.columns_per_positive_ray)
    fixed = K[:, ~active] @ data['current_tau'][~active]
    excess = np.maximum(fixed-data['upper'], 0)
    if excess.max() > 1e-7:
        raise RuntimeError('Restricted fixed contribution already violates an upper bound')
    a.output.mkdir(exist_ok=False)
    report = dict(source=str(a.source), scope='all_rays__restricted_parameter_subset__not_global_minimum_change',
                  active_variables=int(active.sum()), columns_per_positive_ray=a.columns_per_positive_ray,
                  full_variable_optimum_slack=source['weighted_slack'])
    try:
        result = solve_optical_intervals(K[:, active], np.maximum(data['lower']-fixed, 0),
            np.maximum(data['upper']-fixed, 0), caps[active], reference_tau=data['current_tau'][active])
        # Preserve solver precision here; native float32 conversion is assessed
        # separately during renderer replay, not silently in the LP result.
        tau = data['current_tau'].astype(np.float64); tau[active] = result['tau']
        optical = K @ tau
        slack = np.maximum(data['lower']-optical, 0).sum()+np.maximum(optical-data['upper'], 0).sum()
        delta = tau-data['current_tau']
        report.update(weighted_slack=float(slack), slack_above_full_optimum=float(slack-source['weighted_slack']),
                      changed_rows=int((abs(delta)>1e-6).sum()), l1_delta=float(abs(delta).sum()),
                      max_delta=float(abs(delta).max()))
        exported={k:tau if k=='tau' else data[k] for k in data.files}
        # scipy may store sparse indices as int32; replay identities have an
        # explicit int64 schema, independently of sparse storage precision.
        exported['primitive_ids']=data['primitive_ids'].astype(np.int64)
        np.savez(a.output/'joint_solution.npz', **exported)
    except RuntimeError as error:
        report['failed'] = str(error)
    (a.output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
