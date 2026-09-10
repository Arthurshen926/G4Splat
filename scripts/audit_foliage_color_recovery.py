"""Read-only sufficient test for all-direction SH color clamp death."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from types import SimpleNamespace
import torch
from outdoor.hybrid_gaussian_renderer import persistent_static_evidence_mask, LAYER_DYNAMIC_LEAF
from outdoor.moge3_evidence import sha256_file


def all_direction_color_upper_bound(features):
    if features.ndim != 3 or features.shape[2] != 3:
        raise ValueError('Complete real SH bands and three color channels required')
    coefficients = features.shape[1]
    if coefficients < 1 or math.isqrt(coefficients)**2 != coefficients:
        raise ValueError('Complete real SH bands and three color channels required')
    # Addition theorem plus Cauchy-Schwarz: sum(non-DC Y_lm^2)=(K-1)/(4pi).
    return (features[:, 0]*.28209479177387814+.5
            +features[:, 1:].square().sum(1).sqrt()*math.sqrt((coefficients-1)/(4*math.pi)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(4)
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)['foliage']
    features = state['features']; bound = all_direction_color_upper_bound(features)
    foliage = SimpleNamespace(**state, dynamic_leaf_mask=state['layer_role'] == LAYER_DYNAMIC_LEAF)
    persistent = persistent_static_evidence_mask(foliage)
    records = []
    for margin in (0., 1e-6, .001, .01):
        dead = bound < -margin; rows = dead.any(1)
        records.append(dict(negative_margin=margin, channels=int(dead.sum()), rows=int(rows.sum()),
            persistent_channels=int(dead[persistent].sum()), persistent_rows=int(rows[persistent].sum()),
            by_verification_state={str(int(k)): int((rows&(state['verification_state'] == k)).sum())
                                   for k in state['verification_state'].unique()}))
    report = dict(source_checkpoint_sha256=sha256_file(a.checkpoint), source_rows=len(features),
        persistent_rows=int(persistent.sum()), records=records, minimum_upper_bound=float(bound.min()),
        scope='sufficient_all_direction_dead_channel_bound__not_rendered_impact',
        limitations=['A nonnegative bound does not prove every camera has recovery gradients',
                     'No source parameter was changed; this is not evidence for blanket SH projection'])
    with a.output.open('x') as f: json.dump(report, f, indent=2)
    print(json.dumps(report))


if __name__ == '__main__': main()
