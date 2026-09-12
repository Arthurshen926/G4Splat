"""Bounded native surface-refinement pair on a saved stronger canopy state."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import json
import hashlib

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--arm',choices=['geometry','fixed'],required=True)
p.add_argument('--phase',choices=['smoke','full'],default='smoke')
a=p.parse_args()
root=Path('/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1')
version=(517 if a.arm=='geometry' else 518)+(2 if a.phase=='full' else 0)
steps=304 if a.phase=='full' else 8
run=root/f'diagnostic_v{version}_refined_{a.arm}_{steps}'
if a.phase=='full':
    for n in ['diagnostic_v517_refined_geometry_8','diagnostic_v518_refined_fixed_8']:
        smoke=root/n
        if not (smoke/'eval_000008.json').exists() or not json.loads((smoke/'frozen_parameter_audit.json').read_text())['unchanged']:
            raise RuntimeError('Both completed CUDA smokes required')
        manifest=json.loads((smoke/'manifest.json').read_text())
        for field,file in [('script_sha256','scripts/calibrate_native_detail_suffix.py'),
                           ('canopy_loader_sha256','scripts/load_refined_canopy.py'),
                           ('geometry_helper_sha256','outdoor/surface_geometry_delta.py'),
                           ('suffix_helper_sha256','outdoor/native_detail_suffix.py'),
                           ('loss_helper_sha256','outdoor/scoped_detail_loss.py')]:
            if manifest[field]!=hashlib.sha256((Path('/root/G4Splat-v114')/file).read_bytes()).hexdigest():
                raise RuntimeError('Implementation changed after smoke: '+file)
supervisor=Path(str(run)+'_supervisor')
if run.exists() or supervisor.exists():raise FileExistsError(run)
cmd=[sys.executable,'scripts/calibrate_native_detail_suffix.py','--run',str(root/'diagnostic_v498_representation_coarse_400'),
     '--proposals',str(root/'audit_v503_native_layer_association/conditional_native_layer_proposals.npz'),
     '--source_path','/mnt/pool/sqy/G4Splat_runs/cambridge_outdoor_mainline_v1/prepared/StMarysChurch/dataset_qc_tree_v5/train_all',
     '--geometry','native','--steps',str(steps),'--eval-count','48' if a.phase=='full' else '4',
     '--eval-every','76' if a.phase=='full' else '0','--output',str(run),
     '--refined-canopy',str(root/'diagnostic_v480_schedule800/candidates_0800.pth')]
if a.arm=='geometry':cmd+=['--surface-geometry']
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1' if a.arm=='geometry' else '2',
    LD_LIBRARY_PATH='/root/miniconda3/envs/g4splat/lib',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
with Path(str(run)+'_launch.log').open('x') as log:
    proc=subprocess.Popen([sys.executable,'scripts/supervise_detached_training.py','--run-dir',str(supervisor),
        '--max-safe-restarts','0','--']+cmd,cwd='/root/G4Splat-v114',env=env,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=proc.pid,run=str(run),gpu=env['CUDA_VISIBLE_DEVICES'])))
