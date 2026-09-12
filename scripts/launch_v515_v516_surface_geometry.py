"""Paired existing-surface geometry optimization after CUDA gradient smokes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--arm',choices=('geometry','fixed'),required=True)
a=p.parse_args();repo=Path('/root/G4Splat-v114')
root=Path('/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1')
for name in ('diagnostic_v513_scoped_geometry_smoke8','diagnostic_v514_scoped_fixed_smoke8'):
    smoke=root/name
    if not (smoke/'suffix.pth').exists() or not json.loads((smoke/'frozen_parameter_audit.json').read_text())['unchanged']:
        raise RuntimeError('Both successful CUDA smokes required')
    m=json.loads((smoke/'manifest.json').read_text())
    for field,path in [('geometry_helper_sha256','outdoor/surface_geometry_delta.py'),
                       ('suffix_helper_sha256','outdoor/native_detail_suffix.py'),
                       ('loss_helper_sha256','outdoor/scoped_detail_loss.py')]:
        if m[field]!=hashlib.sha256((repo/path).read_bytes()).hexdigest():raise RuntimeError('Math implementation changed since smoke')
version,gpu=(515,'1') if a.arm=='geometry' else (516,'2')
run=root/f'diagnostic_v{version}_{a.arm}_surface304';supervisor=Path(str(run)+'_supervisor')
if run.exists() or supervisor.exists():raise FileExistsError(run)
command=[sys.executable,'scripts/calibrate_native_detail_suffix.py','--run',str(root/'diagnostic_v498_representation_coarse_400'),
 '--proposals',str(root/'audit_v503_native_layer_association/conditional_native_layer_proposals.npz'),
 '--source_path','/mnt/pool/sqy/G4Splat_runs/cambridge_outdoor_mainline_v1/prepared/StMarysChurch/dataset_qc_tree_v5/train_all',
 '--geometry','native','--steps','304','--eval-count','48','--eval-every','76','--output',str(run)]
if a.arm=='geometry':command+=['--surface-geometry']
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,LD_LIBRARY_PATH='/root/miniconda3/envs/g4splat/lib',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
with Path(str(run)+'_launch.log').open('x') as log:
    process=subprocess.Popen([sys.executable,'scripts/supervise_detached_training.py','--run-dir',str(supervisor),
        '--max-safe-restarts','0','--']+command,cwd=str(repo),env=env,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(launcher_pid=process.pid,run=str(run),gpu=gpu)),flush=True)
