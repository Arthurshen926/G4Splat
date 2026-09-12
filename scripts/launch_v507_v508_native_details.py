"""One-epoch paired native-layer depth trial after completed CUDA smokes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--arm',choices=('native','area'),required=True)
a=p.parse_args()
root=Path('/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1')
for name in ('diagnostic_v504_native_detail_smoke8','diagnostic_v505_area_detail_smoke8'):
    run=root/name
    if not (run/'suffix.pth').is_file() or not json.loads((run/'frozen_parameter_audit.json').read_text())['unchanged']:
        raise RuntimeError('Both complete CUDA smokes required')
version,gpu=(507,'1') if a.arm=='native' else (508,'2')
run=root/f'diagnostic_v{version}_{a.arm}_details304'
supervisor=Path(str(run)+'_supervisor')
if run.exists() or supervisor.exists():raise FileExistsError(run)
command=[sys.executable,'scripts/calibrate_native_detail_suffix.py',
    '--run',str(root/'diagnostic_v498_representation_coarse_400'),
    '--proposals',str(root/'audit_v503_native_layer_association/conditional_native_layer_proposals.npz'),
    '--source_path','/mnt/pool/sqy/G4Splat_runs/cambridge_outdoor_mainline_v1/prepared/StMarysChurch/dataset_qc_tree_v5/train_all',
    '--geometry',a.arm,'--steps','304','--eval-count','48','--output',str(run)]
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,
    LD_LIBRARY_PATH='/root/miniconda3/envs/g4splat/lib',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
with Path(str(run)+'_launch.log').open('x') as log:
    process=subprocess.Popen([sys.executable,'scripts/supervise_detached_training.py',
        '--run-dir',str(supervisor),'--max-safe-restarts','0','--']+command,
        cwd='/root/G4Splat-v114',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
        start_new_session=True)
print(json.dumps(dict(supervisor_pid=process.pid,gpu=gpu,run=str(run))),flush=True)
