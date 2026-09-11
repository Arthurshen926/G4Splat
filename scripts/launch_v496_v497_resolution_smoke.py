"""Independent candidate-resolution smoke; only physical GPU2."""
import argparse
import json
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--arm',choices=('coarse','dense_fine'),required=True)
a=p.parse_args()
root=Path('/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1')
command=json.loads((root/'diagnostic_v480_schedule800_supervisor/training_supervisor_contract.json').read_text())['base_command']
command[1]='scripts/calibrate_canopy_resolution_candidates.py'
for flag in ('--candidate-shape-cohort','--candidate-shape-lr'):
    index=command.index(flag);del command[index:index+2]
command.remove('--candidate-orientation-control')
version=496 if a.arm=='coarse' else 497
run=root/f'diagnostic_v{version}_representation_{a.arm}_smoke8'
supervisor=Path(str(run)+'_supervisor')
if run.exists() or supervisor.exists():raise RuntimeError('Refusing overwrite')
for flag,value in [('--output',str(run)),('--steps','8'),('--eval-every','4'),
                   ('--audit-actual-update-every','4'),('--gpu-memory-fraction','.75')]:
    command[command.index(flag)+1]=value
command+=['--representation-policy',a.arm,'--native-rgb-training']
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2',
    LD_LIBRARY_PATH='/root/miniconda3/envs/g4splat/lib',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
log=Path(str(run)+'_launch.log')
with log.open('x') as stream:
    result=subprocess.run([command[0],'scripts/supervise_detached_training.py',
        '--run-dir',str(supervisor),'--max-safe-restarts','0','--']+command,
        cwd='/root/G4Splat-v114',env=env,stdin=subprocess.DEVNULL,
        stdout=stream,stderr=subprocess.STDOUT)
print(log.read_text(),flush=True)
result.check_returncode()
