"""Clean long-horizon convergence pair; no relaxed building protection."""
import argparse
import json
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--arm',choices=('control','shuffle'),required=True)
a=p.parse_args()
root=Path('/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1')
previous=root/('diagnostic_v480_schedule800' if a.arm=='control' else 'diagnostic_v481_schedule800')
if not json.loads((previous/'frozen_parameter_audit.json').read_text())['unchanged']:
    raise RuntimeError('Short arm has not passed its frozen audit')
command=json.loads(Path(str(previous)+'_supervisor/training_supervisor_contract.json').read_text())['base_command']
version,gpu=(487,'1') if a.arm=='control' else (488,'2')
run=root/f'diagnostic_v{version}_schedule3200';supervisor=Path(str(run)+'_supervisor')
if run.exists() or supervisor.exists():raise RuntimeError(f'Refusing overwrite: {run}')
if '--resume' in command:raise RuntimeError('This is a clean new horizon, not an exact short-run resume')
command[command.index('--output')+1]=str(run)
command[command.index('--steps')+1]='3200'
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
env['LD_LIBRARY_PATH']='/root/miniconda3/envs/g4splat/lib'
log_path=Path(str(run)+'_launch.log')
with log_path.open('x') as log:
    result=subprocess.run([command[0],'scripts/supervise_detached_training.py',
        '--run-dir',str(supervisor),'--max-safe-restarts','0','--']+command,
        cwd='/root/G4Splat-v114',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
print(log_path.read_text(),flush=True);result.check_returncode()
