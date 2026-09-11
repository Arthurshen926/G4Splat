"""Single-variable epoch-order pair, with acknowledged detached startup."""
import json
import os
from pathlib import Path
import subprocess

root=Path('/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1')
source=root/'diagnostic_v473_native_rgb800'
assert json.loads((source/'frozen_parameter_audit.json').read_text())['unchanged']
base=json.loads(Path(str(source)+'_supervisor/training_supervisor_contract.json').read_text())['base_command']
assert '--native-rgb-training' not in base
assert '--reshuffle-training-epochs' not in base
for version,gpu,reshuffle in [(480,'1',False),(481,'2',True)]:
    run=root/f'diagnostic_v{version}_schedule800'
    supervisor=Path(str(run)+'_supervisor')
    if run.exists() or supervisor.exists():raise RuntimeError(f'Refusing overwrite: {run}')
    command=list(base)
    command[command.index('--output')+1]=str(run)
    if reshuffle:command+=['--reshuffle-training-epochs']
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
    env['LD_LIBRARY_PATH']='/root/miniconda3/envs/g4splat/lib'
    log_path=Path(str(run)+'_launch.log')
    with log_path.open('x') as log:
        result=subprocess.run([command[0],'scripts/supervise_detached_training.py',
            '--run-dir',str(supervisor),'--max-safe-restarts','0','--']+command,
            cwd='/root/G4Splat-v114',env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT)
    print(log_path.read_text(),flush=True)
    result.check_returncode()
