"""Clean full-horizon native RGB pair following completed 8-step smoke."""
import json
import os
from pathlib import Path
import subprocess

root=Path('/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1')
for version,source,gpu in [(473,471,'1'),(474,472,'2')]:
    previous=root/f'diagnostic_v{source}_native_rgb_smoke8'
    if not json.loads((previous/'frozen_parameter_audit.json').read_text())['unchanged']:
        raise RuntimeError('Smoke failed frozen audit')
    command=json.loads(Path(str(previous)+'_supervisor/training_supervisor_contract.json').read_text())['base_command']
    index=command.index('--stop-after');del command[index:index+2]
    for flag in ('--eval-every','--audit-actual-update-every'):command[command.index(flag)+1]='400'
    run=root/f'diagnostic_v{version}_native_rgb800';supervisor=Path(str(run)+'_supervisor')
    if run.exists() or supervisor.exists():raise RuntimeError(f'Refusing to overwrite {run}')
    command[command.index('--output')+1]=str(run)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
    env['LD_LIBRARY_PATH']='/root/miniconda3/envs/g4splat/lib'
    process=subprocess.Popen([command[0],'scripts/supervise_detached_training.py','--run-dir',str(supervisor),
        '--max-safe-restarts','0','--']+command,cwd='/root/G4Splat-v114',env=env,
        stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    print(dict(run=str(run),gpu=gpu,launcher_pid=process.pid),flush=True)
