"""Matched moved-ray refresh smoke; keep original 800-step learning horizon."""
import json
import os
from pathlib import Path
import subprocess

root=Path('/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1')
for v in [467,468]:
    run=root/f'diagnostic_v{v}_native_depth_geometry800'
    if not json.loads((run/'frozen_parameter_audit.json').read_text())['unchanged']:
        raise RuntimeError('Previous depth pair not safely complete')
base=json.loads((root/'diagnostic_v467_native_depth_geometry800_supervisor/training_supervisor_contract.json').read_text())['base_command']
for flag,value in [('--eval-every','4'),('--audit-actual-update-every','4'),('--reference-cache-size','16')]:
    base[base.index(flag)+1]=value
base+=['--stop-after','8']
for version,gpu,refresh in [(475,'1',0),(476,'2',4)]:
    run=root/f'diagnostic_v{version}_moge_refresh_smoke8';supervisor=Path(str(run)+'_supervisor')
    if run.exists() or supervisor.exists():raise RuntimeError(f'Refusing to overwrite {run}')
    command=list(base);command[command.index('--output')+1]=str(run)
    command+=['--moge-position-refresh-every',str(refresh)]
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
    env['LD_LIBRARY_PATH']='/root/miniconda3/envs/g4splat/lib'
    process=subprocess.Popen([command[0],'scripts/supervise_detached_training.py','--run-dir',str(supervisor),
        '--max-safe-restarts','0','--']+command,cwd='/root/G4Splat-v114',env=env,
        stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    print(dict(run=str(run),gpu=gpu,launcher_pid=process.pid),flush=True)
