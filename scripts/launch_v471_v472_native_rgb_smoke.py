"""Eight-step native RGB paired smoke, full 800-step LR horizon preserved."""
import json
import os
from pathlib import Path
import subprocess

root=Path('/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1')
base=json.loads((root/'diagnostic_v467_native_depth_geometry800_supervisor/training_supervisor_contract.json').read_text())['base_command']
for flag in ('--moge-position-control','--moge-position-reachable-only'):base.remove(flag)
index=base.index('--moge-position-weight');del base[index:index+2]
for key,value in [('--eval-every','4'),('--audit-actual-update-every','4'),('--reference-cache-size','16')]:
    base[base.index(key)+1]=value
base+=['--stop-after','8']
for version,gpu,native in [(471,'1',False),(472,'2',True)]:
    run=root/f'diagnostic_v{version}_native_rgb_smoke8'
    supervisor=Path(str(run)+'_supervisor')
    if run.exists() or supervisor.exists():raise RuntimeError(f'Refusing to overwrite {run}')
    command=list(base);command[command.index('--output')+1]=str(run)
    if native:command+=['--native-rgb-training']
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
    env['LD_LIBRARY_PATH']='/root/miniconda3/envs/g4splat/lib'
    process=subprocess.Popen([command[0],'scripts/supervise_detached_training.py','--run-dir',str(supervisor),
        '--max-safe-restarts','0','--']+command,cwd='/root/G4Splat-v114',env=env,
        stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    print(dict(run=str(run),gpu=gpu,launcher_pid=process.pid),flush=True)
