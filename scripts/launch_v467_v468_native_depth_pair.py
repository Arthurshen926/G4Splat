"""Matched geometry-resolution experiment; GPU0 is never used."""
import json
import os
from pathlib import Path
import subprocess

root=Path('/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1')
contract=json.loads((root/'diagnostic_v466_reachable_moge_position800_supervisor/training_supervisor_contract.json').read_text())
for version,gpu,native in [(467,'1',False),(468,'2',True)]:
    run=root/f'diagnostic_v{version}_native_depth_geometry800'
    supervisor=Path(str(run)+'_supervisor')
    if run.exists() or supervisor.exists():raise RuntimeError(f'Refusing to overwrite {run}')
    command=list(contract['base_command'])
    command[command.index('--output')+1]=str(run)
    if native:command+=['--moge-position-native-directory',str(root/'StMarysChurch_moge3_vitl_exact_k/views')]
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
    env['LD_LIBRARY_PATH']='/root/miniconda3/envs/g4splat/lib'
    process=subprocess.Popen([command[0],'scripts/supervise_detached_training.py','--run-dir',str(supervisor),
                              '--max-safe-restarts','0','--']+command,
                             cwd='/root/G4Splat-v114',env=env,stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    print(dict(run=str(run),gpu=gpu,launcher_pid=process.pid),flush=True)
