"""Exact 4->8 restore check across the first moved-ray refresh event."""
import json
import os
from pathlib import Path
import subprocess

root=Path('/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1')
source=root/'diagnostic_v476_moge_refresh_smoke8'
assert json.loads((source/'frozen_parameter_audit.json').read_text())['unchanged']
command=json.loads(Path(str(source)+'_supervisor/training_supervisor_contract.json').read_text())['base_command']
run=root/'diagnostic_v478b_moge_refresh_resume4to8';supervisor=Path(str(run)+'_supervisor')
if run.exists() or supervisor.exists():raise RuntimeError('Refusing to overwrite resume audit')
command[command.index('--output')+1]=str(run)
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='2';env['LD_LIBRARY_PATH']='/root/miniconda3/envs/g4splat/lib'
launch_log=Path(str(run)+'_launch.log')
with launch_log.open('x') as log:
    # The supervisor owns external initial resumes and detaches itself. Await
    # its startup result rather than mistaking a short-lived Popen PID for training.
    process=subprocess.run([command[0],'scripts/supervise_detached_training.py','--run-dir',str(supervisor),
        '--initial-resume-checkpoint',str(source/'candidates_0004.pth'),
        '--max-safe-restarts','0','--']+command,cwd='/root/G4Splat-v114',env=env,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
print(launch_log.read_text(),flush=True)
process.check_returncode()
