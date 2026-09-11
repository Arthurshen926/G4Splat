"""Boundary observed-risk prefix against immutable v480, no LR-horizon change."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path('/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1')
control=root/'diagnostic_v480_schedule800'
manifest=json.loads((control/'manifest.json').read_text())
for file,key in [('scripts/calibrate_canopy_candidates.py','script_sha256'),
                 ('scripts/canopy_declared_training_objective.py','declared_objective_helper_sha256')]:
    if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=manifest[key]:
        raise RuntimeError('Changed control implementation: '+file)
assert json.loads((control/'frozen_parameter_audit.json').read_text())['unchanged']
command=json.loads(Path(str(control)+'_supervisor/training_supervisor_contract.json').read_text())['base_command']
assert command[command.index('--noncanopy-rgb-target')+1]=='observed_risk'
command[command.index('--rigid-boundary-preservation-weight')+1]='1'
run=root/'diagnostic_v489_boundary_risk400';supervisor=Path(str(run)+'_supervisor')
if run.exists() or supervisor.exists():raise RuntimeError('Refusing overwrite')
command[command.index('--output')+1]=str(run)
command+=['--stop-after','400']
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='1';env['LD_LIBRARY_PATH']='/root/miniconda3/envs/g4splat/lib'
log=Path(str(run)+'_launch.log')
with log.open('x') as out:
    result=subprocess.run([command[0],'scripts/supervise_detached_training.py','--run-dir',str(supervisor),
        '--max-safe-restarts','0','--']+command,cwd='/root/G4Splat-v114',env=env,
        stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
print(log.read_text(),flush=True);result.check_returncode()
