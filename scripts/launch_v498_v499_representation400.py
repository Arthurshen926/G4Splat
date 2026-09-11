"""Clean paired representation trial; shared800 LR horizon, bounded400 prefix."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--arm',choices=('coarse','dense_fine'),required=True)
a=p.parse_args()
root=Path('/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1')
smokes={'coarse':root/'diagnostic_v496_representation_coarse_smoke8',
        'dense_fine':root/'diagnostic_v497_representation_dense_fine_smoke8'}
for smoke in smokes.values():
    heartbeat=json.loads(Path(str(smoke)+'_supervisor/training_supervisor_heartbeat.json').read_text())
    if heartbeat['state']!='completed' or heartbeat['returncode']!=0:
        raise RuntimeError('Both smokes must finish normally before either formal arm starts')
    if not json.loads((smoke/'frozen_parameter_audit.json').read_text())['unchanged']:
        raise RuntimeError('Both smokes must preserve frozen parameters')
from scripts.compare_canopy_candidate_controls import compare
compare(smokes['coarse'],smokes['dense_fine'],8,'representation_policy')
manifest=json.loads((smokes[a.arm]/'manifest.json').read_text())
for filename,key in [('calibrate_canopy_resolution_candidates.py','script_sha256'),
                     ('canopy_representation_resolution.py','representation_helper_sha256')]:
    if hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()!=manifest[key]:
        raise RuntimeError('Experimental implementation changed since smoke')
command=json.loads(Path(str(smokes[a.arm])+'_supervisor/training_supervisor_contract.json').read_text())['base_command']
if '--resume' in command or '--stop-after' in command:
    raise RuntimeError('Require a clean smoke command, not an altered continuation')
version,gpu=(498,'1') if a.arm=='coarse' else (499,'2')
run=root/f'diagnostic_v{version}_representation_{a.arm}_400'
supervisor=Path(str(run)+'_supervisor')
if run.exists() or supervisor.exists():raise RuntimeError('Refusing overwrite')
for flag,value in [('--output',str(run)),('--steps','800'),('--eval-every','200'),
                   ('--audit-actual-update-every','200')]:
    command[command.index(flag)+1]=value
command+=['--stop-after','400']
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,
    LD_LIBRARY_PATH='/root/miniconda3/envs/g4splat/lib',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
log=Path(str(run)+'_launch.log')
with log.open('x') as stream:
    result=subprocess.run([command[0],'scripts/supervise_detached_training.py',
        '--run-dir',str(supervisor),'--max-safe-restarts','0','--']+command,
        cwd='/root/G4Splat-v114',env=env,stdin=subprocess.DEVNULL,
        stdout=stream,stderr=subprocess.STDOUT)
print(log.read_text(),flush=True)
result.check_returncode()
