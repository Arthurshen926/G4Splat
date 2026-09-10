"""Launch one explicit allocator comparison after its GPU predecessor exits.

Waiting is not training. No checkpoint or existing run is edited, and failed
predecessors never silently trigger the next experiment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def predecessor_completed(state):
    if state.get('state')=='completed':
        if state.get('returncode')!=0:raise RuntimeError('Predecessor did not exit successfully')
        return True
    if state.get('returncode') not in (None,0):
        raise RuntimeError('Predecessor failed; queued experiment not launched')
    return False


def experiment_command(base,output,budget):
    if '--resume' in base or '-m' not in base or '--allow-trainer-repair-resume' not in base:
        raise ValueError('Expected a supervisor-owned repair base command')
    if budget<=0:raise ValueError('Positive allocator cap required')
    command=list(base);command[command.index('-m')+1]=str(output)
    flag='--maximum-static-ray-births-per-event'
    if flag in command:command[command.index(flag)+1]=str(budget)
    else:command.extend([flag,str(budget)])
    command.append('--allow-static-ray-birth-budget-ablation-resume')
    return command


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('predecessor','source-contract','checkpoint','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gpu',type=int,choices=(1,2),required=True)
    p.add_argument('--budget',type=int,default=8192)
    p.add_argument('--maximum-wait-seconds',type=int,default=7200)
    a=p.parse_args()
    base=json.loads(a.source_contract.read_text())['base_command']
    command=experiment_command(base,a.output,a.budget)
    root=Path(__file__).resolve().parents[1]
    sources=[root/'scripts/train_unified_outdoor_teacher.py',
             root/'2d-gaussian-splatting/scene/gaussian_model.py',*sorted((root/'outdoor').glob('*.py'))]
    identity=lambda:{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    source_hashes=identity()
    if a.output.exists():raise FileExistsError(a.output)
    deadline=time.monotonic()+a.maximum_wait_seconds
    print(json.dumps(dict(state='queued_not_training',predecessor=str(a.predecessor),gpu=a.gpu,output=str(a.output))),flush=True)
    while time.monotonic()<deadline:
        state=json.loads((a.predecessor/'training_supervisor_heartbeat.json').read_text())
        if predecessor_completed(state):
            free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free>=12500:break
        time.sleep(15)
    else:raise TimeoutError('Queued experiment never obtained a successful predecessor and memory headroom')
    if not a.checkpoint.is_file():raise FileNotFoundError(a.checkpoint)
    if identity()!=source_hashes:
        raise RuntimeError('Trainer sources changed while queued; comparison must be reviewed before launch')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu))
    launch=[sys.executable,str(Path(__file__).with_name('supervise_detached_training.py')),
        '--run-dir',str(a.output),'--max-safe-restarts','0','--initial-resume-checkpoint',str(a.checkpoint),'--',*command]
    subprocess.run(launch,env=env,check=True)


if __name__=='__main__':main()
