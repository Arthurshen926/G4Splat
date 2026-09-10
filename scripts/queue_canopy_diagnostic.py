"""Run one detached diagnostic after a successful predecessor and free memory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.queue_ray_birth_budget_experiment import predecessor_completed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predecessor',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--gpu',type=int,choices=(1,2),required=True)
    p.add_argument('--minimum-free-mib',type=int,default=6500)
    p.add_argument('--maximum-wait-seconds',type=int,default=7200)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();command=a.command[1:] if a.command[:1]==['--'] else a.command
    if len(command)<2 or a.minimum_free_mib<1 or a.maximum_wait_seconds<1:
        raise ValueError('A diagnostic command and bounded resource limits are required')
    script=Path(command[1]).resolve()
    if not script.is_file() or ROOT not in script.parents or a.run_dir.exists():
        raise ValueError('Existing repository diagnostic and fresh output required')
    sources=[script,*sorted((ROOT/'outdoor').glob('*.py'))]
    identity=lambda:{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    before=identity();deadline=time.monotonic()+a.maximum_wait_seconds
    print(json.dumps(dict(state='queued_not_training',gpu=a.gpu,predecessor=str(a.predecessor))),flush=True)
    while time.monotonic()<deadline:
        state=json.loads((a.predecessor/'training_supervisor_heartbeat.json').read_text())
        if predecessor_completed(state):
            free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free>=a.minimum_free_mib:break
        time.sleep(15)
    else:raise TimeoutError('No successful predecessor and memory headroom')
    if before!=identity():raise RuntimeError('Diagnostic sources changed while queued; review before launch')
    launch=[sys.executable,str(ROOT/'scripts/supervise_detached_training.py'),
            '--run-dir',str(a.run_dir),'--max-safe-restarts','0','--',*command]
    subprocess.run(launch,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu)),check=True)


if __name__=='__main__':main()
