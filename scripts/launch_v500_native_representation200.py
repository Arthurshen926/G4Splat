"""Matched native200 read-only audits on the permitted GPU1/2."""
import argparse
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--arm',choices=('coarse','dense_fine'),required=True)
p.add_argument('--step',type=int,choices=(200,400),default=200)
p.add_argument('--audit-version',type=int,default=500)
a=p.parse_args()
version,gpu=(498,'1') if a.arm=='coarse' else (499,'2')
root=Path('/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1')
run=root/f'diagnostic_v{version}_representation_{a.arm}_400'
output=root/f'audit_v{a.audit_version}_native_v{version}_{a.step}'
if output.exists():raise RuntimeError('Refusing overwrite')
if not (run/f'candidates_{a.step:04d}.pth').exists():raise RuntimeError('Published checkpoint required')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,
    LD_LIBRARY_PATH='/root/miniconda3/envs/g4splat/lib',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
with Path(str(output)+'.log').open('x') as log:
    subprocess.run(['/root/miniconda3/envs/g4splat/bin/python',
        'scripts/audit_canopy_candidate_contribution.py', '-s',
        '/mnt/pool/sqy/G4Splat_runs/cambridge_outdoor_mainline_v1/prepared/StMarysChurch/dataset_qc_tree_v5/train_all',
        '--run',str(run),'--step',str(a.step),'--output',str(output),
        '--resolution','1920','--save-rgb-pairs'],env=env,cwd='/root/G4Splat-v114',
        stdout=log,stderr=subprocess.STDOUT,check=True)
print(f'Completed {output}',flush=True)
