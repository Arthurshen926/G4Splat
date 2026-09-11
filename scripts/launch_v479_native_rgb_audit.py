"""Sequential read-only native-resolution A/B, exclusively on physical GPU1."""
import os
import argparse
from pathlib import Path
import subprocess

root=Path('/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1')
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--step',type=int,choices=(400,800),default=400)
p.add_argument('--audit-version',type=int,default=479)
a=p.parse_args()
env=os.environ.copy()
env['CUDA_VISIBLE_DEVICES']='1'
env['LD_LIBRARY_PATH']='/root/miniconda3/envs/g4splat/lib'
for version in (473,474):
    run=root/f'diagnostic_v{version}_native_rgb800'
    output=root/f'audit_v{a.audit_version}_native_v{version}_{a.step}'
    if output.exists():raise RuntimeError(f'Refusing overwrite: {output}')
    with Path(str(output)+'.log').open('x') as log:
        subprocess.run(['/root/miniconda3/envs/g4splat/bin/python',
            'scripts/audit_canopy_candidate_contribution.py',
            '-s','/mnt/pool/sqy/G4Splat_runs/cambridge_outdoor_mainline_v1/prepared/StMarysChurch/dataset_qc_tree_v5/train_all',
            '--run',str(run),'--step',str(a.step),'--output',str(output),
            '--resolution','1920','--save-rgb-pairs'],
            cwd='/root/G4Splat-v114',env=env,stdout=log,stderr=subprocess.STDOUT,
            check=True)
    print(f'Completed native read-only audit: {output}',flush=True)
