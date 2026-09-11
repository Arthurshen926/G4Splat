"""GPU2 read-only native endpoints; horizon AND order differ, not causal A/B."""
import os
import subprocess
from pathlib import Path

root = Path('/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1')
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='2', LD_LIBRARY_PATH='/root/miniconda3/envs/g4splat/lib',
           OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
for version, steps, run_name in [(480,800,'diagnostic_v480_schedule800'),
                                (488,3200,'diagnostic_v488_schedule3200')]:
    output = root / f'audit_v495_native_v{version}_{steps}'
    if output.exists():
        raise RuntimeError(f'Refusing overwrite: {output}')
    with Path(str(output)+'.log').open('x') as log:
        subprocess.run(['/root/miniconda3/envs/g4splat/bin/python',
            'scripts/audit_canopy_candidate_contribution.py', '-s',
            '/mnt/pool/sqy/G4Splat_runs/cambridge_outdoor_mainline_v1/prepared/StMarysChurch/dataset_qc_tree_v5/train_all',
            '--run',str(root/run_name),'--step',str(steps),'--output',str(output),
            '--resolution','1920','--save-rgb-pairs'], env=env, cwd='/root/G4Splat-v114',
            stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f'Completed {output}', flush=True)
