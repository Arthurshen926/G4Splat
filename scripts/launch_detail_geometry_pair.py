"""Geometry-aware versus optical-only detail reconstruction, GPU1/2."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['geometry','fixed'],required=True);a=p.parse_args()
root=Path('/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1');repo=Path('/root/G4Splat-v114')
for name in ['diagnostic_v528_detail_geometry_smoke8','diagnostic_v529_detail_fixed_smoke8']:
    smoke=root/name
    if not (smoke/'eval_000008.json').exists():raise ValueError('CUDA smoke incomplete')
    m=json.loads((smoke/'manifest.json').read_text())
    for key,path in [('script_sha256','scripts/calibrate_native_detail_suffix.py'),('suffix_helper_sha256','outdoor/native_detail_suffix.py')]:
        if m[key]!=hashlib.sha256((repo/path).read_bytes()).hexdigest():raise ValueError('Changed implementation')
    if not json.loads((smoke/'frozen_parameter_audit.json').read_text())['unchanged']:raise ValueError('Source changed')
version=530 if a.arm=='geometry' else 531
run=root/f'diagnostic_v{version}_detail_{a.arm}304';supervisor=Path(str(run)+'_supervisor')
if run.exists() or supervisor.exists():raise FileExistsError(run)
cmd=[sys.executable,'scripts/calibrate_native_detail_suffix.py','--run',str(root/'diagnostic_v498_representation_coarse_400'),
     '--proposals',str(root/'audit_v526_independent_support/rgb_triangulated_proposals.npz'),
     '--source_path','/mnt/pool/sqy/G4Splat_runs/cambridge_outdoor_mainline_v1/prepared/StMarysChurch/dataset_qc_tree_v5/train_all',
     '--geometry','native','--steps','304','--eval-count','48','--eval-every','76','--output',str(run)]
if a.arm=='geometry':cmd+=['--refine-proposals']
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1' if a.arm=='geometry' else '2',
 LD_LIBRARY_PATH='/root/miniconda3/envs/g4splat/lib',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
with Path(str(run)+'_launch.log').open('x') as log:
    proc=subprocess.Popen([sys.executable,'scripts/supervise_detached_training.py','--run-dir',str(supervisor),
      '--max-safe-restarts','0','--']+cmd,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(run=str(run),pid=proc.pid,gpu=env['CUDA_VISIBLE_DEVICES'])))
