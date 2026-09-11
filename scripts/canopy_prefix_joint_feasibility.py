"""Necessary multi-depth RGB-prefix constraints, replacing a single wall median.

For any depth z, unoccluded rigid radiance emitted BEFORE z is B_prefix(z).
Only volumes before z can attenuate it. Crediting all such volumes against the
entire prefix gives a LOWER radiance bound, so K_z tau >= log(B_prefix/I) is
necessary in the uncapped exponential support model, not sufficient. Surface
light behind z is deliberately not used in this inequality.
"""
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
import torch

from outdoor.hybrid_gaussian_renderer import persistent_static_evidence_mask, static_detail_forward_visibility_gate
from scripts.canopy_native_ray_support import sparse_ray_support
from scripts.canopy_joint_optical_feasibility import solve_optical_intervals
from scripts.canopy_radiance_prefix_bounds import optical_prefix_lower_bound


@torch.no_grad()
def prefix_joint_feasibility(base, adapter, candidate, views, manifest, constraints_dir, prefix_path, output, *, snapshot_sha256):
    constraints_dir=Path(constraints_dir)
    old=json.loads((constraints_dir/'joint_feasibility.json').read_text())
    prefix=json.loads(Path(prefix_path).read_text())
    exact_batch=prefix.get('scope')=='native_pure_rigid_radiance_prefix__training_only__not_leaf_depth_truth'
    if exact_batch:
        from outdoor.moge3_evidence import sha256_file
        from scripts.canopy_batch_prefix_constraints import load_batch_prefix_records
        if prefix['constraints_sha256']!=sha256_file(constraints_dir/'joint_feasibility.json'):
            raise ValueError('Batch prefix constraints hash mismatch')
        prefix['records']=load_batch_prefix_records(prefix_path,prefix,old,manifest)
        source_matches=True
    else:
        replay=json.loads(Path(prefix['input_replay']).read_text())
        info=json.loads((Path(replay['solution'])/'report.json').read_text())
        source_matches=Path(info['source']).resolve()==constraints_dir.resolve()
    if (old.get('snapshot_sha256')!=snapshot_sha256
            or old['source_checkpoint_sha256']!=manifest['source_checkpoint_sha256']
            or not source_matches):
        raise ValueError('Exact prefix/source geometry identity required')
    if set(old['training_views'])-set(manifest['training_views']):
        raise ValueError('Training-only prefix constraints required')
    old_data=np.load(constraints_dir/'joint_solution.npz')
    old_K=sparse.load_npz(constraints_dir/'joint_kernel.npz')
    negative=np.array([r['kind']=='visible' for r in old['rays']])
    coo=old_K[negative].tocoo()
    matrices=[sparse.csr_matrix((coo.data,(coo.row,old_data['primitive_ids'][coo.col])),
                               shape=(int(negative.sum()),len(adapter)))]
    metadata=[r for r in old['rays'] if r['kind']=='visible']
    lower=list(old_data['lower'][negative]);upper=list(old_data['upper'][negative])
    # Negative rows already have frozen depth removed; keep that provenance
    # separately without double-subtracting it from their shifted interval.
    fixed_chunks=[old_data['fixed_optical_depth'][negative]]
    xyz,_,alpha=adapter.conditioned_state(None,include_dynamic=False)
    eligible=torch.cat((persistent_static_evidence_mask(base)&base.static_leaf_mask,
                        torch.ones(len(candidate.xyz),dtype=torch.bool,device='cuda'))).cpu().numpy()
    tau=-np.log1p(-alpha.cpu().numpy().clip(0,1-1e-6))
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'experimental/canopy_optical_depth_rasterizer'))
    import canopy_optical_depth_rasterization as native
    omitted=[];projected_view=None;projected=None
    for record in prefix['records']:
        ray=record['ray'];index=ray['view'];v=views[index]
        if index not in manifest['training_views'] or record['maximum_nonmonotonic_rgb']>1e-4:
            raise ValueError('Reliable native training prefix required')
        samples=[]
        for s in record['samples']:
            # A finite black screen may leak a tiny suffix. Subtract a
            # conservative radiance allowance rather than treating it as prefix.
            L=optical_prefix_lower_bound(s['surface_rgb'],record['target'],ray['wall_rgb'],
                                        leakage_fraction=0. if exact_batch else 1e-4)
            if L>.05:samples.append((s,L))
        if not samples:continue
        if projected_view!=index:
            projected=native._C.diagnostic_volume_support(xyz.contiguous(),adapter.scales.contiguous(),
                adapter.normalized_quaternions.contiguous(),v.world_view_transform.contiguous(),
                v.full_proj_transform.contiguous(),v.image_width,v.image_height,math.tan(v.FoVx/2),math.tan(v.FoVy/2))
            projected_view=index
        uv=torch.tensor([[ray['x'],ray['y']]]*len(samples),device='cuda',dtype=torch.float32)
        depths=torch.tensor([s['depth'] for s,_ in samples],device='cuda')
        K,drop=sparse_ray_support(projected,uv,depths,v.image_width,v.image_height)
        gate=static_detail_forward_visibility_gate(base,int(v.colmap_id),include_pending_exact=False)
        gate=torch.cat((gate,torch.ones(len(candidate.xyz),device='cuda'))).cpu().numpy()
        if not np.isin(gate,[0.,1.]).all():raise ValueError('Binary gate required')
        K=K.multiply(gate).tocsr();fixed=K[:,~eligible]@tau[~eligible]
        matrices.append(K.multiply(eligible).tocsr());fixed_chunks.append(fixed);omitted.extend(drop.tolist())
        for j,(sample,L) in enumerate(samples):
            lower.append(max(L-fixed[j],0.));upper.append(np.inf)
            metadata.append(dict(view=index,x=ray['x'],y=ray['y'],kind='occlusion',wall_depth=sample['depth'],
                wall_rgb=sample['surface_rgb'],target=record['target'],
                evidence='native_rigid_RGB_prefix_before_depth__necessary_not_sufficient'))
    K=sparse.vstack(matrices,format='csr');K.eliminate_zeros()
    ids=np.unique(K.indices).astype(np.int64);K=K[:,ids].astype(np.float64)
    if not eligible[ids].all():raise ValueError('Frozen variables cannot enter the solve')
    caps=np.where(ids<len(base),-math.log(.005),-math.log(1e-6))
    lo=np.asarray(lower);hi=np.asarray(upper);result=solve_optical_intervals(K,lo,hi,caps)
    sparse.save_npz(output/'joint_kernel.npz',K)
    np.savez(output/'joint_solution.npz',primitive_ids=ids,tau=result['tau'],lower=lo,upper=hi,
             current_tau=tau[ids],fixed_optical_depth=np.concatenate(fixed_chunks))
    report={k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
    report.update(rays=metadata,variables=len(ids),nonzeros=K.nnz,training_views=old['training_views'],
        snapshot_sha256=snapshot_sha256,source_checkpoint_sha256=manifest['source_checkpoint_sha256'],
        lower_slack=result['lower_slack'].tolist(),upper_slack=result['upper_slack'].tolist(),
        max_omitted_kernel_sum=max(omitted,default=0.),input_prefix=str(prefix_path),
        unique_positive_rays=len(prefix['records']),excluded_views_untouched=True,
        limitations=['Up to six exact native prefix constraints per prior positive training ray; not dense image coverage' if exact_batch else 'Only eight selected training rays have multi-depth positive constraints; all prior visible-wall rays retained',
                     'RGB-prefix constraints are necessary, not sufficient; native all-view replay still required',
                     'Exact contributor prefixes have no screen leakage; visible-wall labels remain conditional' if exact_batch else 'Finite-screen suffix leakage allowance of1e-4 times original rigid RGB subtracted in lower bounds; wall labels remain conditional'])
    (output/'joint_feasibility.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('rays','lower_slack','upper_slack','training_views')}),flush=True)
