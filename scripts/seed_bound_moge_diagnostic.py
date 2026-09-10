"""Use the same immutable MoGe scale/profile authority as a source checkpoint."""
import io
import json
import math
from pathlib import Path
import torch
from outdoor.moge3_evidence import sha256_file
from scripts.extract_canopy_render_state import DeferredUnpickler


def apply_seed_calibration(geometry, audit, expected_metric_scale):
    initialization=audit['moge3_exact_k_front_hit']
    scale=float(initialization['metric_to_cambridge_scale'])
    if not math.isfinite(scale) or scale<=0 or not math.isclose(scale,float(expected_metric_scale),rel_tol=1e-8):
        raise ValueError('Diagnostic metric scale differs from checkpoint seed authority')
    profiles=audit['fresh_canonical_leaves']['depth_profiles_by_image']
    if not profiles:raise ValueError('Missing scene-canonical canopy scale profiles')
    geometry.configure_moge3_metric_scale(scale)
    geometry.configure_moge3_canopy_depth_scales(profiles)
    sigma=float(initialization.get('global_log_scale_sigma',0.))
    if not math.isfinite(sigma) or sigma<0:raise ValueError('Invalid seed scale uncertainty')
    return dict(metric_scale=scale,global_log_scale_sigma=sigma,
                canopy_profile_count=len(profiles),scope='checkpoint_seed_bound_global_and_canopy_scales')


def configure_seed_bound_moge(geometry, initialization, training_contract, expected_metric_scale):
    manifest_path=Path(initialization)/'initialization_manifest.json'
    if sha256_file(manifest_path)!=training_contract['initialization_manifest_sha256']:
        raise ValueError('Diagnostic initialization differs from checkpoint')
    manifest=json.loads(manifest_path.read_text());seed_path=Path(manifest['foliage_seed'])
    if sha256_file(seed_path)!=training_contract['foliage_seed_sha256']:
        raise ValueError('Diagnostic seed differs from checkpoint')
    reader=torch._C.PyTorchFileReader(str(seed_path))
    seed=DeferredUnpickler(io.BytesIO(reader.get_record('data.pkl'))).load()
    result=apply_seed_calibration(geometry,seed['audit'],expected_metric_scale)
    result.update(initialization_manifest_sha256=training_contract['initialization_manifest_sha256'],
                  foliage_seed_sha256=training_contract['foliage_seed_sha256'])
    return result
