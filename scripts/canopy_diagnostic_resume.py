"""Strict same-code diagnostic continuation, never production checkpoint promotion."""
import json
import random
import numpy as np
import torch

CAUSAL_HISTORY_KEYS = ('saw_gradient', 'saw_update', 'saw_depth_gradient',
    'saw_scale_gradient', 'saw_position_gradient', 'saw_source_opacity_update',
    'saw_source_position_gradient')


def validate_causal_history(history):
    if (not isinstance(history, dict) or set(history) != set(CAUSAL_HISTORY_KEYS)
            or any(type(v) is not bool for v in history.values())):
        raise ValueError('Complete boolean causal history required for continuation')
    return tuple(history[k] for k in CAUSAL_HISTORY_KEYS)


def normalized_manifest(manifest):
    value = json.loads(json.dumps(manifest, default=str))
    for key in ('output', 'resume', 'stop_after'):
        value['args'].pop(key, None)
    return value


def validate_resume(payload, manifest, layout, order):
    if payload.get('diagnostic_only') is not True or payload.get('diagnostic_optimizer_state_version') != 2:
        raise ValueError('Complete version-2 diagnostic checkpoint required; old snapshots are not exact resumes')
    step = payload['step']; args = manifest['args']
    if (not isinstance(step, int) or not 0 < step < args['steps']
            or payload['completed_training_images'] != step
            or payload['optimizer_schedule_horizon'] != args['steps']
            or payload['gradient_accumulation_pending']
            or step % args['gradient_accumulation']
            or payload['completed_optimizer_updates'] != step // args['gradient_accumulation']):
        raise ValueError('Resume must preserve horizon and complete optimizer boundary')
    if normalized_manifest(payload['manifest']) != normalized_manifest(manifest):
        raise ValueError('Resume changed configuration, source, evidence or implementation')
    if payload['optimizer_parameter_layout'] != layout or payload['training_order'] != order:
        raise ValueError('Resume changed parameter identities or camera order')
    if len(payload['cuda_rng']) != torch.cuda.device_count():
        raise ValueError('Resume requires the same logical CUDA device count')
    return step


def validate_module_state(module, state):
    current = module.state_dict(); parameters = dict(module.named_parameters())
    if current.keys() != state.keys():
        raise ValueError('Resume module keys changed')
    for key, expected in current.items():
        actual = state[key]
        if actual.shape != expected.shape or actual.dtype != expected.dtype or not torch.isfinite(actual).all():
            raise ValueError('Invalid resume tensor: '+key)
        if key not in parameters and not torch.equal(expected.cpu(), actual.cpu()):
            raise ValueError('Resume altered fixed geometry/bounds: '+key)


def rng_state():
    return dict(torch_rng=torch.random.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                python_rng=random.getstate(), numpy_rng=np.random.get_state())


def restore_rng(payload):
    torch.random.set_rng_state(payload['torch_rng'])
    torch.cuda.set_rng_state_all(payload['cuda_rng'])
    random.setstate(payload['python_rng']); np.random.set_state(payload['numpy_rng'])
