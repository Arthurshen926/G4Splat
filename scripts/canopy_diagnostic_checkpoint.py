"""Atomic diagnostic artifacts; these are NOT production teacher checkpoints."""
import json
import os
from pathlib import Path
import tempfile
import torch


def _write_atomic(path, writer, *, replace):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w+b', dir=path.parent, prefix='.'+path.name+'.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            writer(handle); handle.flush(); os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # Atomic publish without overwriting an existing checkpoint.
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def save_diagnostic_checkpoint(path, payload):
    if payload.get('diagnostic_only') is not True: raise ValueError('Diagnostic artifacts only')
    _write_atomic(path, lambda handle: torch.save(payload, handle), replace=False)


def publish_diagnostic_metrics(path, payload):
    encoded = json.dumps(payload, indent=2, allow_nan=False).encode('utf-8')
    _write_atomic(path, lambda handle: handle.write(encoded), replace=True)
