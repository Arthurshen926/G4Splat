"""Process-local experimental backend for read-only candidate audits only."""
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experimental/canopy_optical_depth_rasterizer'),
               str(ROOT / '2d-gaussian-splatting/submodules/diff-surfel-rasterization')]
import torch
import diff_surfel_rasterization as original
import canopy_optical_depth_rasterization as experimental
from scripts import audit_canopy_candidate_contribution as audit
from outdoor import hybrid_gaussian_renderer as renderer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists():
        raise ValueError('Fresh experimental output required')
    old_backend = original._C
    provenance = dict(experimental=True, read_only=True,
        original_binary=str(old_backend.__file__), original_sha256=digest(old_backend.__file__),
        experimental_binary=str(experimental._C.__file__), experimental_sha256=digest(experimental._C.__file__),
        kernel='volume_projected_optical_depth__surface_unchanged__same_support_and_order')
    # Clone only the render function into private globals. Preserve its path
    # guard, changing the expected path to the explicitly approved extension.
    # No production module globals, source files or module identities change.
    source = inspect.getsource(renderer.render_hybrid)
    old_import = 'from diff_surfel_rasterization import ('
    old_path = '(_LOCAL_RASTER_ROOT / "diff_surfel_rasterization").resolve()'
    if source.count(old_import) != 1 or source.count(old_path) != 1:
        raise RuntimeError('Renderer adapter contract changed')
    source = source.replace(old_import, 'from canopy_optical_depth_rasterization import (')
    source = source.replace(old_path, '_EXPERIMENTAL_BINARY_PARENT')
    namespace = dict(renderer.__dict__)
    namespace['_EXPERIMENTAL_BINARY_PARENT'] = Path(experimental._C.__file__).resolve().parent
    source = 'from __future__ import annotations\n' + source
    exec(compile(source, '<isolated_optical_depth_render_adapter>', 'exec'), namespace)
    audit.render_hybrid = namespace['render_hybrid']
    provenance['adapter_source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
    audit.main()
    if digest(old_backend.__file__) != provenance['original_sha256']:
        raise RuntimeError('Original binary changed during audit')
    with (output / 'experimental_backend.json').open('x') as stream:
        json.dump(provenance, stream, indent=2)


if __name__ == '__main__':
    main()
