# Cambridge native hybrid Teacher mainline

This is the authoritative no-student reconstruction path for Cambridge
outdoor scenes.

## Representation and data contract

- Cambridge database intrinsics and poses are fixed exactly.
- `cameras.bin` and `images.bin` are camera serialization only.
- `points3D.bin`, COLMAP tracks, historical Gaussian checkpoints, and
  historical all-view RGB initialization are forbidden geometry inputs.
- MASt3R dense pointmaps are content-addressed in the evidence store and are
  converted into real fixed-camera multi-view tracks.
- MAtCha charts, G4 structure/plane factors, and inverse-depth caches remain
  separate measurements throughout training.
- MAtCha normalized coordinates are explicitly converted back to the original
  Cambridge world scale before initialization or metric depth supervision.
- The final artifact is the native mixed Teacher: perspective-correct 2D
  surfels and 3D EWA Gaussians share the CUDA tile list and depth ordering.
  No student is trained or exported.

Tree geometry is role-aware:

- rigid surfels own buildings and ground;
- static 3D Gaussians own supported trunk/branch structure;
- canonical 3D Gaussians own stable crown occupancy;
- sequence/time-conditioned 3D Gaussians own non-rigid leaf detail;
- spatial uncertainty owns irreducible image-local variation.

Single-view canopy pointmap samples can only enter the dynamic leaf branch and
are excluded from localization assets.

## End-to-end run

```bash
/root/miniconda3/envs/g4splat/bin/python \
  /root/G4Splat/scripts/run_cambridge_hybrid_teacher.py \
  --scene StMarysChurch \
  --profile quality \
  --stage all \
  --gpu 2
```

The quality profile trains for 50,000 iterations. The runner is restartable,
validates evidence hashes and the free-memory gate, then automatically
evaluates the final Teacher and writes localization-safe geometry assets.

## Rendering API

```python
from pathlib import Path
from outdoor.hybrid_teacher_api import load_hybrid_teacher

teacher = load_hybrid_teacher(
    Path("/path/to/hybrid_teacher_state.pth"),
    sh_degree=3,
)

canonical = teacher.render(view)
conditioned = teacher.render(
    view,
    conditioned=True,
    task=semantic_task_fields_for_view,
)
```

Canonical renders omit sequence-local dynamic leaves. Conditioned renders use
the training image identity and sequence/time code. Downstream localization
should use the fixed-camera scene contract, stable MASt3R tracks, and rigid
surface export rather than treating dynamic foliage as landmarks.
