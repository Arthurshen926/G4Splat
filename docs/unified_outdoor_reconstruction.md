# Cambridge native Hybrid Teacher mainline

The authoritative reconstruction path is a single fixed-camera,
MASt3R/MAtCha/G4 evidence-to-native-Hybrid-Teacher pipeline. It does not train
or report a student model, does not read COLMAP point/track geometry, and does
not initialize from a historical trained Gaussian PLY.

`scripts/run_cambridge_unified.py` is retained only as a compatibility name.
It delegates to `scripts/run_cambridge_hybrid_teacher.py`; there is no second
trainer, checkpoint format or evaluator behind it.

## End-to-end command

```bash
export LD_LIBRARY_PATH=/root/miniconda3/envs/g4splat/lib:${LD_LIBRARY_PATH:-}
/root/miniconda3/envs/g4splat/bin/python \
  scripts/run_cambridge_hybrid_teacher.py \
  --scene StMarysChurch \
  --profile quality \
  --stage all \
  --gpu 2
```

The equivalent compatibility command is:

```bash
/root/miniconda3/envs/g4splat/bin/python \
  scripts/run_cambridge_unified.py \
  --scene StMarysChurch \
  --profile quality \
  --stage all \
  --gpu 2
```

The active stages are:

```text
prepare_cameras
build_mast3r_tracks
build_charts
build_evidence
initialize_teacher
train_teacher
evaluate_teacher
export_geometry
```

`distill_student` is intentionally rejected. A standard 2DGS/3DGS conversion
would be a separately labelled approximation and must not be silently reported
as the native Teacher result.

## Data and evidence contract

- Cambridge intrinsics and poses are fixed and retain exact off-axis
  principal points.
- `cameras.bin` and `images.bin` are serialization containers only.
  `points3D.bin` and COLMAP tracks are not geometry inputs.
- MASt3R pointmaps/tracks, MAtCha Charts, planes, DAV2 ordinal depth and
  foliage ray intervals remain content-addressed observations.
- Renderer births are disposable optimization variables. Track ids,
  observation pixels/depths, source roles, tree instances and lineages remain
  persistent after split/prune.
- RGB, geometry and topology use independent camera schedules.

## Representation and training

The mature rigid stage is a native perspective-correct 2D surfel model. The
mixed stage consumes its validated PLY/handoff manifest, keeps rigid geometry
and topology fixed, and permits only structural SH appearance polishing.
There are no extra rigid completion births in this handoff experiment.

Foliage is represented by static skeleton, canonical crown and
sequence/time-conditioned dynamic-leaf 3D EWA Gaussians. Structural surfels
and volume Gaussians are emitted into one tile list, share center-depth
sorting, and are composited in the same CUDA front-to-back loop.

The quality profile uses a 1.5M volume budget and at most 12k net growth slots
per topology event. Growth is still constrained by measured screen-space
deficit, role/instance/owner balance, posterior reliability and a smooth
startup ramp. Volume opacity uses the same conservative `0.004` learning rate
as colour/topology formation; sparse owner scheduling is not compensated by
accelerating opacity alone.

## Checkpoint and reuse contract

`hybrid_teacher_checkpoint.pth` stores the two Gaussian families, role and
observation metadata, sky, spatial appearance/uncertainty, temporal codes,
optimizers, topology statistics, camera schedules, RNG states and immutable
input/implementation hashes.

A completed run is reusable only when all causal fields match, including:

- evidence and initialization content hashes;
- exact rigid PLY and handoff hashes;
- training profile, horizon and all model capacities;
- geometry-gradient ratio;
- mature-surface policy and completion-seed count;
- volume opacity schedule;
- trainer, renderer, Gaussian model and CUDA implementation hashes;
- final checkpoint content hash.

Changing one of these fields makes the result stale instead of silently
reusing it.

## Evaluation scopes

The evaluator reports three deliberately separate scopes:

1. full 1,487-view database reconstruction under the exact historical uint8
   static-mask protocol;
2. canonical rendering on a disjoint 64-image official Cambridge query set at
   ground-truth poses;
3. conditioned database-view diagnostics, including tree/non-tree and
   high-frequency regions.

Only canonical database rendering is comparable to the historical
`18.756 / 0.830 / 0.0821` result. Conditioned results have no ordinary static
2DGS equivalent. Query64 evaluates reconstruction generalization at known
poses; it is not yet a camera-pose localization benchmark.

## Current downstream interface

The native Teacher requires `outdoor.hybrid_teacher_api` and the mixed CUDA
extension. It is not load-equivalent to an ordinary 2DGS or 3DGS PLY.
`export_geometry` provides geometry inspection artifacts only.

A universal standard PLY export/distillation path is currently absent by
design after the decision to optimize and report the Teacher directly. If a
downstream project requires a standard renderer, that conversion must be
implemented and evaluated as an explicit second model rather than being
claimed as lossless.
