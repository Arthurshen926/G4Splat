# Unified outdoor reconstruction

The unified mainline has one evidence contract, one from-scratch mixed teacher
and one standard 3DGS export. Historical 40k/60k/68k PLY files are benchmarks,
not valid parents of a unified run.

## End-to-end command

```bash
/root/miniconda3/envs/g4splat/bin/python \
  scripts/run_cambridge_unified.py \
  --scene StMarysChurch \
  --stage all \
  --gpu 0
```

The output topology is:

```text
StMarysChurch_unified_v1/
├── evidence/
│   ├── evidence_manifest.json
│   ├── colmap_tracks.npz
│   └── mast3r_tracks.npz
├── initialization/
│   ├── surface_seed.npz
│   └── foliage_seed_gaussians.pth
├── teacher/
│   ├── unified_teacher_checkpoint.pth
│   └── unified_teacher_state.pth
├── student_3dgs/
│   ├── point_cloud/iteration_30000/point_cloud.ply
│   ├── cameras.json
│   └── distillation_manifest.json
├── evaluation/
└── pipeline_manifest.json
```

Stages are restartable:

```bash
python scripts/run_cambridge_unified.py --stage evidence
python scripts/run_cambridge_unified.py --stage initialize
python scripts/run_cambridge_unified.py --stage train_teacher
python scripts/run_cambridge_unified.py --stage distill_student
```

Selecting a later stage validates and reuses completed prerequisites.

## Contracts

`evidence_manifest.json` keeps COLMAP, posed-MASt3R, Chart, plane and
inverse-depth-cache artifacts separate and content-hashed. Sparse archives
retain track ids, camera support, sequence support, role posteriors and
position covariance. Inverse-depth fusion is a cache; training still constructs
independent Chart, plane, inverse-depth and ordinal losses.

Initialization assigns:

- rigid tracks and gated Chart residuals to native 2D surfels;
- linear tree tracks to static 3D trunk/branch Gaussians;
- nonlinear tree tracks and the occlusion-aware ray/depth visual hull to the
  canonical 3D crown;
- cloned cross-sequence crown support to the training-only dynamic leaf branch.

The teacher samples every real RGB camera from iteration one. Surface
densification statistics are accepted only for rigid-dominant primitives, so
canopy residuals cannot allocate 2D surfel topology. The teacher checkpoint
contains both models, sky, spatial appearance/uncertainty, dynamic codes,
optimizers, topology statistics, samplers, RNG states, curriculum phase,
evidence hash and Python/CUDA implementation hashes.

Full-resolution RGB is decoded on demand by a bounded view cache; constructing
1,487 calibrated Cambridge cameras does not decode or retain the entire image
set. This replaces the upstream eager loader, which can exceed 90 GB before the
first optimization step, while preserving the exact off-axis intrinsics.

During the dynamic phases, learned spatial uncertainty is detached and reused
as a static-consistency weight for the canonical crown. Inconsistent leaf
locations therefore stop pulling the canonical geometry toward a broad
cross-frame average. Their complete RGB residual and a masked image-gradient
loss are assigned to the sequence-conditioned dynamic leaf branch. Geometry
sources retain their own finite-valid masks; invalid depth-cache values cannot
propagate NaNs through a nominally zero loss weight.

## Standard downstream interface

The canonical export contains only the usual Graphdeco fields:

```text
x y z
f_dc_*
f_rest_*
opacity
scale_0 scale_1 scale_2
rot_0 rot_1 rot_2 rot_3
```

There are no role, UV-gate, uncertainty, sequence or mixed-renderer fields.
The PLY can therefore be loaded as an ordinary standard 3DGS model. Downstream
projects should use their normal 3DGS loader and the exported `cameras.json`;
they do not import G4Splat's mixed CUDA extension. `distillation_manifest.json`
contains executable schema checks and states every removed teacher dependency.

The canonical model intentionally represents rigid geometry, static
trunk/branch structure, canonical crown and a Gaussian sky shell. A single
static PLY cannot preserve every traversal-specific leaf deformation at once.

## Fast mainline validation

Use the fast profile while iterating on geometry/foliage design:

```bash
/root/miniconda3/envs/g4splat/bin/python \
  scripts/run_cambridge_unified.py \
  --scene StMarysChurch \
  --profile fast \
  --stage all \
  --gpu auto
```

It writes a separate `StMarysChurch_unified_fast_v1` run, so it cannot resume
or overwrite the 80k quality run. The 30k teacher schedule is:

| phase | iterations |
| --- | ---: |
| canonical bootstrap | 1–6,000 |
| bounded surface/volume topology | 6,001–12,000 |
| sequence-conditioned dynamic foliage | 12,001–21,900 |
| per-candidate ownership cleanup | 21,901–27,000 |
| canonical polish | 27,001–30,000 |

The standard 3DGS student is shortened from 30k to 10k for this profile.
Teacher surface topology has a 1.2M global budget and a 20k per-event growth
budget; both can be overridden from the pipeline CLI. The quality profile
keeps the 80k/30k schedules for the final benchmark after the fast profile
has demonstrated a real canopy and rigid-region improvement.

Before running a teacher, the pipeline requires 12,000 MiB of free physical
GPU memory. `--gpu auto` selects the GPU with the most free memory; an explicit
busy GPU fails immediately with a per-GPU memory report instead of spending
minutes preparing the scene and later surfacing only a `CalledProcessError`.

## Validation levels

A short integration run is useful only for verifying that every phase executes:

```bash
python scripts/train_unified_outdoor_teacher.py \
  -s DATASET -m OUTPUT \
  --evidence-store EVIDENCE --initialization INITIALIZATION \
  --iterations 100 --densify_from_iter 10 \
  --densification_interval 10 --volume-densify-every 10 \
  --replacement-every 5
```

It is not a reconstruction-quality result. The fast 30k/10k profile is the
normal design-validation run; the default 80k teacher and 30k render-space
student schedules remain the final quality run. Evaluation always reports
canonical and conditioned teacher outputs separately and then evaluates the
exported standard PLY through a fresh loader.
