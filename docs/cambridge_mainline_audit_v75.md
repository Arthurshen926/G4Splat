# Cambridge mainline causal audit v75

This audit supersedes “component exists” status reports.  A recommendation is
complete only when its producer, persisted identity, training consumer,
topology lifecycle, renderer and evaluator are mutually consistent.

## Bottom line

Not every external recommendation is implemented.  The native mixed renderer,
ray/depth foliage factors, role metadata, spatial uncertainty, local
replace-and-retire and conditioned-gradient ownership are real.  A learnable
continuous MAtCha inverse-depth atlas and a broad descriptor/cycle/union-find
MASt3R graph are still absent.

The weak v72/v73 results also did not measure the value of those repaired
components cleanly.  Two independent initialization/topology regressions
dominated them:

1. surface world scale was capped at `0.12 m`, copied from the foliage voxel
   size, despite a separate projected-radius controller;
2. the standard runner created up to 240k pointmap plus 180k Chart births,
   consuming most surface capacity before RGB topology measured a bandwidth
   deficit.

The v74 matched experiment restores the proven no-COLMAP rigid family:
120k pointmap births, 100k Chart candidates with only independently supported
rows renderer-admitted, a `0.5 m` world safety ceiling and a `24 px` projected
ceiling.  The full Chart/pointmap rasters remain external rendered-surface
factors; reducing births does not discard measurements.

## Strict official-query result

Primary field:
`protocol_aggregate.canonical.non_tree_static`, 64 official Cambridge query
images, fixed ground-truth query poses, zero database image overlap.

| Rigid run | Iteration | PSNR | SSIM | MAE |
| --- | ---: | ---: | ---: | ---: |
| v72, over-dense / scale 0.12 | 4k | 13.738 | — | — |
| v73, supported Chart but still over-dense / scale 0.12 | 4k | 13.655 | — | — |
| v74, matched birth density / scale 0.5 | 4k | 13.987 | 0.585 | 0.156 |
| v74 | 8k | 14.774 | 0.659 | 0.142 |
| v74 | 12k | 15.050 | 0.688 | 0.136 |
| v74 | 18k | **15.194** | **0.700** | **0.133** |
| v74 final | 24k | 15.176 | 0.702 | 0.133 |
| v72 final | 24k | 14.756 | 0.677 | 0.138 |
| v73 final | 24k | 14.698 | 0.671 | 0.139 |

At 12k, v74 already exceeds the 24k v72/v73 results by 0.29/0.35 dB.
The improvement is also visible in frame 00408: v72 writes foreground tree
appearance into elongated facade surfels, whereas v74 recovers a coherent
rigid tower and nave behind the tree.  Foliage quality must be judged only
after the mixed stage; the rigid mode intentionally omits all volume owners.

The corresponding 1,487-view database-fit diagnostic at the final v74
checkpoint is:

| Region | PSNR | SSIM | MAE |
| --- | ---: | ---: | ---: |
| raw RGB | 13.392 | 0.665 | 0.158 |
| static-valid | 14.486 | 0.616 | 0.153 |
| non-tree static | **15.835** | **0.705** | **0.123** |
| tree static, rigid counterfactual only | 9.212 | 0.163 | 0.312 |

The tree row is intentionally incomplete and cannot be used to rank a rigid
surface stage against a mixed reconstruction.  The non-tree result is 0.231
dB above the v69 mixed canonical non-tree diagnostic (15.604 dB), confirming
that the repaired scaffold itself is no longer the source of the mixed
regression.  The historical 18.756 result used a different static protocol;
the evaluator explicitly marks direct comparison to it invalid.

## Densification and geometry-factor diagnosis

v72 performed roughly 194k logged clones by 24k.  v74 performed no logged
clone through 12k and used approximately 243k measured replacing splits.
This is the expected topology for an initially broad but metric scaffold:
large visible footprints split into smaller tangent disks instead of filling
holes with duplicate low-bandwidth rows.

The geometry gradient is normalized against the RGB surface-XYZ gradient.
The current rigid target is 0.25; observed raw geometry/RGB gradient medians
are about 0.012/0.029 before adaptation.  Geometry therefore remains effective
without overwhelming colour optimization.

## Validation-chain repairs

The active evaluator now permits a machine-labelled rigid counterfactual on
the official query split, while still rejecting hybrid or conditioned
query rendering.  It marks the result as
`rigid_surface_stage_diagnostic`, not as a complete reconstruction or a
localization-pose benchmark.

The standard runner now:

- executes geometry-first native rigid training before quality/fast mixed
  training;
- binds the exact rigid PLY and handoff manifest hashes into current-result
  checks;
- binds current RGB bytes and camera identity into evaluation reuse checks;
- automatically evaluates both full database fit and the disjoint official
  query64 split;
- passes the validated rigid optimizer schedule, surface scale/radius and
  geometry-gradient ratio explicitly;
- uses profile-bound supported renderer-birth density instead of the previous
  hard-coded 240k/180k seed family.

## Implementation status

Complete and exercised:

- fixed calibrated Cambridge cameras with exact intrinsics;
- no active COLMAP point/track or historical trained-Gaussian initialization;
- source-resolution rendered Chart and MASt3R pointmap factors;
- observation-level rendered-surface track factors;
- probabilistic occlusion-aware foliage hull and analytic ray posterior;
- persistent primitive role, instance, lineage and observation metadata;
- static skeleton, canonical crown and sequence/time dynamic-leaf branches;
- spatial uncertainty;
- native 2D surfel + 3D EWA same-tile/depth-order renderer;
- volume split/prune with optimizer-state migration;
- per-candidate local replace-and-retire;
- conditioned exact-owner gradient routing.

Partial:

- rigid background completion is evidence-based at a mature handoff, but not
  a full visibility-verified birth/BA subsystem;
- trunk/branch representation exists, but measured line/cylinder support is
  sparse;
- boundary ownership exists, but there is no dedicated small-object patch
  training stream;
- official query RGB is evaluated at ground-truth pose; camera-pose
  localization accuracy is not yet measured.

Not complete:

- continuous learnable MAtCha inverse-depth atlas and UV quadtree;
- broad descriptor-reciprocal/cycle/union-find MASt3R graph;
- universal standard 2DGS/3DGS export or distillation.

The remaining two geometry-front-end items are architectural work, not flags
that can truthfully be reported as enabled.
