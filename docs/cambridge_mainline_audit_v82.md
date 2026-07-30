# Cambridge mainline end-to-end audit v82

This audit uses a stricter definition of “implemented”: a component is complete
only when its producer, persisted identity, training consumer, topology
lifecycle, renderer and evaluator agree. A class, CLI flag or metadata field by
itself does not count.

## Bottom line

Not every historical recommendation is implemented, and earlier status reports
were too optimistic about two front-end items. The native mixed renderer and
most training-side contracts are real. A continuous learnable MAtCha
inverse-depth atlas and a broad descriptor-based MASt3R track graph are not.

The repeated v77--v79 score similarity was not caused by the evaluator. Three
training effects cancelled the intended changes:

1. the v74 rigid handoff was still allowed to change geometry/topology and also
   received 20k new completion births;
2. dynamic observation rows were sampled before split descendants were grouped,
   so the claimed complete-lineage posterior depended on row order;
3. increasing only volume opacity learning rate made the tree layer opaque
   before its colour and screen bandwidth were ready.

v82 is the first clean combination experiment: exact v74 rigid geometry,
appearance-only surface polishing, zero completion births, complete-lineage ray
factors, conservative optical learning and the proven 1.5M volume capacity.

## Implementation matrix

| Recommendation | Status | Current evidence |
| --- | --- | --- |
| Fixed Cambridge calibration including `cx,cy` | Complete | One exact camera contract is consumed by MASt3R retargeting, 2DGS, mixed rendering and evaluation. |
| No COLMAP point/track geometry | Complete | `points3D.bin` is not read; `cameras.bin/images.bin` are serialization only. |
| No historical all-real trained Gaussian initialization | Complete | The rigid stage is evidence-seeded and trained in the current pipeline; the mixed stage consumes its content-hashed handoff. |
| Task-level soft semantics and physical roles | Complete | Rigid/canopy/sky/transient/unknown fields are used separately for RGB, geometry, topology and ownership. |
| Persistent source/track/role/instance/observation/lineage metadata | Complete | Metadata survives split, prune, optimizer migration and checkpoint reload. |
| Independent RGB, geometry and topology schedules | Complete | Camera schedules and evidence-epoch state are independently checkpointed. |
| Source-resolution Chart/pointmap factors | Complete | Geometry is compared to the rendered surface at the native evidence raster; unsupported Chart samples need not become primitives. |
| Observation-level rendered-surface track factor | Complete | Track pixels, calibrated rays and rendered intersections are used instead of pulling every descendant centre to one XYZ. |
| Occlusion-aware foliage ray intervals | Complete | Hit/free/unknown intervals retain camera and pixel identity; camera-z is explicitly converted to unit-ray distance. |
| Complete split-lineage dynamic likelihood | Complete in v82 | Observation-lineage keys are sampled first and all descendants of selected keys enter one transmittance likelihood. |
| Native 2D surfel + 3D EWA same-tile CUDA rendering | Complete forward path; residual ordering approximation | Both primitive types share tile emission, centre-depth radix order and the pixel compositing loop. Perspective-correct surfel intersection depth is pixel-dependent after sorting, so large oblique surfels retain a measurable centre-depth approximation. |
| Volume adaptive split/prune and Adam migration | Complete | Two/four-child optical-mass-preserving splits, atomic prune+split and exact state-row migration are tested. |
| Local replace-and-retire | Complete, secondary | Replacement is per spatial candidate/group and optical mass, not a global prefix gate. It is cleanup, not the primary geometry producer. |
| Static/canonical/dynamic foliage roles | Partial | All three roles render/train, but the measured static skeleton is only 387 rows and has no line/cylinder-supported track count in the current initialization audit. |
| Canonical/dynamic exclusivity | Partial | Exact-view dynamic mass locally suppresses canonical mass and temporal fallback is mass-capped. Dynamic-only instances without a canonical group cannot use that reference and remain a weaker contract than deforming one canonical primitive. |
| Spatial uncertainty | Complete | Uncertainty is image-spatial and curriculum-routed; it is not two scalars per image. |
| Probabilistic per-tree visual hull | Partial | Positive/free/unknown evidence, tree instances and sparse rigid occlusion are present. Coverage is selected-view/sparse-rigid rather than a full dense visibility BA. |
| MASt3R multi-view track graph | Partial | Exact-K pointmap overlap, target-raster observations, cycle checks and triangulation exist, but the graph is limited to selected MAtCha views and uses pose/overlap proposals rather than full-database descriptor reciprocal matching plus union-find merging. |
| MAtCha Chart atlas | Partial | UV identity, adjacency/Jacobian priors and permanent rendered inverse-depth factors exist. There is no continuous learnable inverse-depth atlas or UV quadtree densification. |
| G4 visibility-based rigid completion | Partial/disabled in v82 | An evidence-based completion selector exists; v82 intentionally uses zero new births to preserve the measured rigid scaffold. There is no full visibility-verified birth/BA subsystem. |
| Small-object/boundary patch stream | Missing | Soft boundary ownership exists, but there is no dedicated high-resolution small-static-object sampler. |
| Facade atlas completion / See3D | Missing by current scope | It remains optional and is not allowed to write the primary geometry. |
| Standard 2DGS/3DGS export | Missing by latest decision | The direct native Teacher is the reported model. Geometry export is not a renderer-equivalent universal PLY. |
| Camera-pose localization benchmark | Missing | Query64 is rendered canonically at ground-truth poses; pose estimation accuracy is not measured. |

## Validation-chain audit

The repeated v77/v78/v79 results are genuine optimization outcomes:

- checkpoint iteration and implementation hashes match;
- fixed camera geometry and RGB content hashes match;
- target 408--410 and query64 use the same evaluator implementation;
- target 408--410 are database training-view diagnostics;
- query64 is disjoint and canonical, but uses ground-truth query poses;
- the historical `18.756 / 0.830 / 0.0821` result is comparable only to the
  canonical 1,487-view database field under the exact shared-640 uint8 static
  protocol, never to conditioned or query output.

Two actual orchestration problems were found and fixed:

1. `run_cambridge_unified.py` still launched a retired student pipeline and
   looked for `unified_teacher_checkpoint.pth`, while the active trainer writes
   `hybrid_teacher_checkpoint.pth`. It now delegates to the single
   Teacher-only runner and rejects `distill_student`.
2. completed-result reuse did not bind mature-surface policy, completion-seed
   count, geometry-gradient ratio or volume opacity schedule. Those fields now
   make a mismatched result stale.

Two evaluator defects were also found during v82:

1. a three-view `408--410` diagnostic was labelled comparable with the retained
   all-1,487-view `18.756` aggregate when only its raster/camera protocol
   matched. Historical comparability now additionally requires the exact full
   database view set;
2. `LazyScene` wrote `cameras.json` and the camera contract into the Teacher
   model directory during evaluation. A query evaluation could therefore
   replace the database contract with 64 query cameras. Evaluation scene
   artifacts are now isolated under their evaluation directory, and v82's
   database files were restored byte-for-byte from the validated v74 handoff.

A third recovery-only topology bug was found by the broad test suite: legacy
checkpoints without the dedicated conditioned-stat channel could mark dynamic
rows eligible while counting their observable population as zero. v82 uses one
consistent fallback evidence definition for eligibility, context assignment
and quota normalization.

## Causal checkpoint evidence

### Falsified shortcut: v80

v80 preserved the rigid surface but raised volume opacity LR from `0.004` to
`0.012`.

| Target 408--410 | 500 | 1k |
| --- | ---: | ---: |
| conditioned tree PSNR | 13.343 | 12.569 |
| conditioned non-tree PSNR | 14.744 | 14.528 |

The volume-alpha render became nearly solid while colour remained
low-frequency. This is an optimization schedule failure, not useful
densification, so v80 was stopped.

### Successful ownership combination: v81 prefix

v81 restored optical LR `0.004`, retained the mature surface and used the
1.5M/12k volume topology family.

| Target 408--410 at 1k | v78 | v81 |
| --- | ---: | ---: |
| conditioned tree PSNR | 13.282 | **13.287** |
| conditioned non-tree PSNR | 14.026 | **14.802** |

Tree progress is preserved while adjacent non-tree content gains `0.776 dB`.
This is the first direct evidence that the current branch can combine the
former tree path with the stronger rigid scaffold instead of trading one for
the other.

v81 was deliberately restarted as v82 after the recovery-path topology
contract was repaired. The numerical production path is otherwise identical.

## Measured v82 prefix

The first 4k updates show a real improvement rather than another identical
score:

| Exact protocol | 1k | 2k | 4k |
| --- | ---: | ---: | ---: |
| target 408--410 conditioned tree PSNR | 13.287 | 13.756 | **14.147** |
| target conditioned non-tree PSNR | 14.802 | **14.978** | 14.679 |
| Query64 canonical raw PSNR | 12.939 | 13.473 | **13.787** |
| Query64 canonical non-tree PSNR | 14.542 | 14.859 | **15.113** |
| Query64 canonical tree PSNR | 12.074 | 13.095 | **13.998** |
| Query64 tree-boundary-inside PSNR | 11.677 | 12.388 | **13.102** |

At 2k, all 393,991 surface rows were compared against the handoff PLY:
`xyz`, tangent scale, rotation and opacity are elementwise identical with
maximum absolute difference zero. 382,121 rows changed DC appearance, as
intended by the appearance-only policy. This directly validates that foliage
progress did not secretly move or redensify the mature rigid geometry.

The new surface-only counterfactual identifies the remaining target-view
tradeoff:

| target 408--410 | surface-only 4k | conditioned mixed 4k |
| --- | ---: | ---: |
| non-tree PSNR | 15.207 | 14.679 |
| tree PSNR | 9.527 | **14.147** |
| tree-boundary-outside PSNR | 15.614 | 14.276 |

Thus the remaining adjacent-building regression is volume ownership spill, not
surface geometry drift. Query64 nevertheless improves because canonical volume
also fills genuinely missing content: at 1k it improved every one of the 28
tree-containing query views over the same checkpoint's surface-only render.
Later ownership-cleanup checkpoints must determine whether the target boundary
cost is temporary or persists.

Projected-footprint diagnostics also bound the mixed-ordering concern. On
408--410 at 4k, surface p99 radius is 14--16 px and only
`0.009%--0.044%` of visible surfels exceed 24 px; volume p99 is 19--21 px.
Some Query64 cameras do expose nearly edge-on free residual surfels with
singularly large preprocessing radii, so exact per-pixel cross-type order is
not claimed. They are reported by primitive row for cleanup diagnosis rather
than silently hidden.

## v82 experiment definition

```text
surface handoff:       v74 rigid 24k
surface rows:          393,991, geometry/topology frozen
surface updates:       SH appearance only
completion births:     0
initial volume rows:   1,037,077
volume budget:         1,500,000
growth/event:          <=12,000, smooth startup ramp
volume opacity LR:     0.004
mixed horizon:         12,000
retained checkpoints:  2k / 4k / 8k / 12k
```

The run is judged by continuous trends rather than a binary gate:

- tree PSNR/high-frequency measures should improve with owner-view exposure
  and smaller projected radii;
- non-tree and query64 must remain near the v74 scaffold instead of falling as
  volume density grows;
- surface count must remain exactly 393,991;
- volume growth must obey the measured screen-demand ramp and the 1.5M cap;
- canonical and conditioned results are reported separately.

Final v82 checkpoint metrics are appended after the complete run and exact-hash
evaluation.
