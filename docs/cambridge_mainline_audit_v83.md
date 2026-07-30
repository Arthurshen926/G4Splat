# Cambridge unified hybrid mainline audit v83

This audit deliberately uses an end-to-end definition of "implemented". A
feature counts as complete only if its evidence producer, serialized identity,
training consumer, split/prune lifecycle, renderer and evaluator agree. A
metadata field or CLI option by itself does not count.

## Bottom line

The historical recommendations are **not all implemented**, and the remaining
quality gap cannot honestly be described as one hyperparameter problem.
However, the native mixed renderer is real and the active failure in v82 was
not the 2DGS/3DGS representation itself. Two measured optimization-contract
defects were still present:

1. at the 1.5M volume cap, global low-utility retirement removed 11,900 dynamic
   rows at iteration 8k, while evidence-adaptive splitting recreated only
   6,704 dynamic rows and 5,296 canonical rows. About 5.2k rows per event were
   silently transferred from dynamic leaves to the canonical crown;
2. volume topology stopped with the `dynamic_appearance` phase at iteration
   8,640 even though every event still had 1.4--1.5M integrated screen-space
   split demand.

The tree/building boundary also had an independent supervision hole:
`1-p_boundary_uncertain` simultaneously erased RGB ownership, free-space and
high-frequency evidence at the pixels that most need smaller footprints.

v83 fixes these three contracts without moving or re-densifying the measured
v74 rigid scaffold. It is resumed only from the content-hashed v82 8k
checkpoint, expands the safe volume budget to 2M, allows 20k net rows per
event, keeps topology active through 10k, makes capacity retirement
physical-role matched, and replaces the binary boundary erasure with a
continuous confidence floor of 0.25.

## Status of the external recommendations

| Recommendation | Status | Evidence / remaining gap |
| --- | --- | --- |
| Exact fixed Cambridge cameras including principal point | Complete | One K/pose geometry digest is consumed by the frontend, Teacher, native renderer and evaluator. |
| Remove COLMAP point/track geometry | Complete | `points3D.bin` is not opened; COLMAP camera/image files remain only a camera-container format. |
| No historical trained Gaussian initialization | Complete | The rigid handoff is evidence-seeded and trained inside this pipeline; the mixed stage binds its PLY and manifest by content hash. |
| MASt3R/Chart/DAV2/G4 evidence store | Complete as an immutable store | Every artifact and native raster is content-addressed, but the richer representations below remain partial. |
| Independent RGB / geometry / topology sampling | Complete | Schedules, epoch state and RNG are checkpointed independently. |
| Task semantics, physical primitive role and persistent metadata | Complete | Source, role, track, tree instance, observation owner and split lineage survive topology mutation and optimizer migration. |
| Observation-level rendered-surface track factor | Complete | Calibrated observation pixels constrain rendered surface intersections rather than pulling every child centre back to one XYZ. |
| Source-resolution Chart and pointmap factors | Complete | Losses run at the actual evidence raster; low-resolution evidence is not upsampled and relabelled as high-resolution geometry. |
| Occlusion-aware visual-hull / real ray-depth interval posterior | Complete for the selected 256-view factor | Hit/free/unknown intervals retain camera/pixel identity and use analytic transmittance with explicit camera-z to unit-ray conversion. It is not full-database dense visibility BA. |
| Complete-lineage dynamic ray likelihood | Complete | Sampling is by observation-lineage key, then all descendants contribute to the same ray likelihood. |
| Native 2D surfel + 3D EWA same-tile renderer | Complete forward and backward path | Both primitive types share tile emission, sorting and alpha compositing. Sorting is by centre depth; exact pixel-varying surfel intersection order remains an approximation for large oblique footprints. |
| Adaptive volume split/prune and Adam migration | Complete | Binary/quaternary mass-preserving split, atomic prune/split and row-exact optimizer migration have regression tests. |
| Per-candidate local replace-and-retire | Complete but secondary | It is local cleanup, not the primary birth mechanism. |
| Spatial uncertainty | Complete | Confidence is spatial/per-primitive and curriculum routed, not two scalar escape variables per image. |
| Static trunk/branch role | Partial | The current initialization has 985 static-skeleton rows, but no external trunk/branch semantic predictor or line/cylinder-supported reconstruction. |
| Canonical/dynamic exclusivity | Partial | Exact owner mass plus capped temporal fallback is implemented. It is still a replacement-style approximation, not deformation of one shared canonical primitive. |
| Probabilistic tree-instance visual hull | Partial | Tree instances and positive/free/unknown evidence exist, but the view set is selected and rigid occlusion support is sparse. |
| Full MASt3R reciprocal descriptor track graph | Partial | Current tracks use exact-K pointmap overlap, ray triangulation and cycle checks over selected views. They are not a full-database reciprocal descriptor graph merged by union-find. |
| Learnable continuous MAtCha inverse-depth Chart atlas | Missing | Current Chart support is a discrete UV/anchor/adjacency observation graph. There is no learnable inverse-depth field with UV quadtree densification. |
| High-resolution small-object/boundary patch sampler | Missing | v83 repairs continuous boundary evidence, but does not add a dedicated patch stream for thin static objects. |
| See3D/facade atlas completion | Missing by scope | It is not allowed to become primary geometry in the active method. |
| Renderer-equivalent standard 2DGS/3DGS export | Missing by the current Teacher-only decision | Surface PLY alone is not the full mixed scene. |
| Camera-pose localization benchmark | Missing | Query64 measures canonical rendering at ground-truth calibrated query poses, not pose estimation accuracy. |

## Validation-chain findings

The following bugs could obscure real improvements and are now closed:

- the three-view 408--410 training diagnostic can no longer claim direct
  comparability to the historical full-1,487-view aggregate;
- query evaluation writes camera artifacts into its own output directory and
  cannot overwrite the Teacher's database camera contract;
- completed-result reuse binds the mature-surface policy, completion births,
  geometry gradient ratio, opacity learning rate and volume topology horizon;
- the runner now hashes `intrinsics_utils.py`, matching the trainer's actual
  implementation manifest. Previously a real completed Teacher could be
  rejected forever as stale because the runner omitted this hash;
- the runner now reads the semantic contract from the evidence manifest.
  Several reused evidence stores intentionally point at an immutable semantic
  contract outside their local directory, so assuming
  `evidence/task_semantics.json` could make a valid run fail at evaluation;
- the query database contract now comes from the evidence scene contract.
  Assuming `dataset/scene_manifest.json` was both wrong for this prepared
  Cambridge dataset and made the runner's official query64 stage diverge from
  the standalone evaluator that had produced the reported measurements;
- the v82-to-v83 migration is allowed only for the exact retained 8k trainer
  hash and exact 1.5M/12k to 2M/20k/10k contract. Evidence, schedules, model
  tensors and CUDA kernels must otherwise be identical.

The evaluation scopes remain intentionally distinct:

- target 408--410: known database training views, useful for conditioned tree
  ownership and smear diagnosis;
- query64: disjoint images and canonical map at ground-truth poses, useful for
  generalization but not a localization pose benchmark;
- database1487 canonical: the only current scope that can be compared with the
  historical `18.755777 / 0.830073 / 0.082067` uint8 static protocol.

## v83 causal experiment

```text
resume state:           exact v82 iteration 8,000
rigid surface:          v74, 393,991 rows
surface policy:         appearance only, geometry/topology frozen
completion births:      zero
volume at resume:       1,500,000
volume budget:          2,000,000
net growth per event:   <=20,000
volume topology end:    iteration 10,000 (phase independent)
mixed horizon:          iteration 12,000
boundary evidence:      continuous 1 - 0.75*p_boundary
capacity retirement:    role-matched lineage-safe replacement
```

No hard quality threshold is used to declare success. The final conclusion is
based on continuous metric changes, topology/ownership audits and rendered
comparisons under the exact same evaluator.

## Results

The formal v83 run is:

```text
/mnt/pool/sqy/G4Splat_runs/cambridge_hybrid_contract_closed_v7/
  StMarysChurch_v35_tree408_evidence_repair/
  teacher_hybrid12k_v83_boundary_capacity_repair_v38
```

It completed in 1,375.56 seconds. Volume topology added exactly 20,000 net
rows at each event from 8.1k through 10k, reached 1.9M rows, and then stopped
at the explicit topology horizon. Selected parents still had a median
projected radius of 8--9 pixels against the 2-pixel target, with roughly
650k--760k eligible rows per event. Thus these births were responding to a
measured screen-bandwidth deficit rather than an unconditional population
schedule. Role counts remained conserved by the role-matched reallocation
contract.

| State / scope | raw PSNR | static PSNR | non-tree PSNR | tree PSNR | boundary-in PSNR | boundary-out PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v82 8k, target408--410 | 14.002581 | 15.104729 | 14.533523 | 15.174526 | 12.398401 | 13.797612 |
| v83 10k, target408--410 | 14.085093 | 15.226522 | 14.577419 | 15.284390 | 12.320725 | 13.563354 |
| v82 12k, target408--410 | 14.408593 | 15.654320 | 14.622808 | 15.792972 | 12.727686 | 13.530282 |
| v83 12k, target408--410 | 14.360485 | 15.591201 | 14.605890 | 15.725761 | 12.559285 | 13.469814 |
| v82 12k, query64 | 13.900970 | 15.253034 | 15.252915 | 14.725764 | 13.731312 | 12.328170 |
| v83 12k, query64 | 13.898924 | 15.250714 | 15.252870 | 14.707084 | 13.715785 | 12.321932 |

The causal conclusion is deliberately narrower than “more points are
better.” The v83 repair is correct and its 10k state improves over its exact
v82 8k predecessor, but after the same 12k polish it is slightly worse than
v82 final and query64 is effectively unchanged. Therefore the v82 capacity
and role-transfer defects were real, but they are not the dominant final
image-quality bottleneck. Adding 400k optical-mass-preserving children without
enough subsequent evidence-localized fitting does not by itself remove the
tree smear.

The next active-chain audit found a more direct ownership defect: the
`appearance_only` mature-surface policy froze surface opacity. Tree-front,
sky and transient ownership losses computed a surface-alpha correction, but
that gradient was then discarded. The dense low-area foreground surfels
visible in the target alpha renders were therefore inherited unchanged from
the rigid handoff.

The first v84 correction deliberately allowed only a positive semantic
opacity-logit gradient, expecting that gradient descent could only reduce
alpha. The direction was correct but the optimizer mechanism was not: Adam
normalizes even a very small persistent positive gradient. At 10.5k, while
surface xyz/scale/rotation were still bit-exact, mean alpha had fallen from
0.641426 to 0.074431; 393,108 of 393,991 surfels had decreased and only
11.60% of alpha mass remained. The diagnostic was stopped and rejected. This
is an important counterexample to treating “one-sided gradient” as “local
retirement.”

v85 keeps mature surface opacity completely frozen in Adam. Retirement occurs
only at the existing local replacement audit:

- jointly sorted responsibility accumulates tree-front depth claims,
  sky/transient conflicts, independent sequence coverage, posterior depth and
  volume-only RGB improvement per surfel;
- the resulting posterior is continuous and squared to concentrate demand
  spatially without a pass/fail quality gate;
- positive retirement increments are globally rescaled so one event removes
  at most 0.25% of current structural optical depth;
- the update is applied as a remaining-optical-depth ratio, preserving
  monotonicity under repeated audits.

At the retained v85 10.5k checkpoint, surface geometry remained bit-exact,
total structural optical depth retained 99.9135%, mean alpha changed by only
`-0.000181`, and no surfel increased. The strongest local alpha reduction was
0.281. At 10.1k and 10.2k the realized per-event removals were only 0.0474%
and 0.0221%, below the configured budget because the evidence did not request
more. This closes the global-extinction failure while retaining a real local
exit path.

## v85 formal completion and full-database result

The formally completed run is:

```text
/mnt/pool/sqy/G4Splat_runs/cambridge_hybrid_contract_closed_v7/
  StMarysChurch_v35_tree408_evidence_repair/
  teacher_hybrid12k_v85b_bounded_local_retirement_v42
```

Its final state has 393,900 surface surfels and 1,899,998 volume Gaussians:
986 static skeleton, 636,316 canonical foliage and 1,262,696 conditioned
foliage rows. Surface cleanup removed 91 rows, of which 65 were selected by
the bounded semantic retirement path. At the final audit, mean retirement
fraction was 0.071044, maximum was 0.962698, and the realized structural
optical-depth removal in that event was 0.00444%. This verifies that v85
implements local replace-and-retire without reproducing v84's global alpha
collapse.

All values below use the historical uint8 metric implementation. Target
408--410 is conditioned; query64 and database1487 use the canonical map.

| Run / scope | raw PSNR | raw SSIM | raw MAE | static PSNR | non-tree PSNR | tree PSNR | boundary-in PSNR | boundary-out PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v82 12k, target408--410 | 14.394754 | 0.692457 | 0.127521 | 15.632584 | 14.587632 | 15.773694 | 12.722868 | 13.494835 |
| v83 12k, target408--410 | 14.346549 | 0.692716 | 0.128275 | 15.569458 | 14.570977 | 15.706412 | 12.554454 | 13.434585 |
| v85 12k, target408--410 | 14.346666 | 0.692715 | 0.128272 | 15.569590 | 14.571494 | 15.706505 | 12.554061 | 13.435209 |
| v82 12k, query64 | 13.901396 | 0.689703 | 0.146921 | 15.254889 | 15.253269 | 14.723755 | 13.736189 | 12.304292 |
| v83 12k, query64 | 13.899348 | 0.689646 | 0.146957 | 15.252563 | 15.253223 | 14.705126 | 13.720832 | 12.298246 |
| v85 12k, query64 | 13.899584 | 0.689672 | 0.146952 | 15.252790 | 15.253766 | 14.704447 | 13.720564 | 12.298830 |
| v85 12k, database1487 | 14.195066 | 0.723186 | 0.138399 | 15.643804 | 15.834111 | 13.226037 | 12.122304 | 12.366664 |

The full-database evaluator had one reporting defect: it required
`evaluation_mode == hybrid` before marking a historical comparison valid,
even when a canonical-only evaluation used the exact same 1,487 views,
cameras, raster size and uint8 protocol. This did not change any pixels or
metrics. The protocol check now accepts the canonical branch of either mode
and continues to reject subsets, rigid-only renders and oracle-conditioned
renders. A regression test fixes that contract.

The exact comparable historical static result remains:

```text
PSNR 18.755777 / SSIM 0.830073 / MAE 0.082067
```

Therefore v85 is still behind by 3.111973 dB static PSNR, 0.138639 SSIM and
0.041970 MAE. The difference is much larger than all v82--v85 variations and
cannot honestly be described as restored reconstruction quality.

## Decisive causal diagnosis

The full-view decomposition separates mixed-expression progress from base-map
quality. The retained rigid-only v74 full-view result had non-tree static PSNR
15.8352 and tree static PSNR 9.2115. v85 has 15.8341 and 13.2260
respectively. Thus:

1. Native mixed rendering and the foliage volume branch do work: canonical
   tree quality improves by about 4.01 dB over rigid-only, and overall static
   quality improves by about 1.16 dB.
2. The building/non-tree map does not improve at all. It is inherited almost
   exactly from the weak rigid 2DGS handoff, so building blur is not primarily
   caused by volume over-densification or cross-type compositing.
3. v83 corrected a real capacity/role-transfer defect and v85 corrected a
   real surface-ownership defect. Their near-identical final images prove
   that neither defect is the dominant remaining quality bottleneck.
4. Twelve thousand mixed iterations are enough to expose this plateau:
   topology reached 1.9M rows by 10k and received 2k more fitting iterations;
   v82, v83 and v85 converge to essentially the same query result. A longer
   continuation of the same objective may produce a small gain, but there is
   no evidence that it can recover a 3.11 dB static gap.

The remaining architectural gaps are not all implemented:

- the current Chart representation is a discrete UV/anchor observation graph,
  not a learnable continuous inverse-depth atlas or adaptive quadtree;
- the MASt3R graph is selected-view pointmap overlap, not a full
  descriptor-level reciprocal union-find track database;
- trunk/branch evidence is only a sparse static skeleton (986 rows), not an
  instance-aware semantic branch model;
- the visual hull uses a selected 256-view set and sparse rigid occlusion,
  not a dense all-view visibility posterior;
- canonical and conditioned foliage use exact ownership plus a capped temporal
  fallback, not one canonical primitive set deformed by time;
- same-tile mixed sorting uses Gaussian center depth, so very large oblique
  surfels still approximate per-pixel ordering;
- high-resolution small-object/foliage patch sampling and a facade texture
  atlas are absent;
- query-at-ground-truth-pose evaluation is not a Cambridge localization pose
  benchmark.

These are the reasons it would be incorrect to say that all prior external
recommendations have been fully implemented.

## Next quality-bearing mainline

The next experiment should be rigid-first, rather than another mixed suffix
repair:

1. Preserve v85 as the correctness reference and v82 as the best current
   target-view checkpoint.
2. Rebuild the rigid/native-2DGS stage from zero with exact Cambridge cameras
   and real training RGB, while keeping MASt3R/MAtCha/G4Splat geometry as
   uncertainty-normalized residual factors rather than letting sparse or
   inconsistent geometric anchors dominate birth and topology.
3. Keep structural position, scale and opacity trainable until real
   screen-space coverage and full-view non-tree quality converge. Select the
   handoff continuously from the validation curve; do not introduce a binary
   quality gate.
4. Implement the missing continuous Chart atlas/full reciprocal track graph
   and a high-resolution error-patch sampler before increasing point count.
5. Only then attach the mixed foliage branch, allocate canonical/dynamic
   capacity from ray-local residual evidence, and provide a longer
   post-topology fitting interval.
6. Evaluate reconstruction and pose localization separately; the latter needs
   a pose-estimation protocol rather than rendering query images at known
   poses.

This ordering follows the measured causal chain: first recover the building
and structural base, then use the volume branch for the tree residual it has
already demonstrated it can improve.
