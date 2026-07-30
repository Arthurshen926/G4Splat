# Cambridge mixed mainline audit v63

This note audits the current branch against all external reviews supplied in
this task. A component is called complete only when its producer, persisted
contract, training consumer, topology/checkpoint path, renderer and evaluator
are mutually consistent. Code presence alone is not completion.

## Bottom line

No: not every previous recommendation had been implemented, and not every
known defect had been closed. Several earlier status reports overstated
completion because they checked stored arrays or class names rather than the
end-to-end causal path.

This audit found and repaired six additional correctness/validation defects:

1. `rigid` evaluation still rendered canonical volume Gaussians, so the
   alleged surface-only counterfactual was false.
2. Loading the mature 24k rigid surface restarted its position-LR schedule at
   iteration one, raising the LR by roughly 40x and damaging the scaffold.
3. The conditioned branch passed base geometry/colour/opacity gradients into
   canonical crown and skeleton rows. Because evidence cameras are sampled
   about 12x more frequently in that branch, transient high-frequency
   residuals were repeatedly written into the canonical localization map.
4. Canonical crown splitting was then hard-disabled during
   `dynamic_appearance`, even though its broad primitives still had measured
   screen-bandwidth deficit. The branch was therefore polluted and unable to
   refine.
5. The nominal held-out trajectory was a 64-image subset of the 1,487-image
   training database. It has been quarantined and replaced by the official
   64-image Cambridge query split with a fail-closed disjoint camera
   contract.
6. The first absolute-iteration warm-start repair still used the mixed
   trainer's generic position-LR schedule family.  The rigid producer used
   `1.6e-5 -> 1.6e-6 / 20k`, while the mixed default was
   `1.6e-4 -> 1.6e-6 / 30k`.  At absolute iteration 24,001 this was
   `2.0536e-4` rather than the producer's `8.1770e-5` on St Mary's Church.
   The handoff now persists the complete surface optimizer contract; legacy
   v1 handoffs recover it only from an identity-checked sibling
   `result.json`.

The ownership repair keeps canonical/skeleton primitives in the conditioned
forward render for correct same-tile occlusion and depth sorting, but makes
them read-only for the conditioned loss. Exact dynamic owners alone receive
base-state gradients; temporal neighbours update only low-rank residuals.
Canonical topology now consumes only its unbiased full-database stream and
may perform optical-mass-preserving local splits throughout the topology
window.

## End-to-end implementation matrix

| Design item | Current status | Evidence or remaining gap |
| --- | --- | --- |
| Exact fixed Cambridge K/pose, including principal point | Complete and exercised | All 1,487 database cameras are identity checked. Query cameras use a separate calibrated contract. Camera and RGB identities are content-addressed. |
| No COLMAP point/track geometry or historical trained-Gaussian initialization | Complete on the active mainline | The fixed camera container is allowed; point/track geometry and historical model parents are rejected. The rigid handoff is the current no-COLMAP 24k surface. |
| Evidence Store and independent RGB/geometry/topology schedules | Complete and exercised | Schedules, evidence hashes, consumption state and resume contracts are checkpointed independently. |
| Plane/Chart/pointmap provenance and source-resolution factors | Complete as external factors | Plane no longer overwrites depth; native Chart and MASt3R pointmap factors are consumed at their actual evidence raster. |
| Observation-level persistent track factors | Complete for the current graph | Camera/UV/depth observations survive Gaussian split/prune and reach full Evidence Epoch coverage. |
| Strict descriptor MASt3R reciprocal/cycle/union-find graph | **Not complete** | The saved graph has 56 selected views, 4,968 tracks and 10,375 observations. It is fixed-camera projected pointmap agreement, not descriptor matching over the full 1,487-view database. |
| Continuous learnable MAtCha inverse-depth Chart atlas | **Not complete** | `ChartSurfaceModel.audit()` truthfully reports a discrete observation graph. There is no learnable atlas field or UV-domain quadtree refinement. |
| Occlusion-aware probabilistic foliage visual hull | Complete and exercised | Hit/free/unknown, rigid z-buffer occlusion, sequence/baseline/angle support, spatial covariance and per-ray intervals are retained. |
| True ray/depth posterior | Complete and exercised | Camera-z is converted to unit-ray distance before analytic transmittance likelihood; unknown rays are excluded from free-space loss. |
| Static trunk/branch, canonical crown and dynamic leaf roles | Complete in representation; evidence strength differs | Measured static skeleton exists but is sparse. Canonical crown and dynamic leaves are dense and retain role/instance/lineage metadata. No semantic-only fake skeleton is fabricated. |
| Sequence/time dynamic branch | Complete as explicit local replacement | Exact-camera base ownership, conservative same-sequence residual interpolation and local optical-mass replacement are active. |
| Spatial uncertainty | Complete | Per-view spatial grids estimate canopy/sky uncertainty. Uncertainty cannot scale early geometry gradients down as an escape path. |
| Native 2D surfel + 3D EWA same-tile renderer | Complete and tested | Both primitive families share one tile list, center-depth radix order and front-to-back pixel loop. |
| Volume adaptive split/prune and Adam migration | Complete and exercised | Role/instance/spatial allocation, mass-preserving split, child metadata, prune and optimizer-state migration are one transaction. |
| Per-candidate local replace-and-retire | Complete as a continuous cleanup mechanism | Local depth/RGB/posterior evidence drives gradual optical-depth retirement; there is no global pass/fail gate. |
| Canonical/conditioned gradient ownership | Repaired in v63 | Conditioned base gradients select exact dynamic rows only. Canonical/skeleton rows remain forward occluders but are read-only in that objective. |
| G4 visibility rigid completion behind foliage | **Partial** | Rigid occlusion and ownership factors exist, but there is no standalone background birth/verification producer. |
| Small-object patch stream and asymmetric boundary ownership | **Partial** | Boundary-aware masks/topology priorities exist; the specifically proposed small-object patch sampler is not a separate closed training stream. |
| Standard universal export | Deferred by the explicit teacher-only decision | The authoritative output remains the native mixed Teacher. A universal 2DGS/3DGS artifact must not be claimed without a real bake/distillation stage. |
| Database/query localization validation | Partially closed in v63 | Official query RGB reconstruction is now evaluated at ground-truth poses with zero database overlap. Camera-pose estimation/localization accuracy is still not measured. |

## Validation-chain repair

The old 64-view “held-out” adapter overlapped the database 64/64 and is now
stored only as:

`/mnt/pool/sqy/G4Splat_runs/cambridge_hybrid_contract_closed_v7/query_controls/StMarysChurch_train_holdout64_INVALID_OVERLAP_shared640`

The valid query adapter is:

`/mnt/pool/sqy/G4Splat_runs/cambridge_hybrid_contract_closed_v7/query_controls/StMarysChurch_official_query64_shared640_bilinear_v1`

Its source is the official `test_eval` split. The database/query scene
contracts prove zero source-image overlap. Query evaluation permits canonical
rendering only; per-training-image conditioned codes are rejected. This is a
ground-truth-pose map generalization diagnostic, not a localization-pose
benchmark.

The evaluator now also reports Sobel/laplacian edge fidelity. Region masks are
eroded by one pixel so semantic boundaries do not create artificial edges.
These metrics are diagnostics and are not presented as historical benchmark
metrics.

## Capacity experiment: why more splats was not the main fix

The v20 balanced run had 1,121,077 foliage Gaussians. The v21 experiment raised
this to 1,373,077, increased per-event split capacity from 3,000 to 12,000 and
reduced the selected projected-radius median from 10 to 8 pixels.

Full 1,487-view canonical results:

| Run | Static PSNR | Static SSIM | Static MAE | Non-tree PSNR | Tree PSNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| v20 | 14.9085 | 0.6564 | 0.1350 | 15.4049 | 11.3451 |
| v21, +252k foliage | 14.9016 | 0.6559 | 0.1352 | 15.4078 | 11.3303 |

Official 64-query canonical results:

| Run | Static PSNR | Non-tree PSNR | Tree PSNR | Tree gradient cosine |
| --- | ---: | ---: | ---: | ---: |
| v20 | 14.7664 | 14.8845 | 13.4476 | 0.0734 |
| v21, +252k foliage | 14.7680 | 14.8868 | 13.4358 | 0.0749 |

The extra capacity slightly raises edge energy on fitted tree-heavy frames,
but gives no PSNR or query generalization gain. It therefore cannot be the
primary remedy for the smear. The dominant remaining error is correspondence,
canonical ownership and continuous geometry quality, not a global shortage
of Gaussian count.

## v63 ownership-repair result

The ownership repair improved the deployment-relevant global and official
query map, but the old fixed role quota reduced exact-view fit:

| Run | Database canonical static | Database non-tree | Database tree | Official query static | Query tree |
| --- | ---: | ---: | ---: | ---: | ---: |
| v21 | 14.9016 | 15.4078 | 11.3303 | 14.7680 | 13.4358 |
| v63 | 15.0260 | 15.3826 | 11.9119 | 14.8107 | 13.8624 |

On frames 00408--00410, v63's conditioned static/tree PSNR was
13.8342/14.2920.  These frames are a difficult exact-view diagnostic, not a
replacement for the 1,487-view canonical or disjoint query metrics.

## Population-normalized capacity and schedule-preserving handoff

v65 replaced the fixed role quota with a quota proportional to raw eligible
row count.  It was a causal negative result: the initialization contains
895,761 per-view dynamic birth rows, so producer sampling density
self-reinforced dynamic capacity.  Full-database canonical
static/non-tree/tree PSNR was 15.0410/15.3787/11.9672; official query
static/tree was 14.8100/13.8404.  Increasing the dynamic split count raised
target tree gradient alignment but did not improve reconstruction.

v24 therefore normalizes eligible count by each live role population before
capacity apportionment.  Duplicating an evidence producer's rows no longer
changes its role quota.  v25 additionally makes the rigid producer's complete
surface optimizer schedule authoritative.  The aborted v67 2k diagnostic
proved why this matters: despite improved foliage, query static PSNR fell to
14.5312 and frames 00408/00409 lost about 0.9/1.2 dB on non-tree rigid pixels
relative to the untouched handoff.  That run is not a final model.

## Remaining method ceiling

The current 24k no-COLMAP rigid handoff is preserved by the mixed stage, but
its all-database non-tree ceiling is only about 15.78 dB and the v20/v21
canonical mixed result remains far below the historical
18.755777/0.830073/0.082067 static PSNR/SSIM/MAE. The historical model is not
used as initialization.

After the v63 ownership repair, the next high-value architecture work is:

1. build a true descriptor/cycle MASt3R graph across database keyframes;
2. replace the discrete Chart graph with a continuous learnable atlas and
   UV-domain refinement;
3. add verified rigid background completion behind foliage;
4. only then run the final long mixed optimization and pose-estimation
   localization benchmark.

Those are real remaining components, not parameter sweeps.
