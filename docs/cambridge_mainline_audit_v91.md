# Cambridge rigid-base causal audit v91

## Scope

This audit follows the complete active repository on
`codex/hybrid-causal-repair-v2`.  It does not treat the presence of a metadata
field, loss option or CUDA symbol as an end-to-end implementation.  A feature
counts only when evidence production, serialization, training consumption,
topology mutation, rendering and evaluation agree.

The historical exact full-database static result remains:

```text
PSNR 18.755777 / SSIM 0.830073 / MAE 0.082067
```

The strongest completed mixed result before this audit is v85:

```text
database1487 static: 15.643804 / 0.691434 / 0.124037
non-tree:            15.834111
tree:                13.226037
```

The mixed branch improved the v74 rigid tree result by about 4.01 dB while
leaving non-tree quality essentially unchanged.  The measured bottleneck is
therefore the rigid structural map, not absence of a working 3D foliage
renderer.

## Experiments v86--v91

All query results use the same disjoint official query64 split, fixed
calibrated poses, 640x360 raster and canonical/rigid deployment-valid render.
They are not a pose-estimation benchmark and are not directly comparable with
the historical database1487 aggregate.

| Run | Iteration | raw PSNR | static PSNR | non-tree PSNR | tree PSNR | Conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v86 rigid baseline | 8k | 13.0099 | 14.4406 | 14.8945 | 9.3294 | Supported MASt3R/Chart baseline |
| v86 rigid baseline | 16k | 13.3230 | 14.8353 | 15.3546 | 9.8756 | Stable gain |
| v86 rigid baseline | 32k | 13.4564 | 14.9110 | 15.5209 | 9.7407 | Saturated; 24k--32k non-tree +0.026 dB |
| v87 first DAV2 pilot | 16k | 13.2423 | 14.6529 | 15.2155 | 9.8273 | Rejected: reduced Chart budget, global-kNN frames and aliased source role |
| v88 cross-ray DAV2 | 8k | 13.0800 | 14.4405 | 14.9269 | 9.6887 | Local tree/boundary gain, weak aggregate |
| v88 cross-ray DAV2 | 16k | 13.2584 | 14.6751 | 15.1948 | 9.9712 | Rejected: non-tree -0.160 dB vs v86 |
| v89 native DAV2 frame | 4k | 12.7584 | 13.9148 | 14.2469 | 9.4369 | Frame-only change has negligible aggregate effect |
| v89 native DAV2 frame | 8k | 13.0698 | 14.4259 | 14.9286 | 9.5565 | Rejected: topology still destroys evidence lineage |
| v91 camera-sweep lineage | 4k | 12.7440 | 13.8932 | 14.2322 | 9.4145 | First checkpoint; query55 non-tree improves locally |
| v91 camera-sweep lineage | 8k | 13.0718 | 14.4423 | 14.9352 | 9.6143 | Lifecycle repair is correct, but aggregate gain is only +0.0065 dB non-tree vs v89 |
| v92 observation patch 0.15 | 4k | 12.8493 | 14.0444 | 14.3694 | 9.4659 | Exact source observation restores a broad early gain |
| v93 observation patch 0.05 | 8k | 13.1454 | 14.5428 | 15.0321 | 9.7247 | Positive but weaker than 0.15 |
| v92 observation patch 0.15 | 8k | 13.2295 | 14.6844 | 15.1619 | 9.7787 | +0.227 dB non-tree and +0.164 dB tree vs v91 |
| v94 observation patch 0.15 | 12k | 13.4246 | 14.9510 | 15.4718 | 10.1212 | Continued topology gives a real, non-saturated gain |
| v94 observation patch 0.15 | 16k | 13.4966 | 15.0331 | 15.5422 | 10.1977 | Best completed rigid query result in this audit |

v89 proved that tangent-frame correction alone was not enough.  Its first
topology event removed about 43k surfels.  At 4k only 37,205 of the initial
80,000 DAV2 role-4 rows remained, while 92,848 newly created rows had been
relabeled as source role 3.  At 8k the roles were:

```text
source 1: 205,112
source 2:  38,204
source 3: 261,226
source 4:  26,932
```

This exposed two linked contract defects:

1. a posterior DAV2 birth deliberately initialized near the opacity cull
   floor could be deleted before representative camera exposure;
2. every split/clone child overwrote its source provenance with
   `SOURCE_FREE_RESIDUAL`, so the trainer could no longer distinguish a DAV2,
   Chart or MASt3R lineage after one topology event.

## v91 repair

The surface model now persists one continuous `observation_mass` value per
primitive.  It is serialized in checkpoints and PLY, inherited by
split/clone children, pruned atomically with every other row, and incremented
by the same rigid responsibility used for screen-gradient topology.

For a DAV2 lineage, the opacity threshold is:

```text
effective_cull =
    ordinary_min_opacity
    * (1 - exp(-observation_mass / (4 * database_camera_count)))
```

This is not a protected iteration interval or hard quality gate.  Unseen
posterior rows begin with a near-zero cull threshold; after four weighted
database sweeps they receive 63.2% of the ordinary threshold and after twelve
sweeps 95.0%.  Geometrically invalid rows remain subject to the ordinary
world-scale and screen-radius culls at every event.

Split and clone children now:

- clear direct `track_id`, so a child is not falsely treated as a second
  independent observation;
- inherit source provenance, spatial confidence and cumulative observation
  mass;
- remain renderer-owned and replaceable.

The final export uses the same continuous cull.  This closes the previous
train/export mismatch in which a row retained online could be deleted by an
unconditional final alpha cleanup.

At v91's first topology event (iteration 2,600):

```text
total routine prune:       16,388   (v89/v90: about 43,100)
DAV2 opacity prune:             0   (v90 fixed-scale attempt: 27,352)
DAV2 all-cause prune:       3,515
DAV2 lineage after split:  78,651 / 80,000 initial
continuous maturity mean:  0.1865
```

The remaining 3,515 removals were caused by geometric size/radius rules, not
source-based immunity.  The complete test suite after the repair is
`592 passed, 15 warnings`.

The completed v91 8k model contains 583,165 surface primitives:

```text
MASt3R lineage (source 1): 316,610
Chart lineage  (source 2):  97,405
DAV2 lineage   (source 4): 169,150
free/unknown   (source 3):       0
```

This proves that the source-lifecycle repair works end to end.  It does not
prove that the corresponding observation is used.  Inspection of the
immutable v46 seed exposed the next causal break: all 80,000 DAV2 rows had
`pointmap_view_id=-1`, and the seed retained neither source image nor source
pixel.  The model therefore knew that a row came from DAV2, but no longer knew
which real observation had created it.  Random full-image RGB and visibility
mass cannot replace that missing observation identity.

## v47 observation contract

The v47 initializer records, for every DAV2 rigid-hole proposal:

- the exact source-view name;
- its normalized source pixel centre;
- its aligned source camera-z depth.

The v46 and v47 seeds were compared array by array.  Apart from the version
string, all 17 pre-existing arrays are exactly equal, including xyz, RGB,
scale, rotation, normal, opacity and source role.  The only new values are
the three observation arrays.  Their validation is:

```text
surface rows:                 377,556
DAV2 rows:                     80,000
valid source names/UV/depth:   80,000 / 80,000 / 80,000
unique real source views:         484
COLMAP geometry:               false
historical Gaussian input:     false
```

Training now samples a 5x5 real-RGB patch at these exact source pixels when
their source camera is selected.  The factor is independent of current
primitive count and topology indexing, so split/prune events cannot silently
invalidate its target.  v92 is the controlled experiment with the same seed
geometry, random seed, schedule and capacity as v91; only this observation
factor changes.

The controlled results show that the factor is causal rather than merely
present.  At 8k, v92 improves 49/64 query views in non-tree PSNR and 54/64 in
static PSNR.  The gain is largest on the weakly covered held-out trajectories,
not concentrated in one outlier.  Reducing the factor weight from 0.15 to
0.05 reduces all aggregate PSNR gains, so the retained 0.15 result is not an
over-strong endpoint selected by a hard acceptance gate.

Continuing the exact v92 prefix to v94-16k grows the surface from 584,743 to
770,742 rows without reaching the 800k budget:

```text
MASt3R lineage (source 1): 364,779
Chart lineage  (source 2): 136,377
DAV2 lineage   (source 4): 269,586
free/unknown   (source 3):       0
```

The simultaneous non-tree, tree and boundary improvements rule out the
previous claim that training length cannot help.  They also rule out the
opposite claim that this was only under-training: query55 still contains a
point-sampled rear facade after 269k live DAV2 descendants.  More rows and
more iterations improve fidelity, but cannot turn discrete hypotheses into
the missing continuous cross-view surface atlas.

On the complete 1,487-view database, the historical-uint8 protocol gives:

| Rigid run | Iteration | static PSNR | non-tree PSNR | tree PSNR |
| --- | ---: | ---: | ---: | ---: |
| v86 old rigid control | 24k | 14.8042 | 16.0975 | 9.1328 |
| v92 observation repair | 8k | 14.4434 | 15.4468 | 9.3933 |
| v94 observation repair | 16k | 14.9983 | 16.1966 | 9.6293 |

Thus v94 beats the old v86-24k rigid result with 8k fewer iterations:
`+0.1941 dB` static, `+0.0991 dB` non-tree and `+0.4966 dB` tree.  This is a
full-database result, while the query64 table above measures held-out-camera
generalization.

## Mixed handoff v95

v94 writes a validated `native-rigid-surface-handoff-v1`; the complete mixed
teacher starts from that handoff with structural xyz/scale/rotation/opacity
preserved.  Static skeleton, canonical crown, sequence/time-conditioned
leaves, ray/depth posterior, adaptive volume split and local surface
replace/retire are active from their declared schedule rather than deferred
to an unreachable late phase.

At the first 3k mixed checkpoint, query64 changes from v94 rigid to canonical
mixed as follows:

```text
                    rigid16k    mixed3k    delta
raw PSNR             13.4966     13.5988   +0.1023
static PSNR          15.0331     15.1811   +0.1480
non-tree PSNR        15.5422     15.2761   -0.2661
tree PSNR            10.1977     12.8698   +2.6722
tree-boundary-in     10.6709     12.1954   +1.5245
```

This proves that the 3D branch fills the missing tree rather than merely
changing a flag.  It also exposes the next active error: before sufficient
free-space/ownership accumulation, canonical crown produces false occlusion
on some rigid pixels.  The 3k checkpoint is therefore an early diagnostic,
not the selected final teacher.

The complete v95 run was then allowed to reach every retained checkpoint.
The official disjoint query64 canonical result is:

| Iteration | raw PSNR | static PSNR | non-tree PSNR | tree PSNR | boundary-in PSNR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3k | 13.5988 | 15.1811 | 15.2761 | 12.8698 | 12.1954 |
| 6k | 13.7545 | 15.4187 | 15.4643 | 13.7303 | 12.8884 |
| 9k | 13.8671 | 15.6343 | 15.6594 | 14.1456 | 13.2844 |
| 12k | 13.9180 | 15.6477 | 15.6642 | 14.3326 | 13.4305 |

All primary query regions improve monotonically.  From the v94 rigid handoff
to v95-12k, the changes are `+0.6144 dB` raw, `+0.6146 dB` static,
`+0.1220 dB` non-tree and `+4.1350 dB` tree.  The simultaneous non-tree gain
rules out the claim that the tree improvement was obtained by globally
damaging the rigid branch.

Database views 408--410 are dominated by large, high-frequency conifer
instances and provide a deliberately difficult local diagnostic.  Their
conditioned result progresses as follows:

| Iteration | static PSNR | non-tree PSNR | tree PSNR | tree SSIM | tree MAE | tree edge-energy ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3k | 13.8680 | 12.3369 | 14.1689 | 0.5522 | 0.1457 | 0.4150 |
| 6k | 14.8867 | 12.6665 | 15.3086 | 0.6558 | 0.1229 | 0.5686 |
| 9k | 15.3216 | 12.7805 | 15.8202 | 0.6973 | 0.1139 | 0.6336 |
| 12k | 15.6765 | 12.9912 | 16.2097 | 0.7222 | 0.1083 | 0.6586 |

The old v85 final conditioned tree result on the same three frames was
`15.7258 / 0.6906 / 0.1144`.  Thus v95 improves the difficult tree region by
`+0.4839 dB` while also improving SSIM and MAE.  Visual inspection still
shows overly broad, overlapping leaf clusters: the measured improvement is
real, but it does not constitute a complete solution to foliage smearing.

The topology trace also rejects the previous over-densification and hard
retirement failure modes:

- surface count remains fixed at the validated 770,742-row rigid handoff;
- volume capacity reaches 2,000,000 and then performs role-matched,
  lineage-safe retire-and-split instead of unbounded growth;
- at 8.1k the mean local surface retirement fraction is about 4.98%, with
  only 55 fully retired rows;
- after 10k, volume topology stops and the remaining iterations perform
  ownership cleanup and canonical appearance polish.

The remaining causal limitation is visible in the evidence trace.  The
ray-interval table contains 2,276,243 rays from 256 selected cameras, but
ownerless hit rows are conservatively excluded rather than treated as
positive evidence.  At completion the interval factor had covered about
51.7% of volume rows, while 1,015,415 sampled ownerless hits had produced zero
verified ownerless positives.  This prevents false geometry, but it also
leaves cross-sequence canopy coverage underconstrained.  Together with the
discrete rigid Chart samples, it explains why more split capacity improves
metrics yet does not recover a continuous high-frequency facade or leaf
atlas.

The exact full 1,487-view historical-uint8 evaluation is:

| Run/mode | static PSNR | static SSIM | static MAE | non-tree PSNR | tree PSNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| historical reference | 18.7558 | 0.8301 | 0.08207 | n/a | n/a |
| v85 canonical | 15.6438 | 0.6914 | 0.12404 | 15.8341 | 13.2260 |
| v94 rigid | 14.9983 | 0.6387 | 0.14668 | 16.1966 | 9.6293 |
| v95 canonical | 15.9945 | 0.7070 | 0.12105 | 16.2323 | 13.2231 |
| v95 conditioned | 16.2630 | 0.7200 | 0.11536 | 16.2668 | 14.4375 |

The canonical result is exactly comparable with the historical database
protocol.  Against v85 it gains `+0.3507 dB` static and `+0.3982 dB`
non-tree while preserving tree PSNR within `0.003 dB`.  Against its own v94
rigid handoff it gains `+0.9963 dB` static and `+3.5938 dB` tree.  The
conditioned result is a valid known-database-view diagnostic but is not
historically comparable because it consumes sequence/time identity.

Final observation-aware cleanup removes 28,491 transparent surface rows and
134 fully retired rows, leaving 742,251 surface primitives.  It retains all
1,004 surviving direct MASt3R witnesses, 136,370 Chart-lineage rows and
241,223 DAV2-lineage rows.  The authoritative mixed teacher contains
2,000,000 volume Gaussians:

```text
static skeleton:       1,012
canonical crown:     531,095
dynamic leaf:      1,467,893
```

The historical `18.7558` result remains `2.7612 dB` ahead.  The result is
therefore a measured recovery and a new strongest canonical model in this
contract-closed series, not evidence that the complete ideal architecture has
already been implemented.

## What is and is not complete

Complete in the active chain:

- exact calibrated cameras including principal point;
- no COLMAP point/track geometry and no historical trained Gaussian
  initialization;
- immutable MASt3R/MAtCha/Chart/DAV2/G4 evidence contract;
- persistent source, role, track, instance and optimizer/topology metadata;
- observation-level surface factors and native-resolution evidence factors;
- selected-view real ray/depth foliage posterior;
- native same-tile 2D surfel + 3D EWA forward/backward renderer;
- adaptive volume split/prune with optimizer migration;
- local replace-and-retire and spatial uncertainty.

Still partial or missing:

- learnable continuous inverse-depth Chart atlas with UV quadtree topology;
- full-database reciprocal descriptor union-find MASt3R graph;
- dense all-view visual hull/visibility posterior;
- strong semantic trunk/branch reconstruction;
- one shared canonical foliage set deformed by sequence/time;
- exact per-pixel mixed depth order for large oblique surfels;
- high-resolution residual patch sampler/facade atlas;
- renderer-equivalent standard unified 2DGS/3DGS export;
- Cambridge pose-localization benchmark.

Consequently, it remains incorrect to claim that all historical external
recommendations are implemented.  The current experiment repairs a measured
rigid evidence-lifecycle failure; its final 8k image metrics determine whether
that repair should be retained before any foliage stage is attached.
