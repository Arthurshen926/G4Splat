# Cambridge unified mainline v96 causal repair audit

Date: 2026-07-30

Branch: `codex/hybrid-causal-repair-v2`

Baseline inspected:

`StMarysChurch_v35_tree408_evidence_repair/teacher_hybrid12k_v95_observation_patch_v47_handoff_v94`

## What the current implementation already did before this repair

Several claims in the external v95 review describe older code, not the current
branch:

- The production foliage posterior already integrates anisotropic Gaussian
  optical mass analytically over calibrated free/hit intervals. The legacy
  expected-depth diagnostic exists but is not called by the trainer.
- Ownerless dense tree hits already produce 895,761 sequence-local dynamic
  births during role-aware initialization. They are intentionally not
  self-promoted into cross-sequence canonical evidence.
- The rigid topology pass already renders volume at zero opacity, keeps the
  surface trainable from row zero, detaches the sky, and weights only
  `w_plane`.
- Volume split selection is already balanced by owner-camera context, tree
  instance and spatial cell; role quotas are evidence-adaptive rather than a
  fixed 15/45/40 split.

These paths were retained.

## Repaired defects

### Chart UV endpoint lookup

The UV-edge sampler was independent, but endpoint lookup used only the current
anchor minibatch. It now resolves endpoints from every live direct Chart
witness and selects the nearest live primitive for each evidence row.

The runtime audit now distinguishes sampled and actually matched UV edges and
persists cumulative edge activity in `result.json`.

Real v95 surface replay:

- surface primitives: 742,251
- live direct Chart rows: 5,493
- sampled UV edges: 4,096
- matched UV edges: 1,432

### Foliage ray evidence epochs

Each camera now owns an exact shuffled `EvidenceEpochSampler` with permutation,
cursor, visit counts, completed epoch and RNG state. The state is checkpointed
with a v3 schema; v2 checkpoints migrate without discarding their known
coverage.

The outer camera scheduler now advances the camera with the least normalized
row coverage. This fixes the former equal-camera schedule, which repeatedly
sampled small tables while cameras with up to about 30k valid rays remained
incomplete.

The reported coverage denominator now contains only effective free/hit rows,
not explicit unknown rows.

`--allow-trainer-repair-resume` accepts this Python-only trainer/evidence/Chart
repair set, migrates the old lifecycle contract explicitly and restores the v2
ray coverage mask into v3 per-camera epochs. Model tensors, Adam state and CUDA
kernels are unchanged by that migration.

v95 corrected audit:

- all stored rows: 2,276,243
- explicit unknown rows: 502,262
- effective free/hit rows: 1,773,981
- effective rows already visited: 1,176,287
- corrected coverage: 66.31% (the old report was 51.68%)

From the same v95 continuation state, the old uniform-camera policy needs
about 48,128 training steps at `ray_posterior_every=4` to finish the largest
camera table. Coverage-balanced scheduling needs about 4,880 equivalent
steps, a 9.86x reduction.

### Contradiction prune

An observation identity is now provenance, not permanent existence proof.

- strong evidence (very low occupancy, or at least two confirmed-free views
  with contradiction fraction at least 0.5) can retire an observed row;
- a weak conflict is protected only by independent positive support from at
  least two views and two sequences;
- weak opacity is measured from the current lower opacity tail instead of the
  unreachable fixed `<0.002` cutoff;
- trunk/static-skeleton rows remain protected by their separate physical
  ownership contract;
- every category is explicitly audited per topology event.

Exact v95 final-checkpoint topology replay with splitting disabled:

- old volume count: 2,000,000
- contradiction-pruned: 3,945
- remaining: 1,996,055
- static skeleton removed: 0
- dynamic leaf removed: 0
- canonical crown removed: 3,945
- weak positive rows protected: 20

This is selective cleanup, not a global density collapse.

## Representation work completed in v96

### Continuous Chart atlas and UV topology

`ChartSurfaceModel` now owns a bounded coarse/fine inverse-depth residual
pyramid over all 56 source Charts. Every live Chart cell is unprojected with
the exact fixed Cambridge K and pose on every render. Its normal comes from
the depth Jacobian and its two tangent scales follow the metric source-UV
footprint at the currently learned depth.

Chart geometry is substituted before the one native mixed CUDA call; 2D
surfels and 3D EWA volumes therefore still share one tile list, one depth sort
and one transmittance accumulation.

UV topology is now a real parent-replacing quadtree. Error, screen radius and
confidence rank cells; one parent is retired and replaced by four half-size
children while integrated area is preserved. Ordinary world-space
clone/split explicitly excludes these UV-owned rows.

A forced GPU smoke test replaced 10 parents with 40 children (net +30), while
the mature free surface remained frozen. Both coarse and fine inverse-depth
fields received non-zero gradients. A 600-step real-scene prefix exercised
29.7 million bound-row renders and every Chart anchor/edge epoch reached full
coverage. Because its 12k schedule releases topology at iteration 961, this
prefix intentionally contained no split event; it is a forward/gradient
validation, not a quality result.

The final PLY bakes atlas centres, Jacobian normals and UV-derived scales into
ordinary 2DGS tensors. Exact staged continuation additionally saves a
content-addressed `chart_surface_state.pth`; otherwise UV children would
silently lose atlas ownership at the rigid-to-hybrid handoff.

### Rigid high-frequency completion

The training objective now selects a detached local pool of high-error,
high-target-gradient rigid patches (facade/window/stone/small static object
and the rigid side of tree-building boundaries). Their differentiable RGB,
SSIM and gradient loss prevents small rigid regions from being averaged away
by the full-frame objective. Independently supported MASt3R/Chart/DAV2 rigid
coverage births remain available through the bounded rigid-completion
contract.

### Shared canonical/conditioned foliage state

Sequence-local leaves retain their exact observation identity and local
birth/death probability, but no longer form a disconnected geometry owner.
Rows assigned to a replacement group inherit the learned displacement of
their shared canonical primitive before the low-rank sequence/time residual
is applied. The dynamic branch therefore cannot leave an old canonical copy
behind merely because the canonical row moved.

The `initialization_center` tensor previously aliased the trainable `xyz`
storage, making the intended geometry-deviation floor a no-op. It is now an
immutable detached clone and is covered by regression tests.

### Database-scale reciprocal descriptor track producer

`scripts/build_mast3r_descriptor_track_graph.py` adds the missing MASt3R
reciprocal-descriptor producer: calibrated keyframe selection across all
sequences, temporal and cross-sequence pairing, reciprocal NN matches,
exact-K triangulation/reprojection/semantic checks, observation-node
union-find, robust component triangulation and v6 observation-level export.
The archive declares its correspondence builder and camera scope so the
runtime audit is data-driven rather than hard-coded.

The currently immutable v95 Evidence Store still contains the older
56-camera projection-star graph. The new producer must be used to build a new
content-addressed Evidence Store and matching initialization before its
tracks can affect a quality run; silently swapping archives would invalidate
track ids and the initialization hash.

## Remaining validation boundary

The strict analytic free/hit/behind interval mass and per-camera evidence
epochs, ownerless observation-space births, contradiction-aware prune,
tree/context/spatial adaptive volume split, shared canonical displacement,
spatial uncertainty, UV atlas and local rigid patch path are implemented.

The next quality claim must come from two ordered runs:

1. rigid-only atlas training long enough to pass the release point and execute
   repeated UV splits;
2. mixed training from that atlas-aware handoff, with canonical foliage and
   sequence/time leaves enabled.

The historical v94 free-XYZ Chart positions are deliberately not treated as
geometry truth. Consequently a 600-step prefix that is still inside the
12k-step bootstrap can be below v94 RGB metrics; using it as the final
comparison would test initialization transition, not atlas convergence.

## v24/v99 follow-up causal audit

The first 1k local-ownership replay isolated one variable while retaining the
old 127-keyframe Evidence Store. It completed without NaN or capacity
explosion (507,143 surface and 1,777,073 volume primitives), but did **not**
produce a quality gain by itself:

- conditioned tree-static: 12.4231 -> 12.2763 dB;
- conditioned non-tree-static: 13.0707 -> 13.1024 dB;
- conditioned tree-boundary-inside: 10.7060 -> 10.7515 dB.

The result falsifies the claim that non-local replacement ownership was the
only blur source. It slightly reduces rigid-side contamination, while the
incomplete canonical crown/track topology remains dominant.

The replay also exposed four additional implementation defects:

1. a sparse chain could join two density-supported crowns into one tree
   instance even when the bounded extent check passed;
2. replacement groups allowed a small number of cross-instance links, despite
   tree instance being part of the deformation/lifecycle contract;
3. the rigid handoff had recursively subdivided Chart cells to level 10 even
   though the 128x72 atlas and 640x360 RGB target provide only about three
   observable binary refinement levels;
4. a 12k / every-4 / 512-ray schedule has capacity for 1.536M rows, less than
   the 1.774M effective interval table, so `never_visited == 0` was
   mathematically impossible.

The corrected v99 contract therefore uses density-supported core connectivity,
same-instance local replacement, a maximum UV level of three, and a
prefix-stable ray batch lower bound derived from the final schedule horizon.
The reciprocal descriptor producer additionally admits cross-sequence canopy
components only after at least three fixed-camera observations and a stricter
joint reprojection residual. This supplies a real path for persistent
trunk/branch evidence without promoting a two-view moving-leaf coincidence.

The descriptor producer snapshots its own implementation hash before
inference. This closes a provenance race in which editing a long-running
producer could otherwise label old in-memory behavior with the new file hash.

The from-zero 256-keyframe replay produced 770 cross-sequence canopy tracks
that already satisfy the three-observation and strict reprojection contract.
The original trunk classifier nevertheless retained only seven of them:
its 0.20 m / four-neighbour local cylinder test was too sparse for the new
descriptor graph. A real-seed sensitivity replay showed that a 0.30 m,
three-neighbour, 0.06 m radial test retains 134 tracks across eight tree
instances while preserving both independent constraints (at least three
fixed-camera observations per track and at least three same-instance spatial
samples). Initialization v49 adopts this density-supported skeleton contract;
it does not promote every stable canopy descriptor to a trunk.

The follow-up audit found one additional ownerless-ray consumer defect.
Although dense observation-space hits already created exact-camera dynamic
births, the analytic interval factor still applied a blanket `seed_bound`
mask. It therefore counted those rays as visited while excluding their hit
and free-space loss. The interval factor now has explicit physical owners:

- cross-sequence seed-bound hits consume canonical crown mass only;
- ownerless hits consume exact-camera dynamic births only;
- confirmed-free rays constrain canonical plus exact-camera dynamic mass.

An ownerless observation still cannot promote itself into canonical evidence,
and a leaf owned by another camera receives no gradient. This converts the
large dense-ray suffix from initialization-only geometry into an exercised
training posterior without creating a globally visible dynamic fog layer.

Two further runtime audits were added before the v99 quality replay:

1. the old `ownerless_dynamic_factor_rows` counter equalled every sampled
   ownerless hit, even when there was no exact-camera dynamic optical mass in
   its hit interval. It now counts only rows with non-zero interval mass and
   separately reports `ownerless_missing_dynamic_hit_rows`;
2. `appearance_only` froze native surface xyz/scale/rotation, but the external
   Chart inverse-depth optimizer and UV quadtree remained active. Since Chart
   rows obtain their rendered geometry from that external provider, the
   claimed rigid handoff was not actually frozen. Non-joint mature policies
   now clear atlas gradients (so resumed Adam momentum cannot move them), skip
   the atlas optimizer step, and disable UV topology.

The corrected v99 mixed replay uses the completed 12k rigid atlas handoff
(539,488 exported 2D surfels, including 72,265 baked live Chart rows). At its
500-step mixed checkpoint, the rigid handoff and mixed checkpoint coarse/fine
inverse-depth tensors were bit-identical, and all 200,218 UV evidence ids were
unchanged. This is parameter-level preservation, not a flag-only audit.

The same live replay establishes that the dense observation-space posterior
is physically exercised:

- iteration 100: 560/560 ownerless hits had exact-camera dynamic hit mass;
- iteration 200: 545/545;
- iteration 300: 595/595;
- cumulative through iteration 1,000: 163,151/163,151, with zero missing
  dynamic hit rows and 8.76% effective ray-epoch coverage.

Volume topology ramped from +224 rows at iteration 300 to the bounded
20k/event rate. It first reached the 2M budget at iteration 2,200 by retiring
13,746 redundant descendants before replacement. The exchange was role
matched (5,669 canonical and 8,077 dynamic), selected 2,894
instance/owner/spatial groups, and did not retire skeleton rows. The following
two events remained exactly within budget while replacing 19,828 and 19,826
local descendants respectively; no capacity overflow or global fallback was
observed.

## Verification

- repository suite (`pytest tests`): 627 passed;
- real v95 Chart-surface factor replay completed on GPU 2;
- exact v95 volume checkpoint replay completed on CPU;
- native mixed-render atlas gradient smoke completed on GPU 2;
- UV parent/child replace-and-retire smoke completed on GPU 2;
- real-scene 600-step rigid atlas prefix completed on GPU 2;
- `py_compile` and `git diff --check` passed.

The top-level bare `pytest` command also discovers vendored pybind11 upstream
tests under `2d-gaussian-splatting/submodules`; those require the unrelated
`pybind11_tests` extension. Project validation therefore uses `pytest tests`.

## v100 full-database result and v51 frequency-residual repair

The compact-ray-basis v100 replay completed 12,000 mixed iterations and an
exact historical-raster evaluation over all 1,487 database views. It retained
the complete 1,521,922-row observation-space posterior while reducing dense
dynamic renderer births from 1,521,922 to 260,908. All 256 factor cameras
completed at least one ray epoch; no NaN, capacity overflow, camera-contract
drift, historical RGB Gaussian initialization or COLMAP point-cloud input was
observed.

Against the immediately preceding v99 model, exact uint8 protocol metrics
changed as follows:

| scope | v99 | v100 | delta |
| --- | ---: | ---: | ---: |
| canonical raw PSNR | 14.7243 | 14.7885 | +0.0642 |
| canonical tree-static PSNR | 14.4483 | 15.1856 | +0.7373 |
| conditioned raw PSNR | 14.9652 | 14.9878 | +0.0227 |
| conditioned static-valid PSNR | 17.0031 | 17.0398 | +0.0367 |
| conditioned tree-static PSNR | 15.6810 | 16.2165 | +0.5355 |
| conditioned tree-boundary-inside PSNR | 14.3005 | 14.5198 | +0.2193 |
| conditioned tree-boundary-outside PSNR | 13.6746 | 13.5454 | -0.1292 |
| conditioned non-tree-static PSNR | 16.8958 | 16.8838 | -0.0120 |

This is a real geometry/coverage gain, but not a high-frequency solution.
Layer diagnostics on 00408--00410 show that v100 raises mean volume alpha by
roughly 0.04--0.05 while reducing alpha-edge and RGB-Laplacian energy. The
conditioned and canonical volume-alpha maps are also nearly identical. At the
same time, the topology allocator reaches two million rows with selected
parent radii still around 7 px median / 15 px p90; the median selected
generation is only one. The evidence and split paths are active, but much of
their capacity remains a smooth opacity blanket.

The remaining cause is not another missing loss gate. A 1,024-row
owner-camera RGB basis was selected by uniform silhouette area and each row
was enlarged to represent its share of that area. The full ray posterior
therefore retained geometry supervision, but most owner-view RGB frequencies
were discarded. Subsequent splits duplicate the parent's colour and cannot
recreate those observations.

Initialization v51 changes the sequence-local branch into a true residual
basis over the shared canonical crown:

- the immutable 1,521,922 ray rows, 260,908 dense births and all global model
  budgets remain unchanged;
- per-instance slots are half uniform coverage and half spatially distributed
  owner-view gradient maxima;
- a dynamic witness is capped at about 0.9 px in the 640-wide reference
  renderer instead of being enlarged to tile the whole silhouette;
- the canonical crown continues to own continuous low-frequency coverage and
  local optical-mass replacement still prevents additive duplicate alpha.

On the real initialization, the dynamic maximum-axis median changes from
0.114 m to 0.035 m and p90 from 0.20 m to 0.10 m without increasing row
count. This directly tests whether the v100 tree gain can be converted from
coverage into spatial frequency while reducing rigid-side boundary spill.

The v51 implementation passes the complete project suite:

- repository suite (`pytest tests`): 628 passed;
- `py_compile` and `git diff --check`: passed.

## v103/v104 frequency-basis causal replay

The first frequency-residual replay (v103) exposed a bandwidth coupling bug
in v51 rather than validating the intended two-scale representation.  The
renderer-basis selector correctly reserved half of each tree-instance quota
for spatially distributed RGB-gradient maxima, but the same roughly 0.9 px
footprint ceiling was applied to both those residual witnesses and the
uniform coverage witnesses.  The result increased edge energy without
preserving optical coverage: high-frequency points became sparse speckles
over the still-smooth canonical crown.

Initialization v52 separates those roles without changing the experiment's
geometry or cardinality:

- the immutable posterior remains 1,521,922 observation-space rows;
- the dynamic renderer basis remains 260,908 rows;
- centre, colour, role, track, tree-instance and owner-camera arrays are
  bit-identical to v51;
- 129,758 uniform coverage rows use a bounded roughly 2.4 px reference
  footprint, while gradient-selected residual rows retain the roughly 0.9 px
  footprint;
- the changed-row maximum-axis median rises from 0.035 m to 0.0571 m, while
  the frequency-residual branch is unchanged.

The controlled v104 12k replay completed without NaN, OOM, capacity overflow
or mixed-render contract drift.  It finished with 538,886 surface primitives
and exactly 2,000,000 volume primitives.  Compared with v103, exact interval
consumer coverage improved:

| cumulative ownerless-ray audit | v103 | v104 |
| --- | ---: | ---: |
| sampled hits | 1,612,536 | 1,612,536 |
| canonical factor rows | 1,079,333 | 1,078,745 |
| exact-camera dynamic factor rows | 1,225,592 | 1,323,292 |
| supported hit rows | 1,494,716 | 1,533,891 |
| missing supported hit rows | 117,820 | 78,645 |

Thus the dual-band change removes 33.3% of the missing-consumer events while
holding the posterior, global model budgets and rigid handoff fixed.
Targeted final-iteration conditioned PSNR changes on 00408--00410 are
11.4291/11.4430/10.8142 dB to 11.5116/11.4919/10.8787 dB.  This is a real but
small recovery; the images still contain a smooth dark canopy layer plus
sparse bright residual points.

The exact historical-uint8 evaluation over all 1,487 database views confirms
the same conclusion:

| conditioned scope | v100 | v103 | v104 | v104 - v103 | v104 - v100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw PSNR | 14.9953 | 14.9412 | 14.9565 | +0.0154 | -0.0388 |
| static-valid PSNR | 17.0476 | 16.9674 | 16.9887 | +0.0213 | -0.0589 |
| non-tree-static PSNR | 16.8898 | 16.8840 | 16.8870 | +0.0030 | -0.0028 |
| tree-static PSNR | 16.2247 | 15.8615 | 15.9789 | +0.1174 | -0.2459 |
| boundary-inside PSNR | 14.5297 | 14.3143 | 14.4084 | +0.0942 | -0.1213 |
| boundary-outside PSNR | 13.5312 | 13.4746 | 13.5005 | +0.0259 | -0.0307 |

v104 conditioned tree-static SSIM/MAE are 0.5371/0.1212, compared with
0.5292/0.1227 in v103 and 0.5522/0.1173 in v100.  Tree gradient cosine
recovers from 0.3062 to 0.3098, while edge-energy ratio drops from the
spurious 0.8817 to 0.8723; v100 remains better aligned at 0.3116 cosine with
only 0.8117 edge-energy ratio.  The dual-band repair therefore recovers
coverage and some aligned structure, but does not yet beat the compact v100
basis or solve the exact-view gap.

## v57 all-fixed-camera bounded-ray/depth contract

The v104 replay exposed a more fundamental coverage error in the experimental
contract: database views 00408, 00409 and 00410 are absent from the 256-camera
dense posterior selection.  They therefore have no exact-camera ray interval
or local RGB birth at all.  Their conditioned result is forced through
canonical mass and sequence/time fallback, so changing the footprint of the
selected-camera basis cannot resolve their blur.

Pipeline v29 and initialization v57 replace that keyframe-only contract with
all fixed database cameras while bounding both finite factor bases:

- `selected_foliage_views=0` means all calibrated database cameras;
- cameras without measured tree support receive zero rows;
- total dense-ray allocation remains capped at 1,572,864;
- each supported camera receives a bounded 512-row floor and 2,048-row cap;
- canonical candidate-bound rows use a separate 2,048-row per-camera
  positive/free/unknown and image-space-stratified cap;
- the finite renderer basis is capped at 256 dynamic births per camera;
- allocation uses square-root tree-pixel support so one foreground crown
  cannot consume the global budget.

This holds the ownerless observation-space posterior budget and 2M volume
limit fixed.  Canonical candidate-bound verification rows may increase
because more fixed cameras can independently verify or contradict an
existing visual-hull cell; that growth is audited separately and must not be
described as part of the fixed dense-ray budget.  The production initializer
persists per-view eligible, allocated, accepted-ray and dynamic-birth counts
so a nominal "all-view" run cannot silently collapse back to a small
keyframe subset.

The first real v54 initialization audit correctly rejected its own result:
the CLI sentinel reached `build_foliage_seed`, but an inner
`max(24, selected_view_count)` converted zero back to 24.  That artifact has
only 24 selected cameras, 46,938 dense rows and 6,144 dynamic births, so it is
not a valid all-view experiment and is not passed to training.  v55 resolves
the sentinel at the actual view-pool layer.

The subsequent real v55 audit caught a second selection layer that still
violated the contract.  Its view pool contained all 1,487 records, but
`instance_balanced_diverse_views()` deliberately filled only from camera ids
that already owned sparse track support.  The saved result therefore retained
157 cameras, 310,378 dense rows and 40,192 dynamic births; 00408--00410 were
still absent.  v55 is also rejected before training.

v56 bypasses subset/diversity selection entirely for the all-fixed contract
and preserves the full camera list in deterministic image-id order.  The
bounded mode still uses the instance-balanced selector.  The regression test
now exercises both the view-pool and final-selection layers and requires all
fixed records to survive even when only three ids have sparse support.

Before v56 completed, a producer audit found the same zero-sentinel defect in
the per-frame depth source: `max(72, selected_view_count * 2)` restricted
DAV2 to 72 cameras.  Preserving an RGB camera without a metric depth anchor
does not create a real ray posterior, and 00408--00410 are outside the 157
sparse/Chart observation cameras.  v56 was therefore stopped before training.

v57 preserves all 1,487 already-produced DAV2 depth maps when the posterior
uses the all-fixed contract; bounded profiles retain the former
sequence-balanced DAV2 subset.  Its regression test independently checks the
DAV2 selection layer rather than assuming the RGB/posterior camera fix also
reaches depth production.

The real v57 producer audit then found that selection was still not the last
possible truncation point. `_dav2_foliage_samples()` retained a legacy
120,000-row global limit by breaking the sorted image-id loop as soon as that
limit was reached. With 128 proposals per accepted view, the nominal
all-fixed run could therefore preserve only an early camera prefix. The
non-Chart MASt3R pointmap foliage producer had the same archive-order defect
at its 48,000-row cap.

Initialization v60 removes both early exits. Each producer now processes all
contracted cameras first and then water-fills its fixed global row budget
across accepted camera ids. A hard contract rejects any budget that cannot
retain at least one observation for every accepted camera; the DAV2 audit
also requires visited views to equal the complete fixed-camera selection and
retained sample-camera count to equal accepted depth-camera count. Partial
v57--v59 artifacts were stopped before training and are not quality results.

At the time of this audit update, the corrected real v60 initialization is
running. Quality claims remain attached to completed v104 artifacts until the
v60 initialization, controlled 12k replay and full-database evaluation
finish.

Current repository validation:

- `CUDA_VISIBLE_DEVICES=2 pytest -q tests`: 630 passed;
- `py_compile`: passed;
- `git diff --check`: passed.

## v60--v108 strict-interval and camera-coverage causal result

The completed v60 producer exposed a more serious geometry contradiction than
the camera-selection defects above.  Every one of its 1,682,966 positive hit
rows had `free_end > hit_start`: the pre-hit free-space factor and positive hit
factor required incompatible mass on the same ray.  This happened because the
free endpoint was defined as `depth - 0.25 m`, independently of the lower
posterior bound `depth - 2.5 sigma`.  It was therefore not a renderer or
densification failure.

The producer and consumer now enforce the explicit disjoint contract

```text
0 <= free_end <= hit_start < hit_end
```

for every positive row.  The producer intersects the free endpoint with the
lower hit bound and refuses to persist any subsequently repaired row.  The
runtime loader independently rejects overlap, non-finite endpoints or an
invalid hit interval, so an old contradictory artifact cannot silently enter
training.  The same audit also fixed the single float32 bottom/right endpoint
rounding event without accepting coordinates genuinely outside the declared
source raster.

Loading the complete all-camera posterior initially appeared to hang.  This
was a separate implementation bug: the evidence constructor scanned the full
ray table once for every camera, an `O(N*C)` operation.  One composite
camera/row sort now builds all epoch groups in `O(N log N)`; loading the full
table takes about 8.4 seconds.

The first exact-interval replay, v108, additionally gave 36 cameras that had
failed their own DAV2 metric-depth fit a frame-number-interpolated scale and
offset, then used those inferred depths for exact-camera births.  It completed
12,000 iterations without NaN, OOM, capacity overflow or missing supported
dynamic consumers.  Exact historical-uint8 metrics over all 1,487 database
views were:

| conditioned scope | v100 | v104 | v108 | v108 - v100 |
| --- | ---: | ---: | ---: | ---: |
| raw PSNR | 14.9953 | 14.9565 | 14.9074 | -0.0879 |
| static-valid PSNR | 17.0476 | 16.9887 | 16.9099 | -0.1377 |
| non-tree-static PSNR | 16.8898 | 16.8870 | 16.8335 | -0.0562 |
| tree-static PSNR | 16.2247 | 15.9789 | 16.2722 | +0.0475 |
| boundary-inside PSNR | 14.5297 | 14.4084 | 14.7346 | +0.2048 |
| boundary-outside PSNR | 13.5312 | 13.5005 | 13.3212 | -0.2100 |

This is a useful negative result.  Tree coverage improved, but the
high-frequency tree gradient cosine fell from 0.3116 to 0.3087 and gradient
MAE rose from 0.0352 to 0.0358.  The gain was a darker continuous coverage
layer, not aligned leaf detail.  On the 36 augmented cameras, v108 regressed by
0.4535 dB on average and only two views improved; the other 724 tree views
improved by 0.0724 dB on average.  Views 00408--00410 all belong to the bad
augmented subset and fell by roughly 0.8--0.9 dB.

The causal conclusion is strict: all-fixed-camera training coverage is useful,
but a rejected per-view metric fit cannot be replaced by temporal interpolation
and called a measured ray/depth posterior.  The optional temporal augmenter
remains available as a diagnostic tool, but the production profile no longer
enables it.

## v66 measured all-camera production contract

Initialization v66 is the corrected production input.  It uses only accepted
per-camera metric measurements while retaining every useful fixed-camera
factor:

| quantity | v66 |
| --- | ---: |
| selected fixed cameras | 1,487 |
| DAV2 accepted / rejected cameras | 703 / 784 |
| cameras with persisted ray factors | 714 |
| initial volume primitives | 501,983 |
| observation-space rays | 1,405,193 |
| all persisted ray rows | 3,659,963 |
| positive hit rows | 2,272,564 |
| confirmed-free rows | 545,882 |
| explicit-unknown rows | 841,517 |
| exact-view dynamic births | 273,558 |
| overlapping positive intervals | 0 |

Candidate-bound evidence is capped independently at 8,192 rows per camera:
7,614,542 rows were available and 3,702,479 survived the deterministic
type/image-space basis before global lineage deduplication.  Dense
observation-space support remains 1,405,193 rows rather than being thinned to
make room for additional cameras.  This separates three quantities that had
previously been conflated: immutable ray evidence, finite factor rows and
renderer primitive births.

The corresponding v110 12k controlled replay uses the same rigid warm start,
surface/volume limits and optimization schedule as v108.  Its full result is
reported only after training and the 1,487-view evaluation complete.

Current implementation validation after the v66 contract changes:

- repository suite (`pytest tests`): 643 passed;
- `py_compile`: passed;
- `git diff --check`: passed.

## v110 intermediate result: geometry coverage versus RGB coverage

The v110 controlled replay confirms that the v66 measured posterior is
physically consumed.  At matched milestones:

| audit | v108 6k | v110 6k | v108 10k | v110 10k |
| --- | ---: | ---: | ---: | ---: |
| effective ray coverage | 52.16% | 52.29% | 87.05% | 86.69% |
| unique effective rows | 973,662 | 1,473,868 | 1,624,916 | 2,443,211 |
| ownerless supported hits | 717,594 | 725,949 | 1,175,988 | 1,201,599 |
| missing supported fraction | 4.58% | 3.34% | 5.72% | 4.03% |

Canonical population, split radius and split generation remain almost
identical through 8k; v110's additional population is exact-owner dynamic
coverage rather than extra canonical fog.  By 7k, selected split generation
has moved from zero to one and the selected radius median/p90 has fallen from
9/21 px to 7/13 px.  At the 2M boundary, role-matched local capacity
reallocation selected every requested redundant descendant with zero
shortfall and removed no static skeleton rows.

The 00408--00410 milestone renders nevertheless expose a separate RGB
contract defect:

| conditioned three-view diagnostic | 3k | 6k | 9k |
| --- | ---: | ---: | ---: |
| tree PSNR | 10.5829 | 10.7475 | 10.3930 |
| non-tree PSNR | 12.6861 | 12.7805 | 12.6842 |
| tree gradient cosine | 0.0659 | 0.1292 | 0.1531 |
| mean volume alpha, view 00408 | 0.6155 | 0.7009 | 0.7387 |
| mean RGB bias versus GT, view 00408 | -0.1613 | -0.1722 | -0.1953 |

Thus topology is creating smaller, more aligned structure, but added optical
mass reveals a dark low-frequency appearance estimate.  This is not explained
by missing topology activation.

The checkpoint's own visitation audit identifies the cause.  At 9k, view
00408 has received zero conditioned RGB updates, 00409 one and 00410 two;
63 of 1,487 database cameras have received none.  The legacy
`_evidence_biased_schedule()` replaced up to 65% of the full-scene epoch with
exact-owner cameras.  It did not first reserve an RGB visit for every camera,
so the fixed-length conditioned stream could erase all visits of a
non-evidence view.  This is especially harmful for cameras whose metric DAV2
fit was correctly rejected: geometry transfers from measured neighbours, but
their appearance never receives the RGB correction needed after alpha grows.

The repaired scheduler constructs a fresh balanced epoch only over the
conditioned-active interval, protects up to three visits for every database
camera, and applies evidence weighting only to the remaining positions.  For
a clean 12k schedule it has 10,800 active steps, zero unvisited cameras, a
three-visit minimum, 11.88 mean exact-owner visits and 3.0 mean other-camera
visits.  A repair continuation from the retained 6k checkpoint has 5,040
active steps remaining and still guarantees three visits to all 1,487 views;
all non-conditioned schedules, evidence, model/CUDA code and optimizer state
remain immutable.  An explicit resume migration rejects any unrelated
contract change.

## v110 full result and v111 scheduler intervention

The completed 1,487-view historical-uint8 evaluation separates a real local
tree improvement from the remaining global regression:

| conditioned scope | v100 | v108 | v110 | v110 - v100 |
| --- | ---: | ---: | ---: | ---: |
| raw PSNR | 14.9953 | 14.9074 | 14.9118 | -0.0835 |
| static-valid PSNR | 17.0476 | 16.9099 | 16.9248 | -0.1229 |
| non-tree-static PSNR | 16.8898 | 16.8335 | 16.8332 | -0.0565 |
| tree-static PSNR | 16.2247 | 16.2722 | 16.4201 | +0.1953 |
| boundary-inside PSNR | 14.5297 | 14.7346 | 14.9137 | +0.3840 |
| boundary-outside PSNR | 13.5312 | 13.3212 | 13.3154 | -0.2158 |

Tree high-frequency gradient cosine also rises from 0.3116 in v100 and 0.3087
in v108 to 0.3215 in v110.  The measured strict posterior therefore improves
the intended tree interior and boundary, but it does not yet recover the
historical whole-scene reconstruction quality.

v111 changes only the conditioned-view suffix schedule after the retained v110
6k checkpoint.  Its exact combined visit audit is:

| schedule quantity | v111 |
| --- | ---: |
| active conditioned steps | 10,800 |
| minimum / median / maximum visits | 3 / 6 / 14 |
| unvisited database cameras | 0 |
| visits for 00408 / 00409 / 00410 | 3 / 4 / 4 |

At the matched 9k checkpoint, conditioned tree PSNR over 00408--00410 changes
from 10.3930 to 10.3461 dB and non-tree PSNR from 12.6842 to 12.6915 dB.
Consequently, complete RGB visitation is a required correctness property but
is not the causal fix for these three smeared views.  Their shared defect is
earlier in the evidence producer: v66 persists zero eligible pixels, zero ray
factors and zero dynamic births for all three exact cameras.

The trainer now persists an exact conditioned visit ledger in every retained
checkpoint and the evaluator validates it.  Future analyses no longer infer
visits from a regenerated schedule or combine prefix/suffix counts manually.

## v67 trained-rigid depth posterior

The sparse seed z-buffer used by v66 is too incomplete to calibrate DAV2 in
00408--00410.  Rendering the trained rigid 2DGS instead produces a continuous
rigid depth field.  A reciprocal DAV2 fit is used only when its affine metric
scale is positive; high-residual positive fits become weak continuous
posteriors whose confidence is attenuated by their spatial residual sigma.
Negative or non-physical fits remain rejected.  The trained PLY is used solely
for metric calibration and occlusion, never as foliage renderer
initialization.

Initialization v67 rebuilds only the foliage artifact while keeping the
surface seed and fixed cameras immutable:

| quantity | v66 | v67 |
| --- | ---: | ---: |
| DAV2 accepted views | 703 | 750 |
| weak continuous views | 0 | 27 |
| cameras with zero dense support | 773 | 735 |
| eligible tree pixels | 37,912,126 | 43,905,641 |
| allocated dense rays | 1,453,915 | 1,531,739 |
| retained observation-space rays | 1,405,193 | 1,426,749 |
| exact-view dynamic births | 273,558 | 283,262 |
| initial volume primitives | 501,983 | 504,790 |

Most importantly, the three causal target cameras change as follows:

| rendered view | image id | v66 rays / births | v67 rays / births | v67 candidate intervals |
| --- | ---: | ---: | ---: | ---: |
| 00408 | 1949 | 0 / 0 | 2,048 / 384 | 13,460 |
| 00409 | 1948 | 0 / 0 | 2,048 / 384 | 10,663 |
| 00410 | 1871 | 0 / 0 | 2,048 / 384 | 8,027 |

This is a deliberately local correction rather than an all-camera synthetic
completion: 737 views still fail the physical fit and do not receive invented
depths.  v112 is the clean controlled replay from v67.

The foliage producer also replaces two per-camera full-track Python membership
scans with one reverse image-to-track incidence graph.  This removes the
dominant `O(number_of_cameras * number_of_tracks)` initialization overhead
without changing any selected row.  Current repository validation is 650
passing tests, successful Python bytecode compilation and a clean
`git diff --check`.

## v68--v113 spatial dynamic-authority intervention

The v112 replay exposed a second producer contract defect rather than a need
for more iterations.  Dense DAV2 births stored a synthetic tree confidence
with a 0.65 floor and persisted zero `ray_depth_nll`, so a high-variance
single-view hypothesis appeared as certain geometry.  In 00408--00410 each
camera produced 384 such births, only 14.6--18.2% belonged to a local
replacement group, and the ungrouped rows started near alpha 0.072 despite a
roughly 3 m position posterior.

Initialization v68 keeps every v67 position, ray and primitive but repairs the
continuous authority metadata:

- dense `tree_fraction` is the measured confidence rather than
  `0.65 + 0.35 * confidence`;
- every dense and direct DAV2 birth stores
  `ray_depth_nll = confidence^-2 - 1`;
- track NLL is retained through foliage merge;
- initial and training-time dynamic opacity use spatial depth reliability and
  local replace-and-retire ownership.

The controlled artifact comparison is exact: 504,790 primitives, 1,426,749
observation-space rays, 283,262 dense births and every centre/role/source are
unchanged.  Only 103,319 weak dynamic rows lose unsupported alpha authority.
For the three target cameras the median reliability is about 0.076--0.078 and
initial alpha falls to approximately 0.022--0.036 without a binary rejection.

This change alone is insufficient.  At the matched 3k milestone:

| conditioned 40-view scope | v110 | v112 | v113 | v113 - v112 |
| --- | ---: | ---: | ---: | ---: |
| static-valid PSNR | 12.8035 | 12.5204 | 12.4771 | -0.0433 |
| non-tree-static PSNR | 11.7222 | 11.6268 | 11.6189 | -0.0079 |
| tree-static PSNR | 13.2051 | 12.9389 | 12.8755 | -0.0634 |
| tree-boundary-inside PSNR | 10.7211 | 10.6722 | 10.6090 | -0.0632 |

For 00408--00410 specifically, v113 tree-static is 10.2265 dB versus
10.2967 in v112 and 10.5829 in v110.  The intervention does reduce conditioned
dynamic alpha, but canonical v113 remains within 0.002 dB of v112 and still
trails v110 static-valid by 0.2155 dB.  The run was therefore stopped after
the retained 3k checkpoint instead of spending the full 12k budget on a
falsified mechanism.

## v114 continuous posterior-aware topology allocation

The v113 trace identifies why opacity authority did not change geometry
quality.  Dynamic posterior reliability affected total split demand, but two
later operations discarded it:

1. cross-role capacity used binary `eligible / observable` counts, making
   nearly every exact-owner dynamic row a full unit of demand;
2. `dynamic_split_score` overwrote the reliability-weighted generic score, so
   weak dynamic rows were not even down-ranked inside their role.

Consequently v113 allocated roughly 59% of every mature topology event to the
dynamic role even though many of those rows had weak single-view depth.  The
repair defines a strictly positive per-primitive topology authority from
depth posterior and local replacement ownership.  It enters integrated
demand, role-normalized effective eligible mass, retirement matching and
within-role split priority.  No row is rejected and no fixed role percentage
is introduced.

The first real v114 event keeps the same 224 net-growth slots but changes
static-skeleton/canonical/dynamic allocation from v113's 56/66/102 to
81/68/75.  At iteration 1,000 the total volume count differs by only 240 rows,
while v114 contains 11,364 more canonical and 11,608 fewer dynamic rows.  Its
00408--00410 render remains within 0.0043 dB of v113 in every reported PSNR
region, which verifies area/optical-mass preserving topology rather than an
image-changing initialization shortcut.

At 3k, v114 has 410,616 canonical and 549,763 dynamic rows versus v113's
336,502 and 626,299, while total volume count changes by only -2,431.  The
extra reliable bandwidth raises conditioned tree edge-energy ratio by 0.0281,
non-tree PSNR by 0.0085 dB and the 00408--00410 tree PSNR by 0.0484 dB.  It
does not yet improve the whole 40-view objective: conditioned tree PSNR is
0.0284 dB below v113 and static-valid is 0.0149 dB lower.  Topology authority
is therefore a real local/high-frequency correction, but not the missing
loss-scale correction by itself.

Repository validation after this repair is 653 passing tests.

The same causal audit found one independent loss-scale defect to be tested in
the subsequent clean replay.  The new ownerless ray rows for 00408--00410 all
have confidence between about 0.061 and 0.100 (median approximately 0.078),
but the interval factor computed `sum(confidence * loss) / sum(confidence)`.
When one weak camera supplied uniformly low confidence, the denominator
cancelled its absolute posterior strength and restored a full-strength loss.
The corrected factor divides by the number of evaluated rays instead.  It
therefore preserves relative per-ray weighting *and* absolute camera-level
confidence; weak rays remain active with non-zero gradient but cannot steer a
cross-sequence canonical lineage as strongly as calibrated multi-view
evidence.  This contract is covered by a direct factor/gradient regression
test, bringing repository validation to 654 passing tests.

## v115 ray scale result and v116 optical/RGB authority repair

v115 is a clean replay of v114 with only the absolute ray-confidence scale
repair.  It was stopped at the retained 3k checkpoint after the matched
40-view evaluation:

| conditioned scope at 3k | v114 | v115 | delta |
| --- | ---: | ---: | ---: |
| raw PSNR | 12.0221 | 12.0406 | +0.0185 |
| static-valid PSNR | 12.4649 | 12.4968 | +0.0320 |
| non-tree-static PSNR | 11.6336 | 11.6390 | +0.0054 |
| tree-static PSNR | 12.8465 | 12.9021 | +0.0556 |
| boundary-inside PSNR | 10.6028 | 10.5834 | -0.0194 |
| boundary-outside PSNR | 11.5968 | 11.6081 | +0.0113 |

The global gain confirms the scale correction, but the weak continuous-depth
target views 00408--00410 regress from 10.3066 to 10.1938 dB tree PSNR.  Two
independent downstream contracts explain that local failure:

1. coverage preservation protected the *first* three occurrences of every
   camera.  Evidence oversampling was consequently delayed until three
   complete 1,487-camera epochs had nearly elapsed.  By 3k the three target
   cameras had only 1/2/2 conditioned visits;
2. the v113 spatial opacity intervention projected their weak DAV2 births to
   a hard alpha ceiling near 0.02.  Dynamic optical mass at 3k fell from
   62,503.8 in v110 and 53,124.4 in v112 to 28,925.1 in v114 and 28,855.0 in
   v115.  For the target owner rows, median alpha is only 0.0233.

Both operations confused geometry confidence with optical existence.  v116
repairs the causal chain without removing spatial uncertainty:

- exactly three protected full-scene conditioned epochs are distributed
  uniformly over the active interval and interleaved with evidence epochs;
  final coverage is still exact, while target visits by 3k become 3/3/3;
- depth confidence still controls initialization, ray-factor magnitude,
  topology demand and split priority, but no longer hard-clamps a leaf that
  exact RGB evidence says is optically present;
- dynamic alpha retains the ordinary role-wide 0.40 safety ceiling, and
  calibrated free-space/ray losses remain differentiable spatial controls.

This keeps the canonical geometry posterior conservative while restoring the
conditioned branch's capacity to fill real tree crowns instead of rendering a
sparse set of low-alpha points.

The matched 3k replay confirms both parts of the repair.  v116 restores exact
target-camera visitation to 3/3/3 and removes the training-time spatial alpha
projection:

| conditioned historical-uint8 scope at 3k | v115 | v116 | delta |
| --- | ---: | ---: | ---: |
| raw PSNR | 12.0371 | 12.1393 | +0.1022 |
| static-valid PSNR | 12.4934 | 12.6189 | +0.1255 |
| non-tree-static PSNR | 11.6331 | 11.6726 | +0.0395 |
| tree-static PSNR | 12.9017 | 13.0558 | +0.1541 |
| boundary-inside PSNR | 10.5882 | 10.6928 | +0.1047 |

For 00408--00410, tree-static rises from 10.1618 to 10.3826 dB.  This is a
real gain, but the target views still trail v110 because v68 initialized only
384 observation-local volume rows for roughly 183k--199k valid tree pixels
per view.  At 1k the exact target-owner rows cover only about 33% of 00408's
tree silhouette under a conservative projected-footprint estimate.  Raising
alpha cannot synthesize missing spatial bandwidth; it only makes the same
sparse support more opaque.

## v117 optical existence / geometry posterior separation

The evidence producer formerly multiplied initial leaf alpha by metric-depth
reliability.  This repeated the same conceptual error earlier removed from
the optimizer: a calibrated tree-labelled pixel proves foreground optical
existence along its ray even when its 3D depth posterior is broad.  v69 is a
deterministic, opacity-only migration of v68.  It raises 319,769 observed
dynamic rows to the support/free-space optical prior while leaving every
centre, scale, colour, role, ray, camera and surface tensor byte-identical;
historical RGB is never fitted.

v117 replays v69 with the v116 interleaved schedule.  Its matched 3k result is:

| conditioned historical-uint8 scope at 3k | v116 | v117 | delta |
| --- | ---: | ---: | ---: |
| raw PSNR | 12.1393 | 12.1860 | +0.0468 |
| static-valid PSNR | 12.6189 | 12.6790 | +0.0601 |
| non-tree-static PSNR | 11.6726 | 11.6775 | +0.0048 |
| tree-static PSNR | 13.0558 | 13.1429 | +0.0871 |
| boundary-inside PSNR | 10.6928 | 10.8042 | +0.1114 |
| boundary-outside PSNR | 11.6560 | 11.6801 | +0.0241 |

The causal target is stronger: 00408--00410 tree-static improves from 10.3826
to 10.6282 dB and boundary-inside from 10.6512 to 10.8514 dB.  Target tree
PSNR is now above v110's matched 3k value of 10.5829 dB.  Global tree PSNR is
still 0.0621 dB below v110, so v117 validates the optical-existence repair but
does not remove the finite-basis bottleneck.

The subsequent producer contract therefore keeps the ordinary 384-birth cap
unchanged and raises it to 2,048 only for exact cameras whose observed tracks
carry a weak-continuous DAV2 posterior.  Geometry/ray gradients remain scaled
by metric confidence; the extra rows supply smaller local optical footprints
rather than stronger depth authority.  The unified initializer, rigid-depth
rebuild path, Cambridge profiles and provenance manifest all persist this
parameter instead of relying on a one-off experiment command.

v117 also demonstrates why a 3k causal checkpoint is not a final-quality
proxy. Continuing the same run to 6k raises 40-view conditioned
historical-uint8 static PSNR from 12.6790 to 13.3199 dB and tree PSNR from
13.1429 to 14.0386 dB. Tree gradient cosine rises from 0.1314 to 0.2096. The
00408--00410 tree mean improves from 10.6282 to 10.9984 dB. Longer training
is therefore necessary, but the much slower target-view gain still identifies
a local representation-bandwidth deficit rather than an iteration-only issue.

## v70--v71 measured optical-bandwidth allocation

v70 rebuilds the foliage producer with a 2,048-birth ceiling only for the 27
weak-continuous DAV2 cameras. Its contract audit is exact:

- all 1,426,749 dense ray rows are unchanged;
- dynamic births rise from 283,262 to 328,190;
- every one of the 27 weak views has 2,048 births;
- all other 1,460 database cameras retain the ordinary 384 limit;
- the surface seed is the same hard-linked inode and SHA-256;
- the dense initial alpha median is 0.0901439 and no historical model/RGB fit
  is used.

At 1k, v118 (v70 plus the optical-existence ray-factor split) improves the
00408--00410 conditioned historical-uint8 target mean over v117 as follows:

| target scope at 1k | v117 | v118 | delta |
| --- | ---: | ---: | ---: |
| raw PSNR | 11.5613 | 11.6421 | +0.0808 |
| static-valid PSNR | 11.9560 | 12.0585 | +0.1025 |
| tree-static PSNR | 11.8978 | 12.0134 | +0.1156 |
| boundary-inside PSNR | 10.9052 | 10.9687 | +0.0635 |
| tree gradient cosine | 0.0166 | 0.0262 | +0.0095 |

The per-camera result exposes the remaining categorical-policy error. 00409
and 00410 are weak-continuous, receive 2,048 births, and gain +0.1900 and
+0.1420 dB tree PSNR. 00408 has an accepted metric fit, remains at 384 births
despite 199,229 measured tree pixels, and gains only +0.0149 dB. Optical
bandwidth is a function of measured silhouette area as well as metric-depth
uncertainty; an accepted depth label does not imply that 384 finite splats can
represent a large high-frequency tree.

v71 therefore defines a continuous source-raster bandwidth contract in
addition to the weak-posterior ceiling:

```text
births(view) = clamp(
    ceil(eligible_tree_pixels(view) / 192),
    ordinary_floor=384,
    adaptive_ceiling=2048,
)
```

Weak-continuous views still receive the ceiling. On the measured v70 audit,
the per-camera renderer basis is additionally bounded by the number of
accepted posterior rays.  The exact producer contract is therefore
`min(births(view), accepted_rays(view))`.  The completed v71 artifact passes
this formula for every camera: 379,033 dense births (+50,843 over v70), 226
adapted views, 1,038 rows for 00408 and 2,048 each for 00409/00410.  It retains
the identical 1,426,749-row dense posterior, the same hard-linked surface
seed, zero cross-instance replacement links and no historical model/RGB fit.
The final volume seed has 155,619 canonical crown, 356 static skeleton and
444,586 dynamic rows; 230,500 dynamic rows have a local canonical group and
214,086 remain exact-camera-only.  This is a bounded continuous allocation
rather than a semantic accept/reject gate. Geometry confidence, candidate
intervals, topology authority and local retirement are unchanged.

The retained v118 3k checkpoint confirms that the weak-continuous allocation
is not a three-frame-only effect.  Relative to v117 on the exact same 40-view
historical-uint8 diagnostic set, conditioned static-valid PSNR improves from
12.6790 to 12.7853 dB, tree-static from 13.1429 to 13.2937 dB, and
tree-boundary-inside from 10.8042 to 10.9120 dB.  Non-tree-static also moves
slightly upward (11.6775 to 11.6803 dB), so the gain is not obtained by
covering rigid pixels with extra foliage.  Tree gradient cosine rises from
0.1314 to 0.1374, although edge-energy ratio falls from 0.7115 to 0.7013;
coverage/colour improve faster than leaf-scale frequency at this checkpoint.

For 00408--00410, v118 versus v117 at 3k changes static-valid by +0.2476 dB,
tree-static by +0.2745 dB and tree gradient cosine by +0.0229.  The per-view
tree gains are +0.0588 / +0.3428 / +0.4218 dB.  The two weak-continuous views
receive 2,048 rows and account for nearly all of the gain; accepted-depth
00408 still has only 384 rows.  This both validates measured optical
bandwidth as a causal bottleneck and supplies the direct test for v71's
continuous area allocation.

## v72--v73 ownership-isolated producer repair

The first v70/v71 tensor-level comparison exposed a separate producer bug
that ordinary count/provenance audits could not detect.  Changing only the
number of exact-camera renderer-basis births also changed persistent
MASt3R/Chart/DAV2 rows: 518/634 canonical track scales, 532/634 canonical
track rotations, and 119/136 static-skeleton scales changed.  One persistent
MASt3R track was even reassigned from tree instance 511 to 542.  Centres,
colours and opacity were unchanged, which made the defect easy to miss.

There were two density-to-geometry back edges:

1. local frames were recomputed with a global k-nearest-neighbour query after
   appending all dense renderer-basis rows, so optical sampling density could
   change a persistent track's covariance, scale and rotation;
2. unassociated persistent tracks and new renderer-basis rows were clustered
   together, so adding local optical rows could relabel the pre-existing
   ownership graph.

v72 removes both back edges.  Persistent evidence rows and incremental
renderer-basis rows now obtain local frames in disjoint neighbourhoods.
Persistent instance labels are completed first; incremental rows may inherit
a nearby persistent owner or form a new bounded component, but can never
relabel a persistent row.  This makes measured optical bandwidth a downstream
consumer of geometry/ownership rather than an authority over it.  Two direct
regression tests vary the incremental sample count and assert byte-identical
persistent frames and labels.  The full repository suite passes 660 tests.

A second audit on the real 600,561-row v71 seed found that class separation
alone was not sufficient.  Among a deterministic 60k-row sample, 81.25% of
dense rows used at least one kNN frame neighbour owned by another camera and
11.31% crossed tree instances; 7.10% of persistent rows crossed instances.
Those neighbours cannot be local shape measurements for an exact-camera
dynamic primitive that is never rendered in the neighbour camera.  v73
therefore partitions persistent frame estimation by tree instance and dense
frame estimation by `(exact owner camera, tree instance)`.  The real seed has
1,048 persistent-instance groups, 751 exact dynamic owners and 7,779 dynamic
owner-instance groups.  The grouped frame computation takes about 6.9 seconds
for all 600,561 rows, so the stronger causal contract is not a material
initialization-time cost.  Direct owner/instance isolation coverage brings
the complete suite to 661 passing tests.

The old-frame v119 1k checkpoint is still useful as a directional, but not
single-variable, observation.  Relative to v118 on 00408--00410 it improves
conditioned historical-uint8 static-valid by +0.0289 dB, tree-static by
+0.0306 dB, and tree gradient cosine by +0.0032.  The area-limited 00408 view
gains +0.0703 dB tree PSNR, while 00409 changes -0.0036 dB and 00410 +0.0251
dB.  Non-tree-static changes only +0.0008 dB.  The clean ownership-isolated
replay is therefore required before attributing the full delta to area-based
birth allocation.

At 3k the same old-frame comparison remains positive but small:
00408--00410 static-valid changes +0.0213 dB, tree-static +0.0219 dB and
tree edge-energy ratio +0.0018.  The additional area allocation is concentrated
in 00408, whose tree PSNR changes +0.0532 dB; 00409/00410 change +0.0126 and
-0.0001 dB, while aggregate non-tree-static changes only +0.0012 dB.  Area
bandwidth is therefore a real coverage correction, not the sole explanation
for broad leaf blur.  v73's exact-owner/instance covariance replay tests the
larger ownership-contamination mechanism.

## v74 / v120 exact-ray camera-plane topology repair

The 00408 v119/3k render remained visibly smeared even after increasing its
initial basis from 384 to 1,038 rows.  A checkpoint-level causal audit rules
out the usual explanations:

- all 1,038 source-4 rows are visible in their exact owner, with raster radii
  4--12 px (median 8 px);
- all receive finite non-zero screen gradients (median `1.69e-5`);
- base reprojection drift is only 0.075 px median and conditioned drift is
  0.123 px median;
- depth drift is millimetric in the base state and at most about 2.3 cm after
  conditioning;
- 95% of the rows are more than 15 cm in front of the rigid z-buffer, so the
  missing foliage is not hidden behind the church surface.

Nevertheless every 00408 source-4 row still had generation zero at 3k.  The
owner context itself was not starved: the same camera had 368 source-3 DAV2
dynamic rows, of which 298 were descendants.  The reason is a duplicated
geometry-confidence penalty.  Source-4 depth authority had median 0.0771 and
ownerless occupancy authority was multiplied again, giving median topology
authority 0.0103.  Source-3 rows had median 0.4324, about 42 times larger.
Thus the context-balanced selector allocated the camera's measured demand
entirely to the wrong representation family.

The repair separates two physically different mutations:

1. new 3D occupancy still uses the continuous ray/depth geometry authority;
2. subdivision of a visible broad exact-owner EWA basis uses unit optical
   bandwidth authority, because it does not create a new depth claim.

The second mutation is not the old unconstrained 3DGS split.  Children are
offset only in the calibrated owner camera plane.  A binary split contracts
one image-plane axis by two; a quaternary split contracts two axes by two.
The depth-axis Gaussian extent and the stored position covariance are kept
unchanged, so image bandwidth cannot masquerade as a more certain metric
posterior.  Optical depth is conserved.  Owner contexts are balanced first,
then initialization/evidence families, then tree instance and world cell;
unused capacity is refilled, so this is capacity fairness rather than a hard
accept/reject gate.

The real v119 checkpoint replay is decisive.  On the exact 1,406 dynamic rows
owned by camera 1949, an isolated adaptation event selected 470 parents:
235 source-3 and 235 source-4, with all 235 source-4 parents taking the
camera-plane path.  The old run produced zero source-4 descendants over its
first three visits.  New topology events persist eligible and selected owner
camera ids, per-source parent counts, exact-ray eligible counts, and the
selected-versus-mutated camera-plane count so this failure cannot again hide
behind aggregate split totals.

The completed v73 producer has 379,033 dense births, 751 exact owners and
9,377 `(owner, tree instance)` frame groups.  Cross-owner and cross-instance
frame-neighbour counts are both exactly zero; target births are
1,038/2,048/2,048 for 00408--00410.  The surface seed is byte-identical to
v71 and no historical reconstruction/RGB initialization is used.  The full
project test suite passes 664 tests.

As a long-training reference, v117 at 12k reaches 40-view conditioned
historical-uint8 static-valid PSNR 13.8495, tree-static 14.7454 and
non-tree-static 11.9971.  Its 6k static/tree values were 13.3199/14.0386, so
the second half of training remains useful even though representation
bandwidth, not iteration count alone, caused the target smear.

The clean v120 run combines v73 and the v33 camera-plane topology contract.
Its first active event at iteration 300 already reports 22,059 exact-ray
eligible rows and mutates 35 source-4 parents through the camera-plane path;
the selected source counts are 17/18/1/5/35 for sources 0/1/2/3/4.  Net
volume growth is only 224 because the existing smooth topology ramp is still
in force.  This confirms that the repaired path is live without replacing
adaptive demand by an early global densification burst.

At iteration 700 the direct failed owner also enters the live path: camera
1949 is present in both the eligible and selected owner-id lists.  All 75
eligible owner contexts receive capacity.  The event mutates 1,107 source-4
camera-plane parents (`selected == mutated`) while adding 6,669 net dynamic
rows under the same gradual capacity ramp.  This removes the v119 situation
where camera 1949 received context capacity but every one of its 1,038
source-4 rows remained generation zero.

The retained 1k checkpoint closes the lineage audit.  Camera 1949 now has
1,155 source-4 rows, including 155 descendants (151 generation-one and four
generation-two rows); the initial population was 1,038 and v119 still had
zero descendants at 3k.  Its source-3 family simultaneously retains 142
descendants, so the fix did not merely reverse which family is starved.  The
1k event selects all 80 eligible owner contexts and reports 2,152 selected
and 2,152 actually mutated source-4 camera-plane parents.

## v120 ownership counterfactual and v74 optical-mass calibration

The v120 3k checkpoint was rendered with physical branch isolation on
00408--00410.  Canonical-only volume is a dark low-frequency layer (tree PSNR
10.1143, mean luma 0.2392, alpha 0.7001), while exact/dynamic-only volume is
bright but optically sparse (tree PSNR 10.0736, mean luma 0.6052, alpha
0.0966).  Their jointly sorted mixed render is better than either branch
alone (tree PSNR 10.8830), so deleting one owner is not a valid repair.

A render-only dynamic-opacity sweep supplies a stricter intervention.  With
geometry, colour, mixed ordering and every other parameter fixed, adding
+3.0 to dynamic opacity logits raises target tree PSNR to 12.3239 dB and
reduces MAE from 0.2316 to 0.1903.  The same sweep on the mature v118 12k
checkpoint has a much smaller optimum at +0.5 (12.2489 to 12.3391 dB) and
regresses for larger offsets.  The causal conclusion is not a permanent
opacity floor: exact-owner leaves need enough *initial* optical mass to learn
their RGB/coverage responsibility, after which ordinary opacity optimization
must remain free to reduce it.

Initialization v74 therefore changes only exact source-4 dense-ray dynamic
rows from `0.035 + 0.065 * optical_existence` to the bounded prior
`0.08 + 0.32 * optical_existence` (maximum 0.40).  The real 379,033-row
migration changes median alpha from 0.09014 to 0.35148.  A NaN-aware tensor
audit proves that centres, scales, rotations, colours, observation UV/depth,
ray evidence and the hard-linked surface inode are unchanged; no historical
RGB or Gaussian model is read.

At the matched v120/v121 1k topology event, volume populations differ by only
eight rows and both spend 17,757 net-growth slots.  Source-4 split parents are
2,152 versus 2,162.  The ray hit loss nevertheless falls from 2.3519 to
1.7158; at 1.5k the optical-existence term falls from 0.2612 to 0.0383 while
free-space loss changes from 0.0097 to 0.0093.  This is the intended
same-topology optical-existence effect.  Render metrics are recorded after
the retained 3k checkpoint completes.

## v35 atomic volume topology and bounded optimizer migration

The trainer previously described volume pruning and splitting as one atomic
mutation but materialized a complete pruned model before materializing the
final split model.  The old Adam optimizer still referenced the original
parameters.  At the 2M budget this created an avoidable multi-topology memory
peak; the v119 10k failure occurred on a 636 MB prune allocation while only
180 MB was free because the physical GPU was concurrently occupied.

`replace_and_split_adaptive()` now performs selection in the original row
space and creates the final keep-plus-children tensors exactly once.  The
returned lineage map still addresses the pre-event topology, so optimizer
semantics are unchanged.  Adam migration additionally pops one old parameter
state at a time, transfers unchanged appearance/sky state without copies and
uses the independent storage already produced by advanced indexing instead
of cloning it again.  Old parameter/state references are released as soon as
their replacement is installed.  Regression tests prohibit a fallback to
the old `prune()` then `split_adaptive()` sequence and compare every migrated
Adam row against the pre-event mapping.

The runner records this as teacher protocol v35 / pipeline v37.  Mixed
forward rendering is unchanged and the v34-to-v35 renderer hash pair is
explicitly marked render-equivalent for old-checkpoint evaluation.  Fast and
quality runs now save the roughly 1.5 GB rolling checkpoint every 3k steps
instead of rewriting unretained files every 500--1,000 steps.

## v37--v39 complete ray epochs and conditioned topology settle

The v37/v38 audit separated two previously conflated questions: whether the
ray posterior was consumed completely, and whether the final volume topology
had enough optimization time.  Ray-posterior batch capacity is now derived
from the persisted per-camera cursors and stops only at a complete camera
table boundary.  The v125 12k checkpoint consumes all 2,854,438 effective
rows with interval coverage 1.0, minimum visit one, no missing camera rows and
no geometry/representation warnings.  This repair is necessary provenance,
but its matched v124/v125 render difference is small, so incomplete ray
coverage was not the remaining image-quality bottleneck.

A second audit found that `canonical_polish` disabled more than topology.  It
also disabled the conditioned forward path and independently zeroed the
deformation, dynamic-feature and dynamic-opacity basis gradients.  The last
volume event occurs at iteration 11,000 and the phase changes at 11,040, so
roughly 13k newly replaced dynamic descendants received only about forty
conditioned updates before the old trainer froze them for the final 960
steps.  Protocol v39 defines the corrected contract precisely:
`canonical_polish_disables_topology_not_conditioned_optimization`.  Volume
population and ordering remain fixed, while conditioned RGB, high-frequency,
ownership, dynamic observation and all three dynamic bases continue to train
on the existing full-scene RGB epoch tail.

The v128 replay is a strict causal check from the same v122 iteration-9k
checkpoint.  At 10k, v125 and v128 have identical surface/volume populations,
role counts, split/prune counts and ray coverage; loss differs by only
1.7e-6.  Their 40-view iteration-10.5k PSNR differences are below 3e-5 dB.
At iteration 11.1k the old trace has `dynamic_active=false` and zero
conditioned/HF/conditioned-ownership loss, while v128 has
`dynamic_active=true`, conditioned loss 0.2935 and HF loss 0.0588.  Thus the
observed final difference starts only after the intended phase boundary.

On the complete 1,487-view historical uint8 protocol, v128 conditioned
static-valid reaches **17.036912 / 0.761155 / 0.106523**
(PSNR/SSIM/MAE).  Tree-static reaches **16.674256 / 0.582764 / 0.108974**,
non-tree static 16.943794, tree-boundary-inside 15.311470 and
tree-boundary-outside 13.589653.  Relative to v125 these are +0.056449,
+0.083491, +0.050947, +0.080048 and +0.009957 dB respectively, with SSIM
increasing and MAE decreasing in every listed region.  Relative to v100, the
overall conditioned result is only 0.010726 dB lower, while tree-static is
+0.449549 dB, boundary-inside +0.781723 dB and non-tree +0.054034 dB.

Canonical v128 reaches 16.807028 / 0.750809 / 0.111153.  It gains 0.053601 dB
over v125 and is only 0.027703 dB below v100; its non-tree canonical score is
0.061385 dB above v100.  The repair therefore does not obtain its tree gain by
damaging the rigid map.  Conditioned visits total 11,760 across all 1,487
views (minimum/median/maximum 3/11/14), and the complete ray posterior remains
at 100% coverage.

This is a real but bounded improvement.  Tree gradient cosine rises from
0.341918 to 0.349072, while edge-energy ratio falls from 0.907759 to 0.899516.
The 00408--00410 renders change mainly inside canopy and occlusion boundaries,
but remain visibly low-frequency.  The next bottleneck is therefore not more
iterations or another global opacity adjustment: canonical crown and dynamic
leaf remain two independently coloured optical layers that compensate through
alpha.  The next representation change must make canonical optical mass a
shared state with conditioned residuals, and must suppress volume support on
non-tree rays locally rather than with a scene-wide gate.
