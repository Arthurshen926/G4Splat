# Cambridge native Hybrid Teacher mainline

The authoritative reconstruction path is a single fixed-camera,
MASt3R/MAtCha/G4 evidence-to-native-Hybrid-Teacher pipeline. It does not train
or report a student model and does not initialize from a historical trained
Gaussian PLY. Raw SfM tracks are an optional, explicitly labelled coverage
stream; they never replace MASt3R/MAtCha as the geometry authority.

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
  --geometry-source mast3r_primary_sfm_coverage \
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

For staged quality/fast runs, the first initialization materializes the rigid
surface plus an inert one-row foliage schema placeholder. The expensive
all-camera foliage posterior is built exactly once, after rigid pretraining,
using that trained 2DGS as the occlusion/depth calibrator. Initialization
cache reuse compares the required causal fields rather than demanding byte-for-
byte equality with extra provenance keys.

`distill_student` is intentionally rejected. A standard 2DGS/3DGS conversion
would be a separately labelled approximation and must not be silently reported
as the native Teacher result.

## Data and evidence contract

- Cambridge intrinsics and poses are fixed and retain exact off-axis
  principal points.
- `cameras.bin` and `images.bin` carry the fixed-camera serialization.
  With `mast3r_only`, `points3D.bin` is forbidden. With
  `mast3r_primary_sfm_coverage`, raw SfM tracks may propose renderer births
  only where the primary MASt3R/MAtCha scaffold has a measured coverage
  deficit.
- MASt3R pointmaps/tracks, MAtCha Charts, planes, DAV2 ordinal depth and
  foliage ray intervals remain content-addressed observations.
- Renderer births are disposable optimization variables. Track ids,
  observation pixels/depths, source roles, tree instances and lineages remain
  persistent after split/prune.
- RGB, geometry and topology use independent camera schedules.
- A cross-sequence per-track role posterior separates stable tree surface
  evidence from uncertain crown support. Reprojection error, triangulation
  baseline and cycle consistency are continuous precisions; an absent track
  is unknown, not negative evidence. Stable local line tracks form an explicit
  sparse trunk/branch graph; its projected edges are positive evidence clipped
  by the real tree mask, never a dense semantic hallucination.
- Cross-sequence rigid tracks provide positive-only rescue where a single
  tree, object or sky mask swallowed a repeatably reconstructed facade or
  static object. Exact track observations retain their native cameras; a
  separate content-addressed ragged posterior projects stable 3D tracks into
  all 1,487 fixed cameras. It records geometric support independently from
  robust current-RGB visibility. Visible probability is transferred from the
  mask prior to rigid RGB ownership; supported but currently occluded mass is
  quarantined as unknown/background evidence and cannot explain foreground
  colour. The v2 posterior includes tree-mask conflicts: a projected rigid
  track keeps geometry/depth authority behind foliage, while current-view RGB
  visibility is exactly zero unless that same track has a real observation in
  that same camera. On StMarysChurch this adds 12,268,723 tree-occluded
  projections over 1,449 cameras, but only 163 exact current-camera
  observations; background evidence is restored without turning foliage
  pixels into facade-colour supervision.
- MASt3R pointmap and MAtCha Chart initialization likewise treats masks as
  priors. A masked pixel is admitted only when the immutable cross-sequence
  pointmap posterior or independent metric Chart support is positive, and the
  override is retained in initialization statistics. This restores stable
  trunks, thin objects and mislabeled facade regions without opening an
  unconstrained RGB birth path.
- Static foliage fusion receives the complete 1,487-camera identity/sequence
  table. The bounded ray-posterior view subset is never reused as camera
  metadata. A tree-wide canonical traversal may use a locally coherent
  fallback only for cells already verified in multiple sequences.
- Initialization audits every fixed camera with a deterministic thumbnail
  sharpness/exposure/contrast score. This bounded score breaks ties between
  equally supported canonical observations; it has no geometry or birth
  authority.
- Near-identical, normal-consistent cross-Chart renderer samples use a single
  primary opacity owner. Suppressed secondary samples remain permanent MAtCha
  factors, giving a sparse partition-of-unity without deleting evidence.
- Surface retirement follows the same evidence/renderer separation. A large
  projected 2D surfel is a bandwidth deficit and remains until a bounded
  replace-split preserves its support; historical screen-radius maxima are
  not deletion authority. Opacity-only culling for MASt3R, MAtCha,
  coverage-SfM and DAV2 lineages matures smoothly with cumulative
  responsibility-bearing database sweeps, so the first post-bootstrap event
  cannot confuse an insufficiently sampled calibrated seed with clutter.
- Reaching the global surface memory ceiling does not permit unrelated weak
  geometry to be evicted for a split elsewhere. Candidates are deferred until
  evidence-mature no-contribution retirement releases capacity, or an
  explicit local receiver proves optical handoff. This removes the last
  global "borrow geometry from another object" topology path.
- Optional SfM coverage is appended only after freezing the complete primary
  renderer scaffold. Surface tangent frames are estimated within source
  authority, and DAV2 hole proposals use the pre-SfM primary reference. In the
  formal paired initialization, removing the 124,301 SfM rows from the
  coverage arm leaves all 545,999 MASt3R/MAtCha/Chart/DAV2 rows byte-identical
  across every stored field.

## Representation and training

The mature rigid stage is a native perspective-correct 2D surfel model. The
mixed stage consumes its validated PLY/handoff manifest under the
`atlas_residual` ownership policy: the mature renderer prefix stays fixed,
the calibrated Chart inverse-depth atlas continues at 0.1x learning rate,
and only an appended, independently supported coverage suffix may change
geometry and opacity. This avoids both global rigid drift and a completely
frozen map that cannot fill measured facade/thin-structure holes.

The production target is one unconditional static map. Foliage is represented
by a static trunk/branch skeleton, a persistent porous crown envelope and
cross-sequence-supported static detail. Static runs allocate rank-zero
temporal tensors and contain no sequence-conditioned leaf capacity. A
surface+skeleton counterfactual render routes stable tree-surface RGB, edge
and screen-bandwidth gradients only to the skeleton role. Structural surfels
and volume Gaussians are emitted into one tile list, share center-depth
sorting, and are composited in the same CUDA front-to-back loop.

Training maintains three explicit forward responsibilities. `surface-only`
routes rigid RGB/edge and sky without allowing a wrong foliage alpha to cut
their gradients. `volume-only` exposes intrinsic canopy colour and projected
bandwidth only where a static volume actually renders inside the real tree
mask; opacity is read-only in this pass. `full mixed` remains the authoritative
jointly sorted image and owns occlusion, boundary and final RGB consistency.
The all-camera projected-rigid factor is evaluated only on `surface-only`:
it applies a robust log-depth residual where a background surface already
renders and a sparse coverage residual where it does not. Its screen-space
gradient participates in evidence-driven surface densification. It has no RGB
permission, so a tree-occluded facade can remain geometrically complete while
the full mixed render still shows the foreground crown.

The projected support raster keeps confidence and depth under one primitive
owner. When several stable tracks collapse into a training pixel or its
five-pixel reprojection footprint, maximum geometry confidence selects the
track and nearest depth breaks only an exact-confidence tie. Independently
pooling probability and inverse depth is forbidden because it can grant
authority from one facade track while pulling the surface toward a different
neighbour. The task-field schema and checkpoint implementation hash both bind
this rule. Projected targets outside the renderer's fixed `(0.05, 100)m`
camera-depth interval are removed both when a new evidence store is built and
again when an older v2 archive is loaded; an impossible behind-far-plane row
therefore cannot create a permanent coverage loss.

Occluded projected support is also boundary quarantined. The hidden
geometry-minus-visible posterior is retained in the interior of the measured
uncertain/tree region and attenuated to zero across its scale-aware boundary
band; only a current-camera-visible track may restore ownership at that edge.
This prevents a surface-only completion loss from increasing a broad
background surfel's opacity/scale through the foreground tree boundary.

This v2 posterior is also part of the from-zero runner, rather than a manual
post-processing experiment. `build_evidence` constructs it from the exact
multiview track graph, all 1,487 fixed cameras, the four-channel mask and the
same RGB raster supplied to training; evidence reuse validates the posterior
schema and RGB root before accepting an existing store. The subsequent
pointmap-posterior store therefore inherits the projected artifact and its
content hash automatically.

The 6k causal result does not justify using this factor as an RGB-quality
default. Against the geometry-balanced v85 rigid control, unquarantined v86
changed stratified-64 non-tree PSNR only from `15.9702` to `15.9723` dB and
slightly worsened MAE; v90 boundary quarantine retained `15.9722` dB but
reduced raw/tree/boundary aggregates. It did increase tree-region surface
alpha and improved view 408, demonstrating local geometric completion, while
views 409--412 showed that the available projected tracks do not uniformly
identify the facade explanation needed by RGB. The production reconstruction
selection therefore remains the v85 optimizer/Chart lifecycle; projected
occluded geometry is an optional localization-map factor until an independent
geometry benchmark proves a net benefit. Accordingly,
`--projected-rigid-depth-weight` defaults to zero in production; enabling it
requires an explicit positive value and constitutes a geometry/localization
ablation rather than the selected RGB-reconstruction configuration.

The static leaf handoff first verifies persistent envelope groups using
independent cameras, then selects a coherent per-tree acquisition and at most
two spatial modes per group. If the chosen acquisition did not observe a cell,
an internally coherent local acquisition is allowed only when that same cell
has independent cross-sequence support. In the current causal validation this
repaired complete camera contract increased fused static detail from 1,706 to
5,336 clusters without adding an unconstrained RGB birth path.

The quality profile uses a 2M volume budget and at most 20k net growth slots
per topology event. Growth is still constrained by measured screen-space
deficit, role/instance/owner balance, posterior reliability and a smooth
startup ramp. Unverified child fraction adds smooth verification-debt
backpressure to ordinary splitting; contradiction pruning and independent
cross-sequence ray births remain active. An unverified zero-witness child is
retired only after the configured timeout and only when both contribution and
opacity remain in the bottom utility tail; age alone is not prune authority.
Volume opacity uses the same
conservative `0.004` learning rate
as colour/topology formation; sparse owner scheduling is not compensated by
accelerating opacity alone.

The v69 negative-evidence contract is local and asymmetric. A
surface-only render having lower RGB error is diagnostic, but it may reduce
canopy optical mass only where the independent `p_rigid` posterior grants
that pixel to a rigid owner. `p_canopy` never grants permission to delete its
own foreground; blocked canopy mass and pixel count are written to the
training trace. Rigid/free-space contradiction evidence remains active. This
replaces both failure extremes observed in the causal runs: global
counterfactual cleanup erased genuine crown coverage, while disabling all
cleanup retained broad spill outside the measured tree boundary.

Volume topology is also a reversible proposal. Split children inherit parent
camera rows as candidates, not proof, and must regain real-ray witnesses. If
every extant sibling in one lineage times out with zero witnesses and low
optical utility, the family is merged back to one evidence-supported
representative instead of deleting the whole parent hypothesis. The merge
preserves integrated optical mass and spatial second moment, restores the
immutable evidence identity and clears inherited Adam momentum. Selection is
implemented by a fixed number of GPU sorts/scatters rather than one complete
candidate scan per instance/lineage group.

Screen bandwidth and optical coverage are separate controls. Static
trunk/branch, persistent crown envelope and fused leaf detail have independent
projected-radius ceilings (`12`, `24`, and `12` pixels in the named handoff
profiles). A scale correction preserves `tau * projected_cross_section`, so
the anti-smear bound cannot create transparent holes. Evaluations report
radius and opacity quantiles separately for each layer role.

Both rigid and mixed profiles now end with a topology-free low-learning-rate
suffix. The rigid handoff persists the complete position and non-position LR
lifecycle; legacy manifests may recover it only from a sibling producer
result whose PLY identity and iteration match exactly. The same absolute
24k--32k multiplier is applied to the continuous Chart inverse-depth atlas,
closing the former hidden schedule in which baked surfels entered polish
while their external geometry owner continued moving at a fixed LR. Static
volume SH, spatial appearance and sky similarly decay to `0.1x` during the
canonical-polish suffix while xyz, covariance, opacity and topology remain
frozen.

The v65 lifecycle closes a previously hidden handoff no-op. Earlier
`atlas_residual` runs froze all surface topology, and the mixed-stage surface
ceiling was equal to the already saturated rigid ceiling. A newly exposed
facade therefore had neither a legal birth path nor a free slot. The current
contract partitions the surface by row identity: the handed-off prefix is
immutable to world-space clone/split/prune, while an appended evidence-driven
suffix owns its own topology and 200k quality-profile capacity. Large support
is replaced by area- and opacity-preserving children; an unrelated low-score
primitive elsewhere is never evicted to fund that split. Calibrated DAV2
rigid-hole rows may enter only this suffix as low-opacity hypotheses, while
MASt3R/MAtCha and optional coverage-SfM remain the metric authorities.

The v66 ownership repair closes a separate no-owner failure. In the former
runtime, `object_keep=False` made both rigid RGB and geometry weights exactly
zero, while the pointmap and Chart initializers required all four mask
channels to be true. A persistent clock face in `seq9/frame00048` therefore
remained a clean polygonal alpha hole through 30k iterations despite hundreds
of stable metric projections. The v66 task field transfers ownership
continuously from the noisy mask prior using all-camera positive evidence.
Surface screen/world scale projection now conserves per-primitive
`tau * tangent_area`, and a measured screen-footprint deficit contributes
independent continuous replace-split priority even when blur has suppressed
the centre RGB gradient.

The strict 6k paired prefix selected `mast3r_primary_sfm_coverage`: over a
fixed stratified 64-camera set it improved non-tree reconstruction from
`15.7520 / .7031 / .1328` to `16.0302 / .7170 / .1289`
(PSNR/SSIM/MAE). This is evidence for the coverage stream, not a final quality
claim: both arms were deliberately stopped at 6k and remain below the
historical all-camera rigid target. The selected arm proceeds to the isolated
32k rigid quality run before any new foliage conclusion is accepted.

A later validation exposed a runner/direct-trainer contract regression:
direct rigid experiments inherited the upstream 2DGS position LR (`1.6e-4`
instead of `1.6e-5`) and omitted production geometry-gradient calibration.
Named profiles now resolve these optimizer defaults inside the trainer and
persist their source in the checkpoint. On the same stratified 64-camera 6k
protocol, corrected v85 improved the erroneous v83 non-tree result from
`15.2018 / .6685 / .1412` to `15.9702 / .7164 / .1295`; non-tree gradient
cosine improved from `.5000` to `.6379`. This slightly exceeds the historical
correctly configured v65 6k PSNR (`15.9622`) and validates the repair, while
remaining an early rigid prefix rather than a final 18.756-comparable result.

## Checkpoint and reuse contract

`hybrid_teacher_checkpoint.pth` stores the two Gaussian families, role and
observation metadata, sky and low-capacity appearance/uncertainty,
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
3. role/counterfactual diagnostics, including tree/non-tree, stable tree
   surface, crown, boundary and high-frequency regions.

Only canonical database rendering is comparable to the historical
`18.756 / 0.830 / 0.0821` result. Query64 evaluates reconstruction generalization at known
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
