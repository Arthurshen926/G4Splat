# v132–v134: material-local evidence and geometry revalidation

Status: diagnostic experiments, **not an accepted solution or production checkpoint**.
All quality numbers below are mean per-view PSNR over the preselected canonical48,
640-pixel native rendering. These 48 cameras are excluded from fresh seed creation,
witness selection and diagnostic optimization, but were not held out from historical
training. No ground-truth mask or MoGe depth gates the reported native render.

## Completed comparisons

| Experiment | Steps | Canopy | Rigid | Rigid boundary |
|---|---:|---:|---:|---:|
| v125 source | 2k | 10.93692 | 16.65432 | 14.80528 |
| v127 formal control | 3k | 11.20382 | 16.65624 | 14.71997 |
| v132 calibrated 8192/source, local XYZ/DC/scale | 1200 | 15.00965 | 16.60804 | 14.56316 |
| v133 matched color control, opacity-only continuation | 400 | 14.57510 | 16.64817 | 14.66554 |
| v133 material-local color correction, matched continuation | 400 | 14.58458 | 16.65868 | 14.69610 |

The local-refinement improvement is substantial relative to the old source, but
view657 still has blurred foliage, missing crown detail and boundary contamination.
Frozen rigid parameters do not guarantee unchanged rigid-pixel appearance when
foreground leaf geometry changes. This is **not** quality acceptance.

The isolated color correction yields only +.00947 dB canopy and +.03056 dB rigid
boundary. It fixes a real observation-path error, not the primary geometry deficit.

## Confirmed material evidence errors fixed

The old verification RGB accumulator mixed canopy and background RGB and divided
by total primitive responsibility. Sky was absent from the competing-material test.
The new seven-channel audit carries canopy, rigid, topology, canopy*RGB, and known
sky. Actual canopy mass must exceed 1e-5 and exceed rigid+sky mass. Color divides
by canopy mass. Legacy unmasked RGB cannot recolor mixed footprints. Geometry
reverification can explicitly preserve learned color (`update_color=False`).

Tests: `pytest tests -q --disable-warnings` completed with 1128 passed /15 warnings
after the material changes. The later revalidation script and builder-loop fix
require their own checks; this count is not a claim that every later edit was tested.

## Bearing-only search is not a rendered visual hull

`diagnostic_v134_anchored_bearing_pool2048` retains 373,637 observed canopy rays,
including rays whose monocular depth would previously put the seed behind a known
opaque wall. Their clamped search hints are **not measured depths**: observation
depth stays NaN, verification is unverified, and the pool cannot be loaded as a
trainable/renderable replacement capture.

One location per ray is selected from independent evidence, with a source-wall
upper bound and no layering of depth uncertainty. Tangent scales can preserve the
source angular footprint when the selected depth changes; physical depth thickness
does not increase.

- First supported silhouette hit: 212,501 selected, 209,594 native verified.
- Patch-stereo hit: 18,591 selected and native verified.

Both enter separate 400-step appearance/scale/opacity diagnostics, with zero
positive MoGe thin-hit weight to avoid forcing their different depth evidence
back to the monocular prior. Counts alone do not establish improved reconstruction.

## Reverification after geometry refinement

`scripts/reverify_refined_canopy.py` collects witnesses without changing the input
scene between cameras. XYZ, scales, quaternion, opacity, appearance and candidate
support table are fingerprinted unchanged. Only after all cameras have rendered
the same frozen scene are witness tables updated. Known opaque rigid depth is
rendered independently; actual canopy responsibility must dominate rigid+sky.

The existing-camera-only audit of v132 local1200 retains 715,906 of 847,941 old
verified leaves, rejecting 132,035 (15.57%). This does **not** establish that all
rejected leaves lack support: historical camera IDs are only a narrow candidate
set. A matched all-canonical audit checks real contributions in all 304 eligible
training views and allows new actual camera witnesses, with no held-out views.
It does not grant authority to previously unverified leaves. Both outputs remain
diagnostic and require native quality evaluation after any visibility change.

Completed follow-up: all-canonical actual witnesses retain **840,389 /847,941**
old verified leaves (99.11%); only 7,552 fail to find two witnesses. The narrow
existing-camera-only removal renders at 14.69395 /16.62868 /14.62885, losing
.31571 dB canopy relative to the unmodified local1200 scene. Thus historical
camera-table restriction is not an appropriate general-purpose retirement rule.
The all-canonical post-audit render is being checked separately.

Completed all-canonical render: **15.00453 /16.60807 /14.56331**. The .00512 dB
canopy change is negligible; additional retirement is not a demonstrated solution.

`audit_v135_local1200_rgb_gradient_conflict` decomposes actual native opacity
gradients on 12 uniformly spaced canopy-bearing training views (not the validation
views). Canopy RGB wants opacity growth on 406,479 rows in the aggregated gradient;
rigid RGB + frozen rigid/boundary reference + sky terms reverse 40,668 rows, accounting
for **3.5313% of canopy growth gradient mass**, cosine -.02080. This is not evidence
that rigid protection broadly suppresses the crown. It excludes MoGe and position
gradients and should not be generalized beyond the measured components/views.

Two matched continuations of the unmodified local1200 learned capture now isolate
the remaining positive-depth prior: `diagnostic_v135_local1200_thin_prior_800`
(.02 thin-hit weight) and `diagnostic_v135_local1200_rgb_geometry_800` (zero).
Everything else remains matched, including rigid/sky negative evidence, frozen
rigid reference, train/validation cameras and geometry/color/opacity optimization.
These reset-optimizer diagnostic continuations are not exact training resumes.

## v135 performance causal check and v136 depth parameterization

Zero-weight thin-hit queries previously still built a full million-leaf CUDA
autograd graph. They are now skipped unless needed for explicit growth permission;
empty masked queries are skipped as well. Logs mark skipped readouts as unmeasured,
not fabricated zero coverage. Matched old/archive vs new two-step CUDA runs from
the same first-hit capture have identical XYZ/quaternions and maximum learned
opacity/DC/log-scale differences <=2.3842e-7. This is floating-point agreement,
not a claim of bitwise equality for atomic rasterizer gradients.

Source inspection confirms the optical hit query uses hard center-depth interval
membership. Its derivative adjusts projected opacity/conic/mean, but does not
directly pull an out-of-interval leaf's depth into that interval. General XYZ RGB
updates can also drift laterally off the original measured pixel bearing.

`diagnostic_v136_source_ray_depth1200` isolates a scalar per-leaf displacement along
the original camera ray, starting from the same calibrated8192 v2 fresh seed as
v132. Exact source-bearing alignment is checked before optimization; already
off-ray refined captures are rejected, not silently snapped. Native XYZ gradients
are contracted with the unit ray by the chain rule and a true scalar Adam parameter
is optimized, with the same bounded displacement norm as the free-XYZ control.
Inactive points and rigid parameters cannot move. No depth uncertainty is converted
to physical leaf thickness. Source-pixel invariance, rejection, bounds and scalar
gradient equivalence tests pass. Quality remains to be measured.

CPU source-bearing audit establishes a concrete association drift in v132 free
XYZ refinement: median **6.00036 pixels**, 90th13.68545, 99th27.70085, maximum65.57643;
**90.9922%** of verified rows moved more than2 pixels from their original observation.
The corresponding raw seed reprojection errors are below .003 pixels. The camera
container has 1487 images /1487 unique intrinsic IDs, so this is not caused by
duplicate camera identities in this dataset. Script and output:
`scripts/audit_canopy_source_bearing.py`, `audit_v136_free_xyz_source_bearing.json`.
Semantic region reverification does not establish persistence of the original
pixel/color correspondence. This makes the ray-parameterization ablation motivated,
but does not by itself prove that restricting motion improves reconstruction.

The same free-XYZ run is not generally hitting its bounds: verified displacement
norm/radius quantiles50/90/99 are .13677/.28449/.55229; only .01663% reach the cap.
Only .11982% of tangent axes reach their scale ceiling; none reaches the floor.
Increasing those bounds is therefore not supported as the main solution.

At v136 step400 the same CPU check confirms preserved bearings: median .000093
pixels, maximum .002284 pixels; zero verified rows drift more than2 pixels. This
confirms the scalar parameterization implementation, not final canopy quality.

## v137 structural-detail experiment

The new CPU saved-pair audit reports regional 11x11 interior SSIM and first-gradient
error/strength over the same48 views. Numbers are **8-bit PNG-derived**, separately
labelled from float native PSNR. Complete canopy windows are required for interior
SSIM; gradient error accompanies strength so false building texture is not rewarded.

| Model | Canopy interior SSIM | Gradient MAE | Gradient strength / GT |
|---|---:|---:|---:|
| v1252k | .139419 | .061242 | .805322 |
| v132 free XYZ1200 | .272093 | .051291 | .456506 |
| v133 dense16384 step400 | .263389 | .051529 | .476725 |

These support improved structure but substantial oversmoothing, not resolution.
`diagnostic_v137_source_ray_structural1200` starts from the same fresh seed and
matches v136 except for a .2 canopy SSIM loss. Every accepted 11x11 training window
lies entirely in known canopy. Unit tests show exact zero pixel gradient outside
that mask and zero supervision for empty/thin regions; no rigid or unknown windows
are borrowed. This does not imply that a shared leaf update cannot affect other
pixels/views, so frozen rigid-reference / boundary / sky losses remain unchanged.
Native rendering remains unconditioned and mask-free.

## v138 diagnostic optimizer parity correction

The production `_volume_optimizer` explicitly uses Adam `eps=1e-15`; the standalone
calibrator accidentally relied on PyTorch's `1e-8` default. This can damp small
per-primitive updates in a million-leaf model. The calibrator now exposes an explicit
`--adam-eps` (default matching production) and reports actual opacity-moment epsilon
damping. v132–v137 existing launches remain their original **1e-8 controls**; they
are not retroactively described as fixed. This is a diagnostic configuration mismatch,
not evidence that the production optimizer had the same epsilon bug.

`diagnostic_v138_source_ray_production_adam1200` matches v136 initialization, source-ray
optimization, losses and learning rates with eps1e-15. It evaluates the same48 native
views at0/400/800/1200; it skips only the extra globally scaled thin-depth readout,
which is not a valid calibrated-coverage metric for the new per-view source model.
This avoids redundant diagnostic work without removing RGB, alpha, rigid or boundary
validation. Final quality impact remains unmeasured.

Follow-up outcomes (canopy/rigid/hard): v133 dense16384 at1200 is
**15.04292 /16.59841 /14.55820**, only +.03327 canopy relative to calibrated8192
free-XYZ1200. v136 source-ray1200 is **14.90599 /16.61588 /14.62137**: better boundary,
slightly lower canopy PSNR, not a breakthrough. At matched400, eps1e-15 v138 is
14.77965 /16.62279 /14.67676 versus v136 eps1e-8 canopy14.77790: epsilon parity is
correct, but does not demonstrate a meaningful canopy-quality improvement here.

## Native-layer diagnosis and v139 fixed-transport color solve

`audit_v138_local1200_native_layers` keeps geometry, opacity and occlusion ordering
unchanged and zeros one material's colors at a time. The two RGB contributions sum
to the original within4.3e-6; the reconstructed result matches the deployment API
within1.2e-7. Restored material tensors are fingerprinted unchanged.

Canopy mean volume /rigid contribution alpha:

- 657: .66138 /.30787; rigid RGB contributes .13288 mean, leaf RGB .11954.
- 660: .67022 /.31630.
- 690: .87752 /.08282.
- 713: .82700 /.14009.
- 768: .54032 /.45323.

These object masks also include sparse/bare branches, where genuine gaps should
remain visible. Do not force the entire mask opaque. A manually selected dense
interior ROI in view657 (x470:630,y40:200; chosen after inspecting GT, not a fixed
acceptance region) has PNG-derived mean alpha .96649. The conditional leaf-color
panels still show gray/blurred appearance there. Thus true background transmission,
boundary displacement and foreground color/detail errors coexist.

The separate `audit_v138_local1200_foreground_darkening` applies a foreground-layer
darkening ratio only under an opaque-wall / intrinsic-versus-native-alpha
applicability check. It is not a general matting oracle and is **not used as a
training loss**. In657/660, about16.2%/14.0% of applicable pixels have alpha deficits
>.05 under this conditional bound. No unconditional mask-opacity target was added.

`solve_canopy_material_colors.py` holds geometry, opacity and rigid colors fixed,
and solves the linear leaf-color subproblem with native transport responsibilities.
For transport W and nonnegative pixel weights mu, D=W^T(mu*row_sum(W)) majorizes
W^T diag(mu)W. A simultaneous projected color step using this diagonal is tested
to decrease the fixed-transport weighted least-squares objective; the real run also
checks the objective after every global update. All304 training views contribute
at the SAME colors before an update. No small-Adam-gradient approximation or
target-view render mask is involved. Three color updates plus a final objective
check are planned; nonnegative colors cannot grow beyond max(1, initial color).

Two matched v139 arms start from free-XYZ1200:

- `diagnostic_v139_color_transport_full3`: original measured-single leaves remain
  visible only to their source training camera, but their colors stay frozen.
- `diagnostic_v139_color_transport_global3`: excludes nonverified/private rows from
  the diagnostic model, so all training renders use only the shared global map.

Their initial48-view native metrics are exactly identical. This isolates whether
source-camera private capacity absorbs errors during global fitting. It does not
yet prove that it is the dominant failure mode. Both run via the existing detached
supervisor in separate `*_supervision` directories, with max-safe-restarts0 (a
diagnostic capture must never be passed off as a resumable production checkpoint).
Native alpha is frozen in this color solve, so even a large PSNR gain will NOT by
itself establish that geometric background transmission has been solved.

## v140 material-gradient parity correction

Production `_branch_isolated_rgb_losses` and the intrinsic-color stream already
separate leaf appearance observations from known rigid pixels. The standalone
calibrator, however, differentiated its rigid RGB/reference losses with respect to
leaf DC as well as geometry/opacity. The latter can recolor a misplaced leaf to
camouflage a wall violation, conflicting with the material-ownership principle.
This is an **additional diagnostic-versus-production mismatch**, not a claim that
the production branch has the identical defect.

The calibrator now extracts appearance gradients from canopy RGB/structural loss
only, while preserving full rigid/sky gradients for leaf geometry and opacity.
The legacy behavior remains explicit via `--allow-rigid-leaf-color-gradients`.
Unit tests demonstrate that the rigid color gradient is removed without changing
the optical gradient. `diagnostic_v140_source_ray_material_color1200` matches v138
except for this appearance ownership correction.

`diagnostic_v140_color_transport_canopy3` additionally compares the fixed-transport
color solve with canopy-only appearance observations against v139_full3. Geometry,
opacity and rigid tensors remain frozen; any newly exposed rigid contamination is
measured and must subsequently be resolved by geometry/optical ownership, not by
painting foliage to look like the facade. Neither arm is an accepted final model.

## Small diagnostic builder bug

A camera with no eligible measured-single candidate previously `break`-terminated
the entire independent-witness loop. That condition is camera-local: later views
may still verify the remaining leaves. Changed to `continue`. This is a boundary
case correction, not a retroactive explanation for the measured multi-source gains.

## v139/v140 completed color-only controls and v141/v142 evidence tests

All following values are means over the same prespecified48 native views.
v139 full3 completed with exit0 and its true full-training quadratic objective
decreased41.76868 ->38.94637. Final tree/rigid/hard is
15.16566 /16.64737 /14.71803. Removing private rows gives
15.16590 /16.64420 /14.71104, effectively the same. Source-private capacity is
therefore not demonstrated to be the dominant bottleneck of this fixed-color
subproblem. v140 canopy-only colors gives15.27645 /16.56429 /14.51640:
slightly better crown colors but worse rigid boundaries. Do not accept this as
an optical fix: all three color-only arms keep alpha/geometry unchanged.

v137 source-ray+SSIM1200 completed14.80252 /16.61636. v138 production-epsilon
source-ray1200 completed14.89933 /16.61578. These fail to improve the free-XYZ
baseline's canopy PSNR. v140 source-ray material-color optimization is still
running; its zero-step metrics must not be reported as its trained result.

v141 tests the remaining depth-search limitation: a bearing pool explicitly has
NO observed source depth, yet its old search only explored +/-0.30log around the
untrusted proposal. The new optional source-wall sweep spans the finite front
interval, chooses ONE supported location, and retains local search for unknown
walls. It does not authorize a volume filled across uncertainty. The first
completed pool selected282088 rays /236260 native-verified. This run also exposed
a near-plane mismatch: candidate projection accepted z>.0 and camera znear=.01,
but native auxiliary.h in_frustum culls z<=.2. The corrected
`diagnostic_v141_full_wall_front_native_near2048` aligns candidate visibility to
the native threshold; the earlier result is retained as an audit, not accepted.

v142 `diagnostic_v142_foreground_contrast_dense8192` tests whether coarse object
masks authorize false material in real branch gaps. Optional foreground-positive
evidence requires an RGB discrepancy above the frozen background model's own
eroded rigid residual90th percentile+.02. Insufficient anchors preserve unknown
status/the original candidate policy; weak contrast NEVER becomes free-space
negative evidence. Per-view calibration profiles are persisted and reused.
This is a conservative proposal/witness diagnostic, not a matting oracle, depth
measurement, blanket opacity target, or render-time semantic gate. Its effect
on native reconstruction still needs matched optimization and48-view evaluation.

Project tests after adding the evidence and wall-grid helpers:1151 passed.

The corrected v141 full-source-wall search still did not improve native quality:
`diagnostic_v141_wall_front_appearance400_640` finishes14.37234 /16.55683 versus
the old local-bearing400 tree14.43687 /rigid16.63002. Greater admissible search
coverage is not by itself better geometry; do not adopt it as a successful fix.

The first launches of v141 appearance400 and both v1421200 arms accidentally
omitted resolution640, inheriting ModelParams=-1. They were intentionally
SIGTERM-stopped, with logs/partial outputs retained and exclusion notes. No OOM
occurred. Correct comparisons use only their separate `_640` replacements.
The calibrator now defaults640 and all subsequent launch commands specify640.
v142 raw-control zero-step metrics again match the previous13.79209 /16.33170
exactly; the contrast seed starts13.10044 /16.36683 with1091776 leaves,
686737 verified. Its final optimization result is still pending.

## v144 physical darkening and v145 convergence controls

`foreground_darkening_bound` derives a necessary extinction lower bound from
nonnegative foreground radiance over an opaque frozen background. The diagnostic
loss requires known opaque wall,7x7 canopy interior, and intrinsic wall-front
alpha matching native volume contribution within.01. Its per-view slack is the
95th percentile of maximum-channel background error on eroded visible rigid
pixels+.02. Missing512 anchors disables this extra bound, not negative evidence.
This conditional model is not a general matting oracle; ambiguous ordering and
real gaps receive no forced opacity target.

Two v144400 arms start from the SAME v140 canopy-only color-solver epoch3:
`diagnostic_v144_material_then_optical_control400` and
`diagnostic_v144_material_then_darkening400` (.05 extra weight). Only opacity
may change; positions, scales, colors and rigid state remain frozen. Both retain
native RGB/rigid/sky negative constraints, with MoGe positive weight0. CUDA
integration test shows Adam increases only the actual wall-front leaf; the
behind-wall leaf and surface stay unchanged. Real training step50 has7365
applicable pixels /1480 extinction deficits, with background-error slack.25527.
Actual metric improvement is still pending; this is not a completed optical fix.

v145 compares800-step continuation of free-XYZ1200 under identical new defaults,
with either constant LR or exponential decay to.01 of each initial LR. Both
reset optimizer/bounded-motion references as declared diagnostic continuations,
not resumable production training. This tests residual stochastic fitting error
before adding further capacity. A low-order directional-SH diagnostic option is
implemented/tested but has NOT been launched or adopted; default stays DC-only.

`audit_v144_canonical_epipolar_consistency` is a read-only training-neighbor RGB
check independent of monocular depth. For forward/backward-consistent textured
flow matches, pair640/641 has median Sampson distance2.203px in canopy versus
.226px rigid;658/659 has1.768 versus.203. Other pairs are much better, while
905/906 is unreliable even on rigid (43.36px). These observations flag possible
motion, correspondence ambiguity or camera error; they do NOT establish that
wind is the cause or justify abandoning all canonical canopy supervision.

## Remaining acceptance limitations

- MoGe scale correction does not guarantee correct canopy surface depth.
- Larger seed count can increase initial rigid contamination; density alone is not
  a valid fix.
- Native crown detail and boundary quality remain insufficient.
- New calibrated initialization is still diagnostic, not integrated/accepted as
  the production clean-training initialization.
- No claim of fully solved background transmission is justified yet.
