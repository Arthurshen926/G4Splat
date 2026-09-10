# Position uncertainty and geometry RGB forward repair

## Status

Not accepted as a solved reconstruction. Prior diagnostic canonical48 PSNR
is about15.2dB and gray/blurred dense crowns remain. Rigid contribution and
foreground color/detail are separate errors; an opaque gray leaf is not proof
of continued building transmission.

## v153: fresh position metadata

The new v149 initializer omitted position_covariance. The foliage model then
defaulted to diag(optical_scales²), assigning local optical kernel axes as world
position uncertainty. The production position prior additionally discarded
off-diagonal covariance. This is a newly identified integration bug in the
fresh initializer, not evidence that all historical initializers did the same.

`canopy_position_uncertainty.py` propagates exact-K pixel and log-depth
uncertainty through camera-to-world rotation. Pixel sigma2 is an explicit
conservative assumption, not a measured noise estimate. Log-depth sigma is
max(global scale sigma, per-view rigid log scatter, .025). Optical kernel
thickness is unchanged. The versioned prior uses the full Mahalanobis distance;
legacy unversioned seeds retain their prior for controlled comparisons.

`repair_fresh_canopy_position_covariance.py` wrote a NEW unoptimized seed:
`initialization_v153_measured_position_covariance`. All other tensor fields,
including source rays, colors, opacity, optical scales, centers and witnesses,
are exactly unchanged across1180169 rows. Original artifacts remain intact.
Validation excludes the same48 cameras. Rigid geometry is unchanged.

`formal_v153_measured_position_covariance_3k` runs onGPU1 with the same commands,
initialization geometry, random seed and old geometry-RGB forward asv151. Only
position covariance and the full prior differ. Both retain500/1500/3000 states
with the same30k phase horizon. A small initial library-import failure in the
CPU repair/test invocation was resolved by using the existing conda library
path; it was not a training interruption or CUDA OOM.

New ray births still use the legacy isotropic covariance tied to initial_scale;
that separate uncertainty model is NOT yet repaired. No claim that all
position-covariance pathways are now measurement-calibrated.

## v154: geometry RGB must preserve read-only occlusion

The geometry/opacity isolated branch rendered only the union of live owners.
The appearance branch had already been fixed to retain deployment occluders.
For the measured-leaf profile, deleting read-only foreground rows creates a
different scene and an incorrect RGB residual. No envelope handoff exists in
this fresh initialization to justify that counterfactual.

New code for `static_measured_handoff_quality` keeps deployment static forward
visibility, while exact verified geometry/opacity/SH permissions are unchanged.
The separate SH branch still owns color. Surface/sky are read-only. Legacy
envelope profiles retain their old counterfactual, explicitly distinct.

CUDA fixed-point audit on658/659/686/725/750 uses the current native image as a
synthetic target, NOT ground-truth photography. Current frozen-v145800 model:

|Camera|Live owner rows|Old false loss|Old XYZ gradient norm|New loss/XYZ/opacity gradients|
|---|---:|---:|---:|---:|
|658|19783|.102132|.017503|0|
|659|16115|.137398|.016463|0|
|686|19797|.232772|.013284|0|
|725|14594|.147642|.016987|0|
|750|9315|.178088|.006907|0|

All nonowner gradients are zero. All model tensors remain exactly unchanged.
The audit's raw full-frame RGB max-error field includes unclamped RGB versus
clamped API output outside the scored canopy; the scored photo objective and
its gradients are exactly zero. This is not evidence of a native sorting bug.
Audit directory: `audit_v154_geometry_native_fixed_point_v2`.
The first audit attempt failed because LazyScene had already created the output
directory; it performed no optimization. It is retained, not used as a result.

No v154 training has been launched yet; v151/v153 in memory intentionally keep
their original code. Need a separate matched v154 run and native photo A/B.

Update: two bounded diagnostic400 arms are now launched,
`diagnostic_v154_exact_geometry_native400` and
`diagnostic_v154_exact_geometry_subset400`, onGPU0/GPU1 respectively, each
allocator-limited to25% alongside its existing formal control. Samev145800
start, same304 canonical training cameras/48 excluded, exact verified geometry
owners, XYZ LR.005, scale LR.003, decay to.01, color/opacity/rigid/sky frozen.
Both use canopy photo +.1 oriented multiscale gradient loss and the same bounded
position prior. Only geometry RGB forward visibility differs. These are NOT
production resumes and do not test opacity optimization or the full trainer.
No new formal v154 run yet.

v151500 native canonical48: tree14.516177, rigid15.964987,
hard-boundary13.668172. Hard-region volume contribution alpha.23106. Visuals
660/690 still show gray/blurred crowns and spill at boundaries. This is not an
accepted result. Comparisons to earlier diagnostic starts also change rigid
appearance initialization, so they cannot attribute every rigid-PSNR change
to this foliage repair. Matched v151/v153 checkpoints will use the same start.

Full tests after production forward/covariance repair:1192 passed,15 warnings.

## v155: retained support count blocked fresh-leaf topology

All826005 verified fresh leaves have exactly two actual support-camera IDs and
verified_camera_count2, but cached support_view_count1. The independent witness
context manager retained the second ID without maintaining that cached count.
The canonical lineage eligibility test in `_adapt_volume` consequently rejected
every original generation-zero fresh leaf. Atv151400, all116 actual split
parents were runtime source6 births; no original source4 leaf was split.

`temporary_camera_support` now updates the distinct support count ONLY after a
real new witness is retained. Failed/unknown candidate passes do not increment
it. Fresh-seed validation checks count/table consistency. No camera is invented,
no duplicate slot counts twice, and children still lose verified status until
their displaced centers pass independent verification.

`initialization_v155_measured_position_and_support_counts` is a new copy ofv153
with exactly826005 cached counts repaired and every other tensor unchanged.
The actual `_adapt_volume` regression test confirms stale counts yield zero
splits while corrected counts enable splits, and all emitted children remain
unverified. This is a topology integration bug; it does not itself establish
photo-quality improvement or authorize extra optical mass.

v151500 raw rigid XYZ/scale/rotation/opacity are bitwise equal to the rigid PLY.
Its seed-leaf displacement quantiles50/90/99/max are
.002037/.004494/.006368/.010402 world units. These are measured values, not a
claimed theoretical Adam displacement bound.

v151 was explicitly SIGTERM-requested after its useful500-step control and
bug diagnosis, to freeGPU0 for the corrected mainline. Do not mislabel the
forthcoming supervisor return143 as an unexplained crash. Await atomic stop
completion before reusing its GPU allocation.

Update: v151 has exited with a validated atomic689 checkpoint. The separate
`formal_v155_native_geometry_consistent_support_3k` now runs onGPU0 from the
count/covariance-corrected fresh initialization and native geometry forward.
It retains original learning rates and exact geometry scope as a clean control.
Full tests after the count repair:1194 passed,15 warnings.

## v154/v156 geometry results: scope remains an optimization bottleneck

v154400 from already-refinedv145800: exact-owner native forward15.205683 tree,
16.611648 rigid; old owner-subset forward15.205252/16.611481. The .00043dB
difference is negligible. Fixed-point correctness is not a photo breakthrough.

v156400 starts the SAME unoptimizedv149v2 dense capture (no geometry/color/alpha
warmstart), initial tree13.729836/rigid16.339712. Same native forward, fixed
colors/opacity/rigid, XYZ.005 andscale.003 constant LR, no topology:

- Exact geometry owners:13.749463/16.336130.
- Existing verified leaves in canonical sequence:14.115737/16.276436.

The broader scope improves tree .36627dB over exact, but costs .05969dB rigid
relative to exact. The diagnostic optimizes canopy RGB only; it does NOT contain
the production rigid-pixel negative geometry/opacity constraints. Not accepted
as a final reconstruction or a safe unrestricted permission change.

XYZ displacement median/p90/max: exact .01890/.02825/.05129; persistent
.13270/.23320/.59708. Tangent scale median ratios are1.003/1.003 versus
1.063/1.060. Color and opacity tensors are exactly unchanged, but geometric
footprint changes DO change integrated optical coverage; fixed opacity logits
must not be misreported as fixed image alpha or fixed integrated optical mass.

An explicit `--static-detail-same-sequence-geometry-weight` now exposes the
existing positive-sequence helper, default0 preserving controls. It retains
verified-multiview requirements, cross-sequence blocking and exact topology
ownership. No positive-weight production arm has launched yet.

`diagnostic_v158_fresh_geometry_rigid_balanced400` tests the missing constraint
in that isolated geometry experiment: native visible foliage extinction on
eroded observed rigid interiors, weight3. Samev156 persistent geometry setting,
same start and frozen color/opacity/rigid. Masks define a loss only, never a
render gate. Three tests confirm interior-only support, zero hidden-leaf
gradient behind an opaque wall and differentiable zero without rigid evidence.
Results pending; do not promote before tree AND rigid/native-boundary review.

The covariance-onlyv153 control reached its retained500 checkpoint. A controlled
SIGTERM was requested to stop the obsolete count/forward implementation after
that matched control, preserve an atomic rolling state, and make room for a
verified-sequence production refinement arm. It is not an unexplained failure.

Update: v153 exited with validated atomic512. Its native500 tree/rigid/hard
are14.516868/15.965095/13.668783, versusv15114.516177/15.964987/13.668172.
Covariance correctness alone did not materially improve this low-LR prefix.

v158400 completed: tree13.884374/rigid16.497402, versus same-start
13.729836/16.339712. The added rigid constraint trades some ofv156's tree gain
for an actual improvement in both cohorts. Images still require qualitative
acceptance; the issue is not solved.

`pilot_v159_verified_sequence_rigid_balanced_1k` is a new clean production pilot
onGPU1, not a diagnostic capture resume. It usesv155 initialization and native
RGB/geometry forward, geometry/appearance positive-sequence weights1, native
rigid-alpha weight3, XYZ LR.001, feature LR.0025, scale LR.001; rotation and
opacity LR are unchanged. Exact topology ownership, single-view restrictions,
cross-sequence unknowns and rigid geometry/opacity freezing remain in force.
The pairedv155 original-rate/exact-scope3k control continues onGPU0.
The new pilot retains500/1000 with30k phase horizon. It bundles supported
optimization changes, so comparisons do not attribute its whole gain to one
scalar. Negative loss is on high-confidence observed rigid interiors and is
never a deployment mask. Native negative opacity gradient is restricted by
the existing optical owner; geometry uses the explicitly configured verified
positive-sequence owner. No new positive opacity authority is created.

Latest full tests after production integration:1197 passed,15 warnings.

## Other completed diagnostics

## v160 split-color integration follow-up (not yet quality validated)

v155 topology now actually splits original measured source4 parents:131 at200,
495 at400. This confirms the support-count fix affects production behavior.
At400 all1976 children still had fewer than two accepted color observations;
1730 refreshed from an explicit inherited observation, while legacy metric/ray
candidate counts were zero. The color refresh loader never queried the calibrated
MoGe canopy field, so support fallback could not use the second image in this
MoGe-only pipeline.

The fresh measured profile now passes its task-field provider into split color
refresh. Each canonical RGB sample must match the current calibrated thin MoGe
hit and have a complete bilinear footprint of known, non-transient canopy.
An inherited posterior cannot override a current conflict. Legacy profiles keep
their prior evidence path. This changes initialization of child appearance only,
not XYZ, shape, opacity, surface parameters, rendering gates or support identity.
Distinct camera IDs are now deduplicated across explicit and fallback sampling;
the same image cannot supply two votes or evade the single-view DC rule.

14 targeted refresh tests and1203 full tests pass. `audit_measured_child_color_refresh.py`
compares the current descendants in an immutable checkpoint, restores their
features, and fingerprints geometry/opacity. Audit and native500 evaluation are
in progress. v155/v159 already-running processes retain their launch code and
do NOT include this follow-up. No quality conclusion is drawn from unit tests.

Completed real-state audit `audit_v160_measured_child_color_500_v2`:
1976 current descendants, legacy1731 refreshed but zero with two independent
camera colors; measured1819 refreshed,1215 with at least two distinct cameras.
Geometry/opacity fingerprints unchanged, all feature tensors restored after
the audit. The first audit caught an HxW task-field indexing mistake in the NEW
implementation before any training launch; corrected to reshape against the
depth image and strengthened the fixture with a non-canopy first row. Do not
attribute that development error to historical training.

v155 native500 canonical48: tree14.482111, rigid16.008096, hard13.719655.
Compared withv153500 tree14.516868, rigid15.965095, hard13.668783: not a canopy
improvement. Dense660 leaf alpha.694997 and rigid alpha.279266 confirm substantial
actual background contribution, not merely a color illusion. Dense690 leaf
alpha.945260 and rigid.042409; different views remain distinct failure modes.
500 is an early checkpoint, not a completed3k run.

v148 SH convergence completed normally: SH0 15.261600/16.612285 tree/rigid,
SH3 15.309662/16.612633 at2400 (48 views). Directional capacity adds only.048dB;
visually it does not resolve crown blur. GPU2 was released, then a clean matched
`pilot_v160_measured_child_color_balanced_1k` was launched from the same seed and
same parameters asv159, adding the measured split-color repair and camera dedup.
Actual detached supervisor13653, training13654. No old checkpoint resumed.

v161 opacity-only exact/persistent permission pair freezes geometry and colors,
uses the same native RGB and thin-query forward, and allows increases only on
rows with a current thin-hit gradient. The first pair stopped beforestep1 due
to missing immutable depth profiles in the diagnostic loader; no production
training stopped. The `_v2` pair accepts `--depth-profile-initialization` only
after matching both manifest and foliage-seed SHA256 to the checkpoint, and is
running onGPU0/1 with.30 memory fractions. No uncalibrated fallback is used.

Additional lifecycle audit: v155500 has610 unresolved split families (2436
children), of which2320 children are still unverified. The610 original parents
are in sparse rollback snapshots, not the live render. Native persistence also
requires proposal_kind=NONE, so even independently verified siblings remain
hidden until the family commits. `audit_pending_split_visibility.py` tests
restoring those exact parents in a local diagnostic model, without authorizing
unverified children. It restores all touched tensors afterward. This is a
counterfactual diagnostic, not yet a production fix or accepted quality gain.

v159500 completed native51 evaluation: canonical48 tree14.825734,
rigid16.045138, hard13.969838. Againstv155500: +.343623 tree (40/48 improved),
+.037042 rigid (31/48), +.250183 hard (38/48). The surface geometry/opacity
fingerprints are identical. Tree volume alpha did NOT improve (.724489 versus
.726298 pixel-weighted), and660 remains visibly transparent/gray. This is an
early optimization gain, not completion or a new overall best checkpoint.

v160200 actual split refresh:523/524 children refreshed,451 have at least two
distinct camera colors. This confirms the production integration is exercised.

v162 read-only parent-restoration counterfactual completes at500:
tree14.482111 ->14.484853, rigid16.008096 ->16.007555. The deployment gap is real,
but this early checkpoint's affected mass is small; it is not the principal
explanation of the large remaining canopy leak.

v163 implementation (opt-in, NOT in already running jobs):
`--synchronous-measured-split-verification` validates each new child using its
persisted canonical cameras, calibrated thin depth, known canopy, rigid-front
clearance and existing native color/contribution witness. Whole families pass
or the exact original parent is restored in the SAME topology event, before
the next main render and the single composed optimizer migration. Failed fresh
proposals keep the original parent's Adam history; accepted children get fresh
moments and re-armed ownership ledgers. No renderer change, unverified-child
deployment permission, semantic render mask or image-quality acceptance gate.

Four CPU tests cover full pass, full failure, partial-family failure and mixed
families; tensor equality, visibility, row mapping and capture/restore pass.
`audit_v163_synchronous_split_cuda` tests two actual parents in the full scene:
4 proposed children,3 obtain real witnesses;1 family publishes and1 restores its
parent. Net growth1, no new unresolved family, rigid geometry/opacity unchanged,
single optimizer mapping valid. Synthetic demand is only a causality test,
not a reconstruction-quality result. Latest full suite before the fourth test
was added:1206 passed; final full-suite rerun remains due.

Latest full suite after mixed transaction and coherent depth tests:1209 passed,
15 warnings. v163 remains opt-in; it has not been launched as a production run.

v161 finishes normally after the loader repair:
exact400 tree14.511633, rigid15.981357;
persistent400 tree11.997912, rigid16.455599;
baseline14.482111/16.008096. Broad opacity-only optimization is rejected: it
removes crown coverage while cleaning rigid pixels. Both arms keep geometry,
colors and rigid parameters fixed and gate positive changes with the RAW thin
hit derivative, not a deficit-loss derivative. This is evidence of a coupled
geometry/opacity problem, not a reason to raise optical permission unconditionally.

v164 `diagnostic_v164_coherent_source_depth400` starts onGPU0 from the same
unoptimizedv149v2 diagnostic capture asv158. It optimizes one log-depth factor
per source camera for already persistent rows only, LR.002, bounded +/-.2.
Centers move along source bearings and all three Gaussian scales change by the
same factor, preserving source-camera projected conics. Colors/peak alpha and
rigid parameters stay fixed. Metric integrated mass is NOT fixed under this
scale change. Native geometry photo/HF plus rigid-interior alpha weight3 are
unchanged fromv158; no training or inference image mask is a render gate.
This is a new optimization parameterization and bound, not a pure one-LR
comparison. Unit tests verify projected bearing/footprint invariance,
nonowner identity and the grouped XYZ+log-scale chain rule against autograd.
No result or geometric correction is assumed before evaluation.

An immutablev1551000 checkpoint was additionally retained after checking its
embedded iteration, and render-only extraction completed. Its native evaluation
andv160500 extraction/evaluation are next; thev1553k control is not stopped.

Completed early results:
- v1551000 tree14.647769, rigid15.590739, hard13.177147. Tree alpha.802720
  (vs.726298 at500) but hard-pixel foliage alpha.300299 (vs.226345): growth
  still spills into rigid pixels. This is not acceptable reconstruction quality.
- v160500 tree14.825484, rigid16.045207, hard13.969457. Essentially identical
  tov159500; successful color refresh counts alone do not establish a visual
  gain. Both500 states still predate widespread child-family publication.
- v164 has entered real optimization (step175 log-scale range-.1004..+.0903,
  median-.00934). No parameter hit the +/-.2 bound then. Outcome pending.

v150 color-only800 fromv145800: exact camera permission15.201663 treePSNR,
persistent canonical permission15.229723, rigid16.611588 versus16.600193.
The .028dB tree gain is small, not a breakthrough or justification for arbitrary
geometry/opacity authority expansion.

Three-view flow audit `audit_v152_three_view_static_consistency` retains only
forward-backward, triangle-cycle and local-NCC consistent correspondences.
582/583/584 canopy epipolar median2.289px vsrigid.302;725/726/727 canopy1.932
vsrigid.315.686/687/688 canopy.377 with no valid rigid controls. Local
nonstatic/correspondence problems remain plausible; optical flow is not motion
ground truth and does not authorize dropping frames or explain all crown blur.

v164 completed normally (supervisor exit0,400steps). Canonical48 treePSNR
13.729836 ->13.754597, rigid16.339712 ->16.440906, hard14.337400
->14.366528. Source-camera coherent depth/footprint transport is not a
breakthrough: the660 native image still has severe background transmission.
Do not promote this diagnostic transform into the fresh initialization.

v159 reached1000 and wrote its immutable checkpoint. At10:31 the child was
still in CPU-heavy finalization, not another interrupted optimization run.
The supervisor still correctly reports running until finalization exits.

v159 subsequently completed with exit0. Native1000 evaluation: tree15.177876,
interior15.629696, boundary13.319877, rigid15.713119, hard13.741587.
Relative tov1551000 this is+.530107 tree,+.122380 rigid,+.564440 hard.
However versus its own500 rigid worsens.332019dB and hard worsens.228252dB;
660 remains visibly translucent. This is not a successful final reconstruction.

`formal_v163_atomic_measured_split_balanced_3k` launched onGPU1 afterv159
normal completion and evaluation release (supervisor31452,child31453).
It usesv160's exact seed/configuration plus synchronous split verification,
3000steps with30k phase horizon, retained500/1000/1500/3000. Launch sources
archived. No new rendering gates or rigid geometry training are enabled.

`diagnostic_v165_native_opacity_no_thin_gate400` launched onGPU0 alongside
the control (supervisor31226,child31227,memoryfraction.30). Matched tov161
persistent400_v2 except removing `--require-thin-growth`. Only verified
persistent leaf opacity is optimized, using native RGB and unchanged rigid/
boundary/sky preservation. Geometry/colors/rigid tensors remain frozen.
This tests the asymmetric positive-update restriction; it is not a proposal
to deploy unconstrained opacity or infer trustworthy depth from RGB alone.

## Export-only rigid freeze violation

v159 result.json confirms final cleanup removed388 rigid rows despite
appearance_only policy (1006817 ->1006429), with no retired replacements.
Final cleanup had no mature-surface authority filter. Added a policy filter:
appearance_only/frozen permit no deletion; atlas_residual only its mutable
suffix; joint retains existing pruning. Three regression cases plus four
atomic-split cases pass. Chart baking is left intact because it represents
the native atlas geometry, rather than being a topology decision.
This does NOT explain checkpoint-based canopy metrics, and the already
runningv163 process predates this export repair. Its checkpoint remains the
evaluation authority; do not claim its final export includes the new guard.

Full suite after export guard:1212passed,15warnings. Native layer audit
`audit_v159_dense_layers_1000` (unchanged five posthoc ROIs):
657 leafalpha.878625/rigidalpha.101105/PSNR17.8560;
660 .840007/.148927/17.1836;
690 .988609/.008051/26.5232;
713 .940328/.053839/19.4180;
768 .987282/.012474/19.9687.
The660 crown still leaks substantially, while690 is already predominantly
opaque and lacks fine texture. One global thickness criterion cannot explain
both. Layer decomposition matches the native API within1.2e-7.

Runtime note: v1551400 topology maintenance took several minutes and then
advanced normally. An isolated temporary py-spy installation could not inspect
the live stack because this container lacks SYS_PTRACE; no signal was sent to
the trainer. v1601000 trace appears before its large checkpoint is published:
an early extraction saw FileNotFoundError, not a failed trainer/checkpoint.
Wait for the immutable checkpoint before retrying extraction.

v165 completed normally: native opacity-only400 without thin-growth gate
tree14.825822/rigid15.997647/hard13.715423 versus same starting state
14.482111/16.008096/13.719656. The matched thin-gated persistentv161 arm
gave11.997912/16.455599. Thus asymmetric clipping is damaging in this
controlled opacity refinement, butv165660 remains gray/translucent.
Important: production's evidence-trust-region gate is a LATE settle policy,
not active at500/1000. Do not misattribute early production failures to that
late policy. Early production instead keeps RGB optical permission exact
because same-sequence-optical-weight=0. The existing verified same-sequence
optical option can test that scope separately, without rewriting the renderer
or promoting unresolved/single-view rows.

v163200 actual production transaction:131attempted parents/524children;
469children individually verified,100whole families (400children) published,
31families restored their exact parent, netgrowth300. No new unresolved
family. This verifies live publication, not its eventual quality.

v160 completed normally. Native1000 tree15.178099/rigid15.713068/
hard13.739818: effectively identical tov1591000. No material color-refresh
quality gain through1000.

v165 dense660 rigidalpha.279266 ->.177302, treealpha.694997 ->.809831,
PSNR15.4159 ->17.3035;657 rigidalpha.20632 ->.12168. Still leaky.

`pilot_v166_verified_native_optical_balanced_1k` launched afterv160 normal
completion onGPU2 (supervisor4951,child4952). Samev163 seed/configuration,
but same-sequence-optical-weight=1;1000steps,30k horizon,retain500/1000.
This uses the existing verified positive-sequence permission helper, not an
unverified growth bypass. Native rigid extinction3 remains active and now
shares that same optical permission. Training code also includes the
export-only frozen-surface guard added afterv163 launch; checkpoint-render
comparisons are unaffected by that export guard. Sources archived separately.

v1551500 native evaluation: tree14.602125,rigid15.233300,hard12.804533;
hard foliage alpha.353228. Both tree and rigid declined since1000. The
control was deliberately stopped after preserving1500; SIGTERM reached its
safe checkpoint boundary at1600, supervisor exit143/maxrestarts0. This is
an explicitly rejected control, NOT a completed3k run or spontaneous crash.
See its CONTROLLED_STOP.md and interrupted_training_state.json.

## v167 joint native refinement diagnostic

Added an explicit `--refine-checkpoint-leaves` diagnostic mode: matched
initialization and seed hashes supply immutable calibration; no replacement
capture, opacity-only replay, or implicit fresh-seed claim is allowed. Output
remains diagnostic/nonresumable. Added independent native-rigid-alpha loss
using the existing eroded-rigid extinction helper for joint xyz/scale/opacity;
it never changes deployment visibility. Foliage colors retain canopy-only
gradients.13targeted validation/extinction tests passed.

`diagnostic_v167_joint_native_rigid_extinction800` launched onGPU0 afterv155
released it (supervisor7824,child7825). Sourcev1591000,800steps/eval400/800,
opacityLR.02,XYZ.005,scales.003,SH3colors.0025,SSIM.2,anneal.1,
native-rigid-alpha3,sky.1. Rigid geometry/colors/opacity and sky are frozen;
no preservation toward the already-contaminated mixed input is imposed.
All304canonical training cameras,48excluded. This is a bundled joint
optimization diagnostic, NOT a single-variable causal comparison or a clean
production training. Diagnostic scale/opacity bounds differ from production;
any successful result needs a matched production integration audit.

Full suite including checkpoint-refinement validation:1222passed15warnings.
`diagnostic_v168_joint_native_rgb800` is the matchedv167 control with only
native-rigid-alpha-weight=0 (GPU0 fraction.30 instead of.60 for coexistence,
no numerical batching change). Supervisor9644/child9645; archived separately.
Same checkpoint, losses, LRs, camera order, and800step annealing schedule.
Both retain real rigid RGB loss; neither trains rigid parameters or permits
leaf-color gradients from rigid pixels. This pair isolates the extra visible
foliage extinction term within the joint diagnostic.
