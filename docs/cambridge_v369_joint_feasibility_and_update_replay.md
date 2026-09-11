# Joint feasibility and actual-update work packages (2026-09-11)

Attachment read in full. Its recommendations are reference material, not
automatic authority to weaken building protection or use evaluation images as
training targets. Work remains in codex/moge3-static-canopy-v119, initially
75c6250. Physical GPU0 excluded. No production map or renderer replaced.

## B: continuation and actual update

Added strict version-2 diagnostic continuation into a new output directory.
It preserves schedule horizon, parameter layout, camera order, RNG (Torch,
CUDA, Python, NumPy), fixed candidate geometry/bounds and source eligibility.
Old version-1 snapshots are explicitly not accepted as exact resumes.
Controlled stop is at an evaluation/complete optimizer boundary; step count
still means training images. Reinitialization currently regenerates and checks
the seeds before loading, so restart is correct-by-contract but not fast.

v369 continuous8 and v370 stop4/horizon8 completed successfully. v371 launcher
was rejected before training because the external checkpoint must use the
supervisor's --initial-resume-checkpoint, not an arbitrary base-command resume.
v372 used that supported route and completed4->8 successfully. No OOM.

v369/v372 final mean metrics agree, but tensors are NOT bitwise identical:
12/33 tensors exact; maximum candidate position-code difference .000255227,
source position-code .000055998, source opacity logits .000005007. Adam moment
differences about 6.37e-12 or smaller. Independent CUDA runs have atomic
reduction nondeterminism; no claim yet of bit-exact interruption recovery.
Need a matched checkpoint-origin/repeated-resume comparison to separate this
from restoration defects. Report compare_v369_v372_resume.json.

ActualUpdateAudit records gradient dot actual delta, zero-gradient rows moved,
and stable softplus(logit) optical-depth increments for raw Adam and final
postprocessing. Optional training flag audits the same rendered view/objective
after the update and verifies an independently reconstructed before-loss equals
the actual declared loss exactly. Only full-objective single-view gradients
supported, fails closed otherwise. CPU checks pass. v376 audited8/v377 control8
are running; no quality or optimizer-bug claim from their launch.

## A: conditional sparse multi-view feasibility

Independent CUDA extension exports its native volume preprocessor outputs
(projected means, conics, depths, tile radii). Original installed extension is
untouched. CPU sparse construction reproduces tile getRect truncation and
mean-depth ordering; single-volume native alpha/depth-exclusion test passes.
Independent native suite now4 passed. This submodel omits alpha cap/cutoff,
multiple background surfaces and any full-volume ray-integral interpretation.

v373:16 TRAINING views,112 eroded-tree rays and121 visible-wall rays. Wall
constraints require source pure-surface RGB agreement plus two neighbor
training-view rigid/depth agreements; still conditional on existing geometry,
masks and camera correctness, not independent depth truth. No 48 evaluation
view selected training rays. Intervals use RGB tolerance .03, not a fixed
canopy opacity target. Needs tolerance and sampling-density sensitivity.

Fixed support permits a zero-slack solution for this SMALL subset.52,782
variable columns,81,476 entries. Frozen-source violations zero. Maximum omitted
kernel sum1.358e-6. Candidate tau cap -log(1e-6), existing persistent cap
-log(.005); unsupported existing leaves frozen. No conclusion of full-scene
feasibility from this underdetermined sample.

v374 lexicographically preserves minimum slack, then minimizes L1 change from
the current tau:74 changed rows, total abs tau change17.18765, max2.39009.
v375 native new-kernel before/after, all48 evaluation views plus training rays,
completed0. Mean evaluation tree+.000281, rigid-.000450, hard-.001575 dB.
Selected tree rays above the actual black-foliage RGB floor tolerance24->13;
their mean maximum-channel RGB error .063274->.052914. Visible-ray mean error
.026482->.024635. This is a diagnostic counterfactual, not an adopted fix.
Native attenuation versus single-plane prediction differs by up to .01579 RGB;
zero LP slack does NOT mean the actual renderer satisfies every constraint.
Parameter mutations are temporary process-local counterfactuals, restored in
finally; no teacher/candidate checkpoint is exported.

Next: denser genuinely violated training rays and matched visibility risks;
separate single-background approximation error before declaring global optical
feasibility. Combine this with actual-step replay, not ten unrelated recipes.
The dense-canopy see-through remains UNSOLVED.

## Follow-up results

v376 audited8/v377 control8 both completed0; final mean metrics exactly equal.
All8 audited updates reduce the actual same-view objective; raw Adam and final
gradient-dot-delta remain negative. Many zero-gradient rows still move through
ordinary Adam momentum (step8:289168 candidate rows,462641 source opacity rows).
This is measured behavior, not proof of damage or a standard-Adam bug.

v380 resumes the EXACT v376 step4 checkpoint (not an independent initialization)
and completes4->8. Frozen geometry/bounds/IDs match bitwise; mean metrics exactly
match. Maximum position-code difference6.71e-6, source position4.90e-6, candidate
logits1.00e-5. Initial v369/v370 already differed BEFORE interruption (candidate
position1.139e-4). These controls support numerical continuation, not bitwise
deterministic CUDA. No false claim that remaining differences are all diagnosed.

v381 uses saved v363800 optimizer state, all trained parameter groups, exact
declared objective and original postprocessing. Eight independent one-step
counterfactuals (872,681,746,906,619,611,904,598) all lower actual same-view loss;
none reverses total descent after postprocessing.39–49万 candidate rows move
with zero current-view gradient per probe. No full-training-objective descent
claim; no model saved. RowActiveAdam is now an optional diagnostic-only policy,
and unsupported saved-Adam replay rejects it. v382 full-scale8-step causal smoke
is running with actual-update audits; first4 steps have zero inactive opacity
movement and decreasing loss. Need matched full training before retention.

v378 dense TRAINING-only probe:1724 rays,97365 variables,394716 nonzeros.
Only two residual constraints produce total slack .8825165. Removing all visible
wall constraints leaves the SAME slack; both rays are individually undercovered
at tested bounds:688(214,353) has6 candidate tails sumG .004381;
689(331,343) has16 candidate tails sumG .076751. No persistent foreground
support on either ray in this submodel. Thus this sample does not demonstrate
an added multi-view tree/wall contradiction. Still not full-scene feasibility.

v379 unbounded-time minimum-change CPU solve was deliberately terminated after
over6 minutes, with all input artifacts preserved; this was not a training crash.
v383 bounded interior-point solve succeeded:1247 changed rows, L1 tau change
694.217, maximum13.8154, slack .88251654. Some extreme values are driven by
otherwise-unreachable rays and are diagnostic only, not a sensible deployment
prescription. v384 native revalidation is running; provenance now binds the
exact diagnostic checkpoint as well as primitive IDs/current tau.

v384 failed during JSON ray-record construction (new provenance already
contained target); fixed the duplicate-field merge, v385 completed normally.
Dense LP native replay: evaluation tree+.020182, rigid-.015857, hard-.021938 dB.
810 selected tree rays: actual floor violations809->172, mean max-channel RGB
error .144585->.085096.914 selected visible-wall rays: mean error .027821->
.024176. Thus training-ray feasibility does not imply unobserved-view safety.
Not adopted; some tau values near cap are intentionally only counterfactual.

v388 adds negative rays from TRAINING views where the preceding LP solution
worsens visible-wall RGB, requiring the same eroded mask/pure-wall RGB and two
neighbor depth agreements. Original constraints retained; no evaluation pixels
are admitted. Total optimal slack remains .88251653. v390 minimum-change solution
changes1669 rows, L1 tau709.433. v391 native verification running.

v382 row-active8 completed0, all frozen/causal checks pass. Every audited step
decreases its actual objective; inactive candidate/source opacity AND source
feature rows do not move after postprocessing. v386 ordinary Adam and v387
row-active800 are running, one policy difference, same candidates/cameras/LR
horizon and all building parameters frozen. Actual-step audits at400/800.
Archive v386_v387_row_clock800_sources.tar.gz. No production optimizer change.

## Metric-domain correctness

Found an audit-domain mismatch: historical nonnegative-color floor used raw
RGB while final PSNR used RGB clamped to[0,1]. Added explicit metric_domain;
current building-floor audit reports clamped-domain numbers plus raw_domain
for transparency. Training losses unchanged. Also corrected the near-opaque
counterfactual ratio denominator to use that counterfactual's RGB, not the
unaltered checkpoint RGB. Regression test covers saturated radiance.

v389 recomputes v363800 over48 views: largest floor-fraction domain difference
is .00017244 (view713 canopy).660 leakage ratio remains EXACTLY .71534389.
This correctness fix does NOT explain or remove the observed background leak.

## Consolidated intermediate results: v386--v392

At the matched400 checkpoint, row-active minus ordinary Adam is tree-.301385,
interior-.339366, boundary-.243005, rigid+.008467, hard+.001170 dB.
Absolute tree gains are+.179622 versus+.481008. This does not support adopting
row-active updates: preventing inactive-row motion reduces fitting here.
Both same-view400 actual-step objectives decrease; ordinary Adam decreases
.00141525 versus row-active .000584416. Neither is evidence of a broken Adam
descent direction. The800 comparison remains pending.

v391 training-risk-augmented LP native replay completed: tree+.019949,
rigid-.015643, hard-.021656 dB. Essentially unchanged from v385; not adopted.
v392 therefore expands conditional joint constraints to ALL304 training
cameras (8 rays per kind per view), with no evaluation targets admitted.
This remains sparse within each image and cannot certify full-scene feasibility.

The two individually infeasible v378 rays lie near bottom-of-image tree/post
interfaces on visual inspection. They are not unambiguous missing-leaf labels;
no automatic geometric birth is justified by those two constraints alone.

Latest full CPU suite:1385 passed,30 skipped. Subsequent independent CUDA suite:
5 passed, including cap/cutoff exceptions to optical-kernel split equivalence.
No production model, optimizer, or renderer is replaced by these diagnostics.

v386/v387 both completed800 normally. Row-active minus Adam: tree-.425175,
interior-.473918, boundary-.343880, rigid+.011231, hard+.017464 dB.
Absolute tree gains+.242094 versus+.667269. Reject row-active as an improvement
under this matched budget; no claim that ordinary Adam is universally optimal.

v392 completed304 training-camera support extraction:3781 rays(1437 occlusion,
2344 visible),327684 variables,664906 nonzeros. Joint slack8.06416324 equals
both the sum of individual maximum-capacity deficits and positive-only LP slack.
Twelve rays account for all residual deficits, all lacking persistent foreground
support in this conditional model. At least the largest(661,427,205) visually
lies inside the evergreen canopy; several others are near image-bottom or
tree/post interfaces. Do not treat every deficit as a verified leaf birth label.
No additional joint wall/tree conflict is established by this sparse sample.
v393 bounded minimum-change CPU solve is running before native replay.

Independent CUDA suite now6 passed, including order reversal after swapping
surface/volume depths and exact per-pixel crossing forward/backward checks.

v393 full-variable minimum-change phase reached its120-second solver limit;
no usable solution and no model changes. Added bounded restricted-column solve:
all3781 ray constraints retained, strongest positive support plus contributors
needed to remove frozen negative excess,37628 active variables. This is NOT
global minimum L1 change. v394 initially rounded tau tofloat32; corrected solver
artifact precision in v395. v395 changes1761 rows, L1tau1671.8384, optimal slack
8.06416677 versus unrestricted8.06416324 (difference3.53e-6). Native replay v396
is running across all304 training and48 regression cameras. Still not adoption.

Added optional diagnostic --leaf-optical-kernel projected_tau. Candidate seeds
and immutable source-reference targets remain native and identical to control;
only trained foliage renderer and its auxiliary/update replay use the new kernel.
Step0 disabled-candidate identity uses the same selected kernel, while reported
source baseline remains the common ORIGINAL native image. Manifest binds the
independent adapter and binary. No production renderer/global monkeypatch.
v397 projected_tau and v398 native eight-step paired smokes are running.
Contribution auditor now follows and verifies the checkpoint kernel; old saved
gradient probes reject projected_tau rather than silently using the wrong model.

Project tests after initial kernel integration:1387 passed,32 skipped. Root-wide
pytest additionally discovers a third-party tetra-triangulation test failing to
import unbuilt tetranerf.cpp (1391 passed,1 failed,32 skipped); this optional
dependency is outside the current canopy path and has not been changed.
New comparison provenance regression:5 tests passed.

v397/v398 eight-step smokes both completed normally; frozen/causal audits pass.
New minus native: tree+.011287, rigid-.004867, hard-.017623dB. Not an improvement
claim. v399 independently reloads projected_tau checkpoint: maximum discrepancy
from training evaluation is1.76e-6dB tree and1.91e-6dB rigid across48 views.
Kernel-aware contribution audit now verifies adapter/binary provenance; unsupported
saved-gradient probes still reject the new kernel explicitly.

Fixed another resume edge case before v400/v401: cumulative causal observation
flags were reset after resume, so an unobserved short tail could falsely fail
the 'ever updated' checks. Save/validate/restore all seven boolean flags and mark
final audits as cumulative. Same-code guard rejects old-code checkpoints; no
fabricated history fallback. Nine targeted resume/comparison tests pass. Earlier
numerical full-scene resume tests predate this metadata extension.

v400 native and v401 projected_tau800 are running onGPU1/2 respectively, same
recipe and immutable native reference images; only kernel changes. Source archive
v400_v401_kernel800_sources.tar.gz. Audits at400/800; production untouched.

## Native surface-prefix correction to the feasibility diagnosis

v396 completed all352 views. Regression means: tree+.027929, rigid-.025462,
hard-.042461dB; training means tree+.024078, rigid-.019410, hard-.143697.
660 still visibly leaks and tree PSNR declines .020727dB. Not adopted.
The median-depth approximation itself fails badly on some rays: predicted
versus native black-foliage RGB error up to .503869 (tree-ray median .008253).
These conditional LP results are NOT a safe feasible-scene certificate.

v402 temporary black probes measure native pure-rigid radiance in front of a
depth, no scene edits. Eight worst TRAINING rays show radiometrically feasible
probe depths6.57--13.33 versus rigid medians16.29--54.85. Depth/median ratios
.185--.818; all sampled prefix curves monotone. A new synthetic native CUDA
test proves that an opaque leaf before a surface median can leave .4 foreground
surface radiance; its first run had a pytest CUDA-scalar conversion typo, fixed,
then passed. This is a limitation of median-based diagnosis, not renderer sorting.

Seed-bound MoGe and the current .4--1.2 search range already include those
nearer depths. Reconstructed checkpoint XYZ/K support shows7/8 rays have NO
persistent source contributor before the radiometric frontier; the foreground
candidate optical depths are .000075--.00513, while total median-front optical
depths are around2.85--5.20. Most apparent optical coverage sits too far behind
the bright foreground rigid contribution and is useless for suppressing it.

v403 replaces old positive constraints with67 multi-depth PREFIX constraints
on those8 rays, retaining2344 visible-wall constraints from304 training cameras.
For each z, only rigid RGB before z enters log(B_prefix(z)/(GT+.03)), and only
volumes before z enter K. These are necessary, not sufficient constraints;
finite-screen suffix leakage allowance is subtracted. Full-variable slack0.
v404/v405 restricted solve varies1175 variables, changes603 rows, L1tau33.1935,
maximum change2.34018, slack1.05e-8. Corrected sparse-index artifact dtype to
explicit int64 in v405; v403/v404 int32 artifacts retained, not silently rewritten.
v406 native all-view replay is running. Still no model promotion.

v400/v401 matched400: new kernel tree+.006554, rigid-.001611, hard-.006634dB.
No meaningful kernel-only improvement established. Latest project tests before
the new native-prefix test:1389 passed,32 skipped.

Potential live-source numerical-floor inconsistency identified for follow-up:
diagnostic source projection permits1e-6, exactly the pixel cutoff, whereas
native recovery test uses1e-5. Examined v353800 and both kernel400 checkpoints
have ZERO source/candidate rows at or below1e-6 (min values above1.69e-5).
It cannot explain those checkpoints' observed leakage; do not call it that cause.

v400/v401 completed800: new-minus-native tree+.011377, rigid-.001031,
hard-.002841dB. Contribution audits v407/v408 match their saved kernel.
660 supplemental ROI: PSNR13.269472->13.346395, surface contribution
.371241->.366411;657 .213252->.208719,690 .009176->.008505. Real but small;
660 still visibly leaks. No production kernel adoption. v409 resumes exact
v400400 into a fresh output with unchanged horizon/recipe to validate the newer
cumulative-history continuation at real scale; running, not yet complete.

v406 corrected-prefix replay completed352 views. Regression means tree+.001815,
rigid-.000119,hard+.000120; training means tree-.005733,rigid+.000300,
hard-.001523. Worst regression rigid -.006155(view674), training-.020634(673).
The8 UNIQUE target rays (67 depth constraints are NOT67 independent pixels):
mean positive black-floor excess .188266->.019091; mean max-channel RGB error
.279766->.198672. Only8 candidate rows increase tau;595 rows decrease.
This demonstrates some useful foreground opacity degrees of freedom, NOT
whole-canopy recovery or proof of geometry truth. One ray retains substantial
excess .110193; necessary sparse constraints are not sufficient.

Prepared UNCONNECTED CandidateOpticalCoordinate prototype; three CPU chain-rule,
lower-bound recovery and state-identity tests pass. User approval for an actual
parameterization experiment requested asynchronously; no trainer imports it.

Additional confirmed diagnostic coordinate mismatch: candidate ray helpers
use(x-cx)/fx but unchanged native ndc2Pix maps that point to x-.5. A narrow
NativePixelCamera proxy subtracts.5 from helper cx/cy (NOT rendering camera or
building K). CPU roundtrip and actual CUDA projected-mean roundtrip tests pass.
Not yet integrated into training; preserve the running exact-resume experiment.
This half-pixel defect cannot alone explain multi-unit foreground depth gaps.

v409 completed exact400->800 continuation. All regional mean differences from
v400 are below6.2e-7dB; frozen geometry/buffers/IDs are bitwise equal and cumulative
causal histories agree. Trainable tensors are NOT bitwise equal (maximum candidate
DC difference .01837); this is functional agreement, not bit-exact replay proof.

Integrated the helper-only pixel correction after v409 finished. Fresh diagnostic
default is native_center; v410 legacy_corner / v411 native_center800 are a matched
pair with only that argument different, running on physicalGPU1/2. Both use the
unchanged native kernel. Sources archived in v410_v411_pixel_coordinates800_sources.tar.gz.
No rendering camera, building geometry, production seed initializer or original
renderer was changed. Do not alter the trainer/helpers while this pair runs.

Live eligible source opacity numerical floor now1e-5 instead of the exact1e-6
pixel cutoff; frozen rows unchanged. Coupled native/new-kernel tests confirm a
recoverable gradient. The inspected prior checkpoints have no rows at that floor,
so this corrects a latent dead-state issue, not their demonstrated leakage cause.
Old saved-update probes reject a changed source-projection helper hash.
Depth calibration inspection found global scale followed by per-view scale once,
not duplicated scale multiplication. No global depth rescaling is justified.

Latest project CPU suite:1396 passed,35 skipped,15 warnings (pytest -q tests).
Latest independent optical CUDA suite:8 passed. Tau-coordinate training remains
unconnected and unlaunched pending the user's response to the controlled proposal.

v410/v411 completed matched400 evaluation: native-center minus legacy tree
+.000895, interior+.002142,boundary-.006697,rigid+.000044,hard-.002098dB.
No meaningful quality gain; continue800 before final assessment.

v412 read-only selected-prefix optical signal audit is running on GPU1, sharing
with the bounded training pair within memory budget. It selects ONLY the8
candidate rows increased by v405, evaluates the exact full declared RGB+guard+
auxiliary objective over304 training views, and separates weighted auxiliary
gradients from RGB/guard gradients. Native responsibility counts distinguish
contribution from gradient sign. No optimizer/model updates and no evaluation
camera supervision. Archived v412_selected_optical_signal_sources.tar.gz.
At the original800 checkpoint, these candidates' seed cameras were visited2--3
times; that is NOT their total effective-observation count. Last moments have
mixed signs, which is NOT proof of optimizer error. Two helper CPU tests pass.

v410/v411 completed800. Pixel correction minus legacy tree-.001454,
interior-.000913,boundary-.003277,rigid-.000896,hard-.001548dB. Not a quality
breakthrough. v413 contribution audit of v411 completed:660 leakage ROI
PSNR13.274555, surface contribution.371785 versus v407's13.269472/.371241.
Visual inspection still shows strong lower-canopy building streaks. The latter
comparison is historical same-recipe, not the strict v410 paired contribution
audit; do not exaggerate that tiny difference. No production promotion.

v412 completed304 fixed-state training views. The8 LP-increased candidates have
net gradients requiring increase for ONLY2 and decrease for6. Nonzero-gradient
view counts19--38; contribution>1e-5 view counts9--43. These are fixed-state
counts, NOT actual historical update counts. Important: saved v353 auxiliary
weight is ZERO, so the reported weighted auxiliary gradient is correctly zero,
not evidence of a broken auxiliary path. Prepared/running v414 separates tree
RGB, background/guard, and an explicitly COUNTERFACTUAL unit absolute auxiliary
gradient. The latter is not part of the saved recipe's declared objective.
v414 sources archived separately; no scene updates or LP-target fitting.

v414 completed304. ALL8 selected candidates have tree-RGB gradients favoring
increased opacity;6 are reversed by background/guard gradients. A hypothetical
unit absolute auxiliary reverses only2 of those6, leaving4 total still favoring
decrease. This diagnoses an objective/support conflict, not failed Adam execution.
It remains a local fixed-state result; no claim all canopy primitives behave so.

Next attachment work-package C: bounded candidate-axis selectivity. v415 audited
all304 training views at v411800. Cohort requires measurable tree responsibility
in>=2 views, eroded-rigid responsibility in>=2 views and >1% rigid responsibility
fraction.456034/557514 candidates qualify (broad low-threshold overlap, NOT proof
all are incorrect geometry). Initial candidate XYZ/scales/quaternions are bound
by hash, along with source identity, train/excluded lists and cohort checksum.
The cohort grants experimental shape freedom only, never negative/verification
authority; no eval-view contribution is used for selection.

SelectiveSpatialFootprintCandidates adds zero-initialized deviatoric axis codes
only on that cohort. Each axis STILL stays within.5--2 initial scale, position
bounds unchanged, no rotations/new primitives/source-shape changes. Zero shape
matches both old forward and isotropic-code derivative. Native finite difference
passes;4 shape tests passed onGPU2. v416 lr0/v417 lr.02 matched8 smokes running
GPU1/2, sources archived. Native old kernel, Adam and all loss weights unchanged.
Tau-coordinate prototype remains inactive, and no production promotion occurred.
Latest full CPU suite before shape additions:1398 passed,35 skipped.

v416/v417 smokes complete. Both frozen audits true. Control shape changed0 rows;
treatment304106 rows, no ineligible changes, axis ratios .87056--1.13596 (well
inside .5--2). Shape gradient dot actual update negative at4/8; zero-gradient
rows can still move via ordinary Adam momentum, not a new bug. Mean image
differences around1e-5dB are not a quality result. Report CLI initially lacked
the newly allowed argument; unified its duplicate argument lists and reran it.

v418 exact v4174->8 continuation running; v419 shape-lr0/v420 shape-lr.02
matched800 runs startedGPU1/2, eval400/800. No other recipe difference, original
native renderer and all building protection retained. Sources archived as
v418_v420_selective_shape_sources.tar.gz. Do not edit their trainer/helpers while
running. Latest full CPU suite1402 passed,36 skipped,16 warnings; native shape
finite-difference test passed separately. No model adoption.

v418 complete: fixed candidate XYZ/scales/quaternions/shape cohort and optimizer
layout match exactly. Max shape-code difference .00036831 (NOT bit-exact);
regional mean differences at8 at most1.99e-8dB. v417 actual objective decreases
at audited4/8 by.0019688/.00009437. No new update-path failure detected.

v419/v420 matched400: selective shape tree+.010462,interior+.012780,
boundary+.003942,rigid+.000159,hard+.003739dB. Small, not a decisive canopy fix;
continue800. Read-only low-pass surface-depth branch check found native mixed
matches original surfel formula (uses center depth on low-pass branch), no bug
established there.

v421 extends the same8-candidate fixed-state audit with a COUNTERFACTUAL
ground-truth background RGB derivative, keeping regional normalization and guard
weights. Tests whether reference preservation agrees with observed-image error;
this is NOT new training supervision or permission to occlude noncanopy pixels.
RunningGPU1, sources archived v421_selected_reference_target_bias_sources.tar.gz.

v419/v420 completed800: selective-minus-control tree+.017347,interior+.019370,
boundary+.010291,rigid-.000083,hard+.002194dB. Both frozen audits pass. All456034
selected shape rows changed in treatment, none in control; ineligible rows stay
zero. Treatment axis ratios .50802--1.99907 within original .5--2 limits.
v422/v423 kernel/shape-aware48-view contribution audits completed. Strict paired
660 ROI PSNR13.274484->13.303595, surface.371788->.370748;65716.783401->16.776129,
surface.213375->.212883;69016.832193->16.833422,surface.009199->.009192.
Directly inspected660/738/674:660 bright building streaks remain substantial,
other canopy structure remains blurred/missing. Small real effect, NOT a solution
or justification to lengthen the same recipe. No production adoption.

v421 completed. Of the6 candidates suppressed by reference background/guard,
4 have OPPOSITE ground-truth background derivatives. Conversely both candidates
encouraged by reference background have GT derivatives asking for decreased
opacity. These are fixed-state regional raw-L1 derivatives, not geometry truth
or permission to add foliage on noncanopy pixels. Reference copying is therefore
not synonymous with protecting observed building quality; switching to plain GT
everywhere would not establish correct optical ownership either.

Next isolated observed-risk test (no production change): background loss is
relu(L1(pred,GT)-L1(immutable_source,GT)), region-normalized. It gives ZERO reward
and derivative for reaching/beating the source error, including equality. In
both arms retain a negative visible-rigid term on a conservative conditional mask:
eroded noncanopy, original surface contribution>=.95, source-vs-GT meanL1<=.05.
Penalize only reductions of the original surface contribution, squared, weight4.
This mask combines observations/model agreement, NOT independently verified
geometry. Soft losses cannot guarantee zero held-out risk; explicit contribution
audit added. No inference gating, pseudo leaf-depth truth, forced foliage alpha
or building-parameter updates.

Added one-channel immutable CPU reference cache mode for source surface alpha;
default RGB behavior unchanged. Training now directly calls declared_objective,
removing its duplicate inline objective. Actual update replay passes the same
native current/source surface fields. Old standalone saved-step/selected-gradient
probes reject new visibility recipes rather than silently omit the term.
v424 source_reference/v425 observed_risk matched8 smokes runningGPU1/2, same
visible weight4, native old kernel, original isotropic candidates (shape off).
Sources archived v424_v425_observed_risk_smoke_sources.tar.gz. Latest full CPU
suite1408 passed,36 skipped,16 warnings. No quality conclusion yet.

Coupled native visibility test passes finite differences: added leaf opacity
before the wall gets a positive (repulsive) loss gradient, behind the wall gets
zero.7 risk tests passed onGPU2. v424/v425 smokes complete, frozen audits true,
source-alpha caches each8 immutable CPU entries. First training view had80103
conservative visible-wall pixels, not an empty guard. Actual losses decrease at
4/8 in both arms. Risk-minus-reference8 tree+.000450,rigid-.000458,hard+.000661:
not a quality result. v426 exact risk4->8 continuation and v427 reference/v428
observed-risk800 startedGPU1/2, same visible weight4 and only target policy differs.
Sources archived v426_v428_observed_risk_sources.tar.gz. No trainer/helper edits
while these run. Contribution auditor now also reports sky PSNR/volume alpha,
because the new objective covers sky as well; never use those eval pixels for
positive foliage supervision. No production adoption.

v426 exact observed-risk4->8 continuation completed: all five regional mean
metric differences versus uninterrupted v425 are zero; frozen audit true.
Trainable tensor bitwise equality was not checked for this pair.

v427/v428 matched400 comparison: observed-risk minus reference tree+.145669dB,
interior+.123947,boundary+.228497,rigid+.037154,hard+.162838. Absolute vs original
source tree+.608144,rigid+.028080,hard+.170911. This is larger than the recent
kernel/shape changes but is not a solved-canopy result. Directly inspected660:
bright lower-canopy building streaks remain. Hard-region source-relative losses
persist on738(-.678904dB) and674(-.518233); averages cannot certify preservation.
Continue both800 completions and paired native contribution/sky audits; no
production promotion and no new training family yet.

v427/v428 completed800 normally, both frozen audits true. Matched new-old:
tree+.202649,interior+.178869,boundary+.287216,rigid+.040509,hard+.179466dB.
New absolute source-relative tree+.841032 (48/48 positive),rigid+.027905,
hard+.173500. Actual same-view loss replays decrease at400/800 for both arms.
Latest full CPU suite1408 passed,37 skipped,16 warnings.

v429 risk/v430 reference native contribution audits completed48 each:
660 ROI controlPSNR13.271508 -> risk13.432705; surface contribution
.371985 -> .363160. Real but insufficient extra occlusion: risk ROI fixed-
geometry/opacity/background nonnegative-color error lower bound fraction=.718878.
657 ROI surface .213510 -> .209125, but PSNR16.801718 ->16.684929;
690 ROI PSNR16.833448 ->16.790176, surface .009211 ->.009295.
Thus increased opacity is not equivalent to improved reconstruction.
Risk738 hard vs original -.931273dB,674 -.512257. Conditional visible-mask
mean contribution drops:738 control.001237 ->risk.002841;674 .007900 ->.011399.
Risk sky mean source-relative PSNR+.298186 over45 nonempty views, worst-.047335.
No production adoption; average gains do not certify building preservation.

v431 reference/v432 risk read-only joint leave-one-component-out audits launched
GPU1/2. Revert sourceDC/SH/opacity/staticposition, disable candidates, or revert
candidateposition/size separately, keeping all other learned factors. Report
48-view PSNR/surface contributions and conditional visible-mask drops. These are
nonadditive counterfactual differences, not independent causal credits. Exact
touched-tensor restoration in finally;16 CPU tests including failure paths pass.
No fitting, exports, new rendering kernel or building changes. Sources archived
v431_v432_joint_components_sources.tar.gz. Await results before another recipe.

v431/v432 finished48 views each. In riskv428, disabling candidates recovers
738 hard+.83817dB,674+.42444, but loses660 tree-.51478 and increases660 ROI
surface contribution+.03468. Reverting candidate sizes alone recovers738
hard+.50216,674+.29135; loses660tree-.29408 and increases ROI surface+.02102.
Reverting candidatepositions recovers738hard+.27442; reverting originalsource
opacity only+.09690, sourceposition actually worsens738hard-.02564. Thus current
damage is predominantly candidate support/optics, not mainly source staticXYZ
or directionalSH. These effects are conditional/nonadditive, not a decomposition
that can be summed. In711, restoring originalsourceDC/SH/opacity helps tree,
but enabled candidates now offset those losses. No one rollback is a solution.

Next bounded shape-direction test: added separate OrientedSelectiveCandidates
helper; existing selective-axis helper unchanged. Only existing training cohort
may rotate via a smooth normalized quaternion delta strictly below90degrees.
Existing axis .5..2 and XYZ limits unchanged; ineligible rows have zero rotation
gradient. Zero code exactly matches old quaternions and has finite derivatives.
CPU constraints and original native CUDA rotation finite difference tests3pass.
Trainer saves helper hash/state/optimizer layout and checks rotation authority;
auditor reconstructs it explicitly. v433 rotationLR0/v434 LR.02 matched8 smokes
startedGPU1/2; both use observed-risk, visibleweight4, shapeLR.02, same v415 cohort.
Sources archived v433_v434_orientation_smoke_sources.tar.gz. Do not infer this
necessarily solves the support conflict; no production renderer/model changes.

v433/v4348 complete, frozen audits true. Rotation changed0 /285587 rows;
ineligible unchanged, treatment maximum angle.218344rad. First-step rotation
gradient~1e-11 total is expected near numerical zero for initially spherical
primitives; bystep8 gradientL1=.00007345. Matched8 metrics differ only~1e-7dB:
correct execution, not demonstrated quality benefit. Latest full CPU1426pass,
38skip,16warnings. v435 exact active4->8 resume startedGPU1; v437active800
startedGPU2. v436 matched rotationLR0 control800 to startGPU1 after resume.
All use archived v433_v434_orientation_smoke_sources.tar.gz unchanged training
sources. No optical-coordinate experiment has been enabled (user reply pending).

v435 resume completed normally; layout equal, frozen audit true. Final mean
tree/interior/rigid/hard equal, boundary difference -1.9868e-8dB. BUT trainable
rotation_code max difference=.0199042 (row419301), shape_code maxdiff=.00072445,
candidate logits maxdiff9.54e-7, candidateposition maxdiff.00001639. Do NOT claim
bitwise parameter replay. Worst rotation row remains weak (logit-6.94092) and
near axisymmetric at8; numerical/gauge sensitivity is plausible but its exact
cause is not established. No evidence of missing optimizer layout or lost
checkpoint state. v436 matched rotationLR0 control800 now runningGPU1; v437
active800GPU2. Treat any covariance/reconstruction benefit separately from
parameter reproducibility. No training-source edits during this pair.

Numerical follow-up found a reproducible issue in the NEW orientation prototype,
not a claimed explanation of old canopy failures. Native pure-sphere probe at
opacity.001 has identical loss.03291941434 at rotation codes0/.2, yet float32
rotation gradients up to9.59e-10. Adam eps1e-15 turns these into~.02 steps even
though sphere orientation is unobservable. Explicitly stopped v436/v437 via
verified supervisor PIDs7350/6505 SIGINT (last trace50/175), not OOM or spontaneous
training failure. No files removed; unpublished intermediate progress was not
saved. Old supervisor records these stops as failed/invalid_interruption_pair;
records retained unchanged. Do not use unfinished pair as quality evidence.

Fixed prototype by canonicalizing quaternion and zeroing its gradient when
principal-axis spread <=1e-5 relative to maximum axis (detached numerical
isotropy predicate). No image/visibility/depth masks, no size/XYZ expansion.
Native isotropic tests now verify exact zero rotation gradient and zero Adam
movement, even at nonzero latent rotation; anisotropic finite difference still
passes.4 orientation tests passGPU2. This removes the demonstrated roundoff
path, not a guarantee of bitwise CUDA continuation for all learned tensors.

Also fixed supervisor status classification for a stop requested through the
supervisor with no valid interruption checkpoint: stopped_uncheckpointed,
explicit requested_stop_without_valid_checkpoint detail, zero restarts, NEVER
completed or checkpointed. Unexpected child signals still failed; OOM/SIGKILL
path unchanged. Integration tests exercise requested and unrequested SIGINT;
supervisor+orientation CPU suite32passed2skipped. This is reporting correctness,
not an implementation of checkpoint-on-signal for diagnostic trainers.

v438 control/v439 active corrected8 smokes launchedGPU1/2 with unchanged recipe.
Sources archived v438_v439_observable_orientation_sources.tar.gz. Earlier
orientation artifacts require their archived helper for exact old interpretation.

v438/v439 corrected8 complete, frozen audits true. First-step rotation gradient
is exactly0 in both, while other losses/updates match. At8:282767 observable
rows, rotation changes0/146764, ineligible unchanged, active maxangle.195703rad.
Metrics remain indistinguishable at~1e-7dB (no quality claim). Full CPU1428pass,
39skip,16warnings. v440 corrected exact4->8 startedGPU1; v442 correctedactive800
startedGPU2; v441 rotationLR0 control800 to follow onGPU1 after resume. No new
training-source edits and no production adoption.

v440 corrected exact4->8 completed, layout equal/frozen audit true. Final
regional mean differences <=3.974e-8dB, but rotation_code maxdiff=.0190512 still
not bitwise. Reconstructed world covariance comparison: maximum relative
Frobenius difference .00093997, mean .00000095233, no row above.001. Worst-code
row430448 covariance relative difference .00035257. Thus the remaining latent
code sensitivity is small in observable covariance here; do not claim identical
parameters or that sphere canonicalization removes all numerical variation.
v441 corrected rotationLR0 control800 now onGPU1, v442 active800GPU2. Await
matched400/800 results and original native contribution checks.

v441/v442400 matched orientation result: active-minus-fixed tree-.011417,
interior-.012686,boundary-.011774,rigid+.000293,hard-.002130dB. No orientation
benefit established; finish800 without adding freedom or another recipe.

While waiting, added independent read-only native surface preprojection and
per-contributor pixel depth exports. They reuse mixed_preprocess_surfaces and
mixedEvaluateCandidate, including the low-pass depth fallback, but do not
modify any forward/backward renderer. The production binary is untouched;
independent diagnostic binary rebuilt (old projected-tau checkpoint binary
provenance must not be silently reinterpreted). New prefix helper combines
ORIGINAL native per-ray surface responsibility with those depths and native SH
colors, ordered by (depth,primitiveID). Surface ties precede volumes, so prefix
queries include surfaces at depth<=query. No inferred alpha, median surrogate,
model edits or leaf-depth targets. Synthetic plane/tilt/black-screen and existing
independent kernel checks12pass. Initial test import-path error corrected; no
training process failed from this work.

v443 native8 prefix audit completed: max reconstructed RGB error5.96e-8 and
max discrepancy from prior v402 opaque-screen samples5.96e-8. Exact first
over-bright contribution depths remain far ahead of median on7/8 rays (ratios
.185-.402; remaining.818). New depth is a discrete contributing-surface crossing;
old probe gave a finite bisection lower bound, so tiny depth differences expected.
Sources AND new independent binary archived v443_native_surface_prefix_sources.tar.gz.
v444 extends the same read-only measurement to all3781 prior TRAINING diagnostic
rays across304 views onGPU2; no new fitting or evaluation-view supervision.

v444 completed all3781 rays: maximum native RGB reconstruction error1.78814e-7.
Of1437 positive diagnostic rays,178 cross GT+.03 before.8 times the rigid median,
91 before.5 times median. Most are near median; the prior eight worst rays are
not representative of the whole canopy. Of2344 conditional visible rays,311
cross somewhere, none before.8 median. This is sampled training evidence only.
Worst ray605:(403,246) has frontier1.016597 vs median15.194246. Surface268017
contributes responsibility.261049 there; its center camera-z1.022151, scales
.003645/.027611, opacity.302481. Historical full support reports88 rigid views,
8 canopy views, rigid/canopy mass78.606/29.796. No independent contradiction is
established; no rigid parameters or masks changed and no deletion authorized.

v441/v442800 and native48 contribution auditsv446/v445 completed normally.
Rotation active-minus-control: tree-.000578,interior-.001584,boundary+.002604,
rigid+.001983,hard+.002821dB. No useful rotation improvement. Both mean tree
improvements vs original are~+.883dB, not a solved canopy.660 lower ROI fixed
orientationPSNR13.485213 / active13.463035; surface contribution.362584/.362805
vs original.424097. Nonnegative-color lower-bound error fractions.716777/.716716
remain large. Do not retain the added rotation freedom on this evidence.

Added bounded full native-prefix constraint loader with artifact/source/view/ray
hash validation, positive training rays only, up to6 exact cumulative-depth
constraints per ray (complete tied surface groups), preserving all prior visible
constraints. Exact prefixes use zero screen leakage allowance; older screen
format remains supported unchanged. Projection is cached per consecutive view.
13 selected CPU tests passed. v447 full-prefix joint diagnostic launchedGPU2,
samev353800 fixed geometry asv392 andv403; no optical solution is a training
target or retained model. The finite constraint subset remains necessary, not
sufficient, and any proposed solution still requires native all-view replay.

v447 completed10966 constraints/555083variables/5759187nonzeros. Joint slack
11.14515366866 equals exactly the sum of independent capacity deficits with all
allowed variables at caps;21 constraint rows are undercovered, no extra joint
conflict demonstrated by this finite subset. Current optical deficits1737.01090,
upper excess12.31937. This is still an uncapped necessary submodel, not proof of
a safe render. v448 restricted minimum-change solve completed55140active vars,
1923changed rows,L1tau1355.99815,maxdelta13.81548,slack11.14515367866 (1e-8 above
full optimum). No LP values used for training or exported as a map.

Important replay clarification/fix: historical `replay_joint_solution` always
called independent_render(), the projected_tau CUDA kernel, not production
alpha_peak*G. Prior references to native replay meant actual CUDA mixed order,
but did NOT verify the original optical kernel. Added explicit replay kernel
selector (legacy default retained) and kernel/binary/source/snapshot identities
in reports. v449 original production kernelGPU1 andv450 projected_tauGPU2 replay
the SAMEv448 counterfactual over304train+48regression views. Production renderer
unchanged. Sources archivedv447_v450_full_prefix_replay_sources.tar.gz.

Project CPU suite1436pass41skip16warnings. A broader root pytest invocation
also collected vendored tetra-triangulation tests and failed its unbuilt cpp
import (1failed1440pass41skip); this optional dependency is outside the current
canopy chain and was not compiled or modified. Do not report root suite allgreen.

v449/v450 completed352 views each normally. Full-prefix minimum-change LP
counterfactual is NOT accepted. Original production kernel48 deltas: tree+.025911,
rigid-.012279,hard-.019469dB; training means+.088042/-.010025/-.083123. Exponential
kernel48 deltas+.028870/-.020207/-.029416; training+.089950/-.016345/-.113191.
660 tree is slightly WORSE under both counterfactuals, and direct panel inspection
still shows extensive lower-canopy bright building leakage.674/738 buildings also
regress. Thus full-prefix sampled necessary feasibility still does not establish
dense native RGB/visibility safety. No LP-opacity model is retained, no unsupported
rigid correction, and no production adoption. Direct tau-coordinate prototype
remains inactive pending the earlier user confirmation. Latest project tests
1437pass41skip16warnings; added batch artifact completeness/identity tests7pass.

## Authorized candidate optical-coordinate comparison

User now explicitly authorizes direct candidate tau optimization and future
same-scope controlled experiments without repeated permission questions. Physical
GPU1/2 only; validated buildings remain protected. This does not authorize
uncontrolled rigid edits or production promotion without evidence.

Integrated optional bounded_logit/bounded_tau coordinates into diagnostic trainer.
Both use opacity[1e-5,1-1e-6], same native alpha_peak*G renderer, same candidate
geometry/source recipe. Only candidate optimizer coordinates differ; source
leaves retain their original parameterization. Tau gradients transfer by dividing
the accumulated logit gradient by sigmoid(logit), then projected tau is synced
through stable inverse softplus. Renderer logits gradients clear at every optimizer
boundary. Save/resume records tau, optimizer row identities and helper hash;
actual-step audits operate in optimizer coordinates then re-render the declared
objective. Same numerical LR is NOT the same physical opacity step; the comparison
tests this different optimization behavior, not representation capacity or truth.
Offline old optimizer probes reject new coordinate modes rather than silently
reconstructing a logit optimizer. Rendering audits still consume ordinary logits.

v451 bounded-logit8GPU1 andv452 bounded-tau8GPU2 launched, same seed/shape/risk
recipe asv441, no rotation updates. Short test before800-step paired runs and
exact4->8 continuation. Initial coordinate/comparison tests10pass. Sources archived
v451_v452_optical_coordinate_sources.tar.gz. No result or safety claim yet.

v451/v4528 completed; frozen_parameter_audit unchanged=true for both. Matched
tau-minus-logit tree+.091991,interior+.080258,boundary+.134455,rigid-.052586,
hard-.036040dB. Native660 visualization still leaks substantially. Tau candidate
mean opacity.007237/max.202712 vs logit.000975/.001247. This is effective optical
updating, not proof of safe geometry. Actual declared loss deltas at4/8:
logit-.00199005/-.00009303; tau-.01174536/-.00241057. Both decrease the objective.
Latest project CPU tests1439pass41skip16warnings. v453 exacttau4->8GPU2 andv454
bounded-logit800GPU1 launched. v455 bounded-tau800 follows successful resume
verification. No production adoption; no GPU0 use.

v453 restored4->8 completed normally, frozen audit true, optimizer layout equal.
Tau maximum difference1.60187e-7, candidate tensor maximum difference.000181198
vs uninterrupted8; regional final metrics agree at printed precision. This is
numerically consistent continuation, NOT bitwise-identical parameters. v455
bounded-tau800 launchedGPU2 after this check; v454 bounded-logit800 runsGPU1.
Both evaluate48 fixed views at400/800. No training helper edits during these runs.

## MoGe supervision and thin-structure follow-up

User requests inspecting poor reconstruction regions and making more actual use
of MoGe geometry supervision. Scope clarification: the production method already
contains hit-interval geometry/optical losses in outdoor/training_evidence.py;
recent isolated candidate experiments deliberately omitted them. Do not claim
the entire method has only ever used MoGe for initialization.

v454/v455800 completed normally. Tau-minus-logit means tree+.323610,
interior+.288208,boundary+.362516,rigid-.226007,hard-.346642dB. Native contribution
auditsv456/v457 completed48 each.660 leakage ROI PSNR13.485416->14.318653,
surface contribution.362589->.302674,volume contribution.624409->.690069.
This is real local occlusion improvement, but global rigid-image regression
prevents adoption. Frozen buildings do not guarantee unchanged building images.

v458 read-only original1920x1080-to-runtime640x360 MoGe audit completed.660
railing rectangle(20,285,390,355):18.39% of3x3source blocks span>10%depth,
9.94% span>25%; max ratio1.8231. Cache equals weighted area mean within1.91e-6.
Canopy interior(390,45,610,195) andleakage(470,205,630,335):no3x3block spans>10%,
max ratios1.0543/1.0672. Thus averaging mixes depth discontinuities in the railing
rectangle, but does not explain canopy failure as the same issue. Rectangles
include background; these percentages are not per-bar failure rates. No claim
of multiview depth accuracy. Source files/cache/checkpoints unchanged. Initial
audit-script unmatched-parenthesis error fixed before successful run; training
was unaffected. Synthetic invalid-depth/mixed-depth tests2pass.

Actual supervision must preserve distinct foreground/background observations:
use original high-resolution rays for thin structures, not the mixed area mean;
calibrate canopy/rigid depths with their respective existing contracts; check
independent training views and occlusion before geometry targets. Do not use660
or other48regression views as fitting evidence. No unverified hard MoGe target
or global rigid unfreeze has been enabled by this inspection.

## v459+: conditional MoGe geometry-only candidate supervision

Added optional --moge-position-control with --moge-position-weight0/.1.
Same initial bounded-logit candidates/native renderer; no tau coordinate change
in this pair. Build fixed observations from304training views only: tree interior,
finite depth,3x3runtime depth range<5%,initial candidate camera-z agreement<10%,
sample at most16384rows/view; retain candidates sampled in>=3views and>=1degree
direction variation. Calibrated depth uses the source-bound global and canopy
scale exactly once. Target interval is+/-3%, an experimental tolerance NOT a
calibrated MoGe confidence interval. This is conditional projected agreement,
not established correspondence/leaf identity or new verification authority.

Loss acts only on candidate center camera-z outside the interval (relative
SmoothL1); no gradients to opacity, scales, original leaves, buildings or target
bounds. Existing candidate displacement bounds remain. Geometry observations
and initial xyz are content-hashed; saved target artifact retained. Same term
included in before/after actual-step rerenders. Old offline optimizer probes
explicitly reject this mode until they reconstruct its objective. Depth cache
and production renderer unchanged. Thin railings are NOT supervised from area
mean depths and not included in this tree-only experimental branch.

v459/v4608 completed, frozen audits true. Identical evidence hashes:
111824candidates/881054sampled observations. Tree delta-.00000427dB, no visual
benefit established. Complete loss decreases at4/8 for both; MoGe-active
-.00200654/-.00009381. v461active exact4->8GPU2 andv462zero-weight800GPU1 launched.
Project tests1443pass41skip16warnings. Sources archivedv459_v460_moge_position_sources.tar.gz.

v461 exact4->8 completed, frozen audit true, optimizer layout/evidence hashes
equal. Candidate maximum tensor difference.000172525, final printed metrics
equal; not bitwise-identical continuation. CPU recomputation over304views gives
mean geometry loss.00812972(control8) vs.00811575(active8), a small decrease only.
v463MoGe-weight.1 active800 launchedGPU2; v462weight0 control800GPU1. Both use
bounded-logit/native rendering with matched400/800 regression evaluation; no
training helper changes during these runs. Railings remain outside this tree-only
geometry experiment; no global rigid unfreeze or production model replacement.

## v462/v463 final; v464 reachability audit; v465/v466 filtered pair

Both 800-step runs completed normally. MoGe-active minus zero-weight control
over the same 48 regression views: tree -0.00504092 dB, interior -0.00536424,
boundary -0.00545061, rigid -0.00032244, hard -0.00392481. No useful improvement;
do not adopt. Native view660 still has lower-canopy background streaks/blur.
Strict comparison: compare_v462_v463_moge_position800.json in the run root.

CPU audit_v464_moge_position_reachability finds 377802/881054 observations
(42.9%) cannot reach their conditional depth interval inside the existing
position box. Initial relative violation sum19128.538; final control19238.696,
active15048.554; per-observation box lower bound7759.228. This establishes
unreachable conditional targets, not incorrect MoGe depth or joint feasibility.
Reduced prior residual is not independent geometry accuracy.

Added optional --moge-position-reachable-only. The exact camera-z box extent
is radius*sum(initial_scales*abs(camera_z_axis)); reject disjoint intervals
BEFORE per-view subsampling and three-view support eligibility. Rejected
observations cannot authorize geometry supervision. Targets remain conditional
and the fixed 3% interval remains experimental, not calibrated confidence.
Individual reachability is necessary, not sufficient for joint compatibility.
Four focused tests pass including all-eight-corner box validation and rejection
before support counting. No production renderer or rigid parameters changed.

v465(control weight0)/v466(active weight.1), both reachable-only, launched800
steps on GPU1/GPU2 respectively. Both passed initial48-view rendering and
reached training step1; results pending. No GPU0 use or production replacement.
Remaining work includes calibrated uncertainty, joint correspondence/visibility
validation, selective relocation/birth for unreachable coverage, high-resolution
thin-structure geometry, per-epoch reshuffling and observation-conditioned SH.

## v465/v466 final; native-depth pair v467/v468

Reachable-only active minus control at800: tree +0.00027208 dB,
interior +0.00055583, boundary +0.00018460, rigid +0.00013570,
hard -0.00037716. No meaningful reconstruction improvement, no adoption.

NativeDepthViews now lazily loads original per-view depth and masks, records
source SHA256, and queries projected native pixels directly. Exact intrinsics
are scaled BEFORE pixel rounding. No average-depth or nearest-low-resolution
ray is used. Smoothness uses an approximately matched angular neighborhood
(9 native pixels vs3 runtime pixels at3x resolution); semantic masks remain
nearest-upsampled runtime masks, NOT new high-resolution semantic evidence.
Both depth sources retain conditional correspondence, reachability and fixed
three-view support policies. RGB rendering stays640 in this isolated test.
v467(runtime)/v468(native) weight.1 pairs launched800 onGPU1/GPU2 with the
same current sources and initial candidates. No production renderer changes.
Five focused native-depth/geometry tests pass. Results pending.

Added separate fixed-ray-validity primitive and two passing tests: reject
old depth assignments after lateral projection leaves the anchor footprint,
or becomes nonfinite/behind-camera/outside-image. It is NOT yet integrated
into the running pair and does not establish actual visibility or leaf identity.
All-scene high-resolution RGB optimization, full multiview re-association,
selective birth and general rigid geometry supervision remain pending.

Completed CPU stale-ray audit onv466:442296 saved observations;420631 move
over1runtime pixel (median3.70034,p95 7.85285),280077 over3pixels.
Crucial qualification: of441767 valid old/new depth pairs, only22642 (5.13%)
change depth over3%,5243 (1.19%) over10%. Large projection drift is NOT proof
that95% of depth targets are wrong; local smoothness explains many unchanged
depths. This supports refreshing correspondences but does not establish the
dominant visual bottleneck. Audit excludes semantic/visibility revalidation.
Full project suite with required LD_LIBRARY_PATH:1449pass41skip16warnings.
An earlier invocation omitted this path and had44 import collection errors;
this was test-environment setup, not a training process failure.

## Native crop prerequisite audit v469

Prepared deterministic full-horizon per-epoch schedule (3tests) and exact
native crop intrinsics (2tests); neither integrated into running training.
Read-only GPU1 v469 source-checkpoint render compares full1920 image cropped
to[900,600,1540,960] versus direct crop. Max RGB error0.03873318,
mean0.0001735863; interior max same. Therefore direct crop is NOT yet accepted
as an equivalent training renderer. No production native binary changed.
Code inspection identifies a plausible cause: mixedComputeCov2D and backward
clamp x/z,y/z symmetrically to1.3*tan(FOV) without principal-point offset;
changing off-center viewport changes the EWA Jacobian even for shared rays.
This is not yet a complete causal ablation and cannot be called the historical
centered-camera canopy root cause. Full-frame native-resolution training is
a safe alternative to investigate while crop equivalence is unresolved.
v467/468 now both training; original-depth cohort64580candidates/430743obs
versus runtime66290/442296. Full quality evaluation pending400/800.

## v470 full-native backward; v471/v472 complete-training smoke

v470 read-only GPU1 full1920 source render backward succeeds: source features
gradient L1 .33214885, opacity .02860686, all finite; peak allocated1913.36MiB.
No optimizer step or saved model; view660 used only for renderer diagnosis.
This excludes candidates/Adam/full training objective and is not a quality gain.

Added NativeRGBViews bounded16CPU-frame cache, training-camera whitelist and
SHA256 validation. --native-rgb-training changes only training RGB cameras to
full original frames, with unchanged low-res initialization/regression cameras.
Off-center crops remain disabled by construction. Native RGB hashes/camera
helper hashes bound to checkpoint manifest. Immutable reference caches already
key on camera ID AND dimensions. Boundary masks already scale with image size.

Added --reshuffle-training-epochs using immutable full-horizon schedule and
private RNG; first epoch identical to legacy, exact suffix reconstructible.
Flag remains OFF in native RGB pair to isolate resolution. Existing v467/468
keep previously loaded code, with archived sources; no changes to live helpers.

v471640/v472native full-training smoke launchedGPU1/GPU2 alongside depth pair,
8steps with800-step horizon, eval/update audit4/8, reference CPU cache16,
MoGe-position-control OFF in both. Paired flags differ only native-rgb-training.
Sources archivedv471_v472_native_rgb_sources.tar.gz. Focused comparison/resume/
schedule/native RGB tests14pass; training results pending. Not production use.

v471/v472 completed normally; both frozen audits true. Native-minus640 at8:
tree -0.00293928dB, rigid +0.00025549, hard +0.00165405. No quality claim.
Actual loss deltas4/8: control -.00199005/-.00090744,
native -.00185390/-.00082028. Native step1 peak4699.55MiB with actual candidate
and source opacity updates. Smoke uses horizon800 but stops8, not a full fit.
Cleanv473640/v474native launched800 GPU1/GPU2, eval/update audits400/800.
Do not label this an exact8->800 resume: it is clean same-initialization fitting.
Reshuffle remains OFF and MoGe-position remains OFF to isolate RGB resolution.

Native contribution evaluator now scales fixed640 ROI boxes with actual camera
dimensions, exports full tree/interior/boundary/rigid/hard metrics and optional
GT/source/updated image panels (--save-rgb-pairs). New ROI test passes. Native
1920 evaluation will supplement, NOT replace, original48-view640 regression.
Formal MoGe rigid code audit confirms existing plane/Chart/inverse owners are
excluded from its depth/normal terms; normal confidence includes direct/depth
normal agreement. This is an existing conservative scope policy, not proven
bug or justification for globally replacing validated building geometry.
Full project tests before evaluator addition1456pass41skip16warnings.

v467/v468 common400 strict paired result: tree -.00016650dB, interior-.00018964,
boundary-.00013010, rigid-.00009994, hard-.00059797. High-resolution depth
sampling alone has no useful measured benefit at this stage. Both continue800.

## Conditional moved-ray refresh implementation and CPU audit

New canopy_refresh_moge_observations.py requeries moved projections, keeps only
original camera/ID associations, rejects new depth layers more than3% away
from ORIGINAL target centers, and recounts >=3 surviving cameras with angular
separation. No new ownership, heldout cameras, visibility proof or topology.
Three tests pass (new sampled depth, lost-view support revocation, excluded
camera rejection). Not yet integrated into the active training pairs.

CPU audit on completedv466 includes actual current semantic interior and3x3
depth-smoothness validity:442296initial observations ->376209survivors,
57471eligible candidates. About85.1% survive; refresh does not erase all
geometry supervision. This is conditional validity only, not independent
geometry accuracy and not a visual improvement claim. Artifact:
audit_v466_refreshed_moge_targets.json in run root.

Refresh now wired into the NEXT trainer via --moge-position-refresh-every;
default0 leaves existing behavior unchanged. Refresh happens only between
complete optimizer windows. Active target records/history are checkpointed
and restored, with subset/camera/finite-bound validation on resume. Before/
after optimizer audits use the same refreshed records. Native depth sources
now reject changed NPZ content on repeated queries instead of overwriting the
initial hash. Existing running processes keep their previously loaded code;
their archived source bundles remain available. No new refresh GPU run yet.
Full project tests1463pass41skip16warnings; five refresh-specific tests pass.

Additional correction before refresh smoke: recheck target reachability in the
ORIGINAL immutable displacement box before recounting views. Moving the target
must not reintroduce unreachable supervision. Sixth refresh test passes.
CPUv466 with this check retains350663/442296observations (79.28%),54598candidates;
artifact audit_v466_refreshed_reachable_moge_targets.json. Earlier85.1% result
was before rechecking reachability; retain both audits with distinct scope.

v467/v468 both completed800 with frozen audits true. Native-depth minus runtime
depth: tree +.00087639dB, interior+.00059630, boundary+.00203776,
rigid-.00001991, hard-.00000846. No useful improvement; branch not extended.
Viewed native660final: lower-canopy building streaks and railing artifacts
remain. Full comparison compare_v467_v468_native_depth800.json.

After depth pair completion, v475/v476 refresh smoke launchedGPU1/GPU2:
same weight.1/runtime depth/initial candidates; refresh-every0 vs4. Stop8,
horizon800, eval/actual-update audit4/8. Check prefix4 equality, refresh5,
actual geometry update and frozen audit; then exact4->8 continuation before
any long refresh fit. Source archivev475_v476_moge_refresh_sources.tar.gz.
v473/v474 full800 RGB-resolution pair remains active; no outcome claimed.

v475/v4768 both completed, frozen audits true. First4 actual-loss and reported
metrics match; refresh executes5 and retains438232/442296 observations,
65883eligible candidates. At8 tree delta0, hard+1.39e-7dB: no quality benefit.
Actual complete loss deltas4/8: control-.001997389/-.000909790,
active-.001997389/-.000909742. Active step8 loss_before .038086876 vs control
.038090032, so target refresh is not a constant/no-op term, but influence is
small here. Exact4->8 refresh-event resume launchedv478GPU2; outcome pending.

## v477 training-only native rigid depth/normal consistency

CPU audit completed304 canonical TRAINING views, no48 regression views.
77824 uniform valid rigid rays, nearest6 camera-center neighbors,407468
in-frame pairs,384801valid rigid target pairs. Depth agreement counts:
2%85505,5%186450,10%283003; depth5% plus normal angle30deg159753.
50005rays(64.25%) have>=2depth5%-consistent neighbors;43384(55.75%) also
have normal agreement. Uses global metric scale only, NOT canopy profiles.
Consistency is not absolute depth accuracy, independent correspondence or
proof of source-model error. Uniform rays are NOT a thin-rail-specific sample;
nearest cameras can have weak angular diversity. Audit source:
scripts/audit_moge_rigid_multiview.py, nonidentity camera roundtrip test passes.
Artifact audit_v477_moge_rigid_multiview/audit.json. No geometry changed.

Rechecked historicalv236b/v237 before considering rigid footprint corrections:
22528local atlas nodes on88independently attributed surfaces; only4negative
nodes, suppressing these did not improve tree metrics. Do not redo this as a
new discovery or use it to justify blanket surfel shrinking/deletion.

## v473/v474 matched 400 and v478 startup correction

Matched 400-step native-RGB minus 640-RGB results: tree +0.03083064 dB,
interior +0.02885630, boundary +0.04596921, rigid -0.01195089,
hard -0.05989261. Viewed both 660 panels: no clear visual breakthrough;
lower canopy leakage and poor railing detail remain. Do not confuse the
native arm's +0.65991282 versus historical source with its incremental gain.
Control v473 completed 800; native v474 still running on GPU2. Started v479
read-only full-native 1920 RGB/region evaluation of BOTH 400 checkpoints,
sequentially on GPU1 (launcher script launch_v479_native_rgb_audit.py).

v478 never started training: the launcher supplied an external checkpoint in
the child --resume argument, conflicting with supervisor-owned checkpoint
validation. Its DEVNULL startup stderr hid the failure; this was not OOM.
Preserved its empty supervisor directory. Fixed launcher to use
--initial-resume-checkpoint, persist startup stdout/stderr, await and check
the supervisor startup result. New distinct v478b launch returned started,
supervisor 2849 / training 2853; observed seed initialization progress.
Exact resumed update/refresh equality remains pending, not yet validated.

Rigid-consistency audit nearest-neighbor helper now explicitly masks the
self diagonal; coincident camera centers previously could include self as
support. Synthetic test added. Actual 304-view camera inputs have ZERO old
self neighbors and ZERO changed neighbor sets, so v477 counts are unchanged;
this robustness bug is NOT an explanation for reconstruction failure.
Supervisor/refresh/rigid targeted suite passed 37 tests before the extra
coincident-camera test; latest git diff --check clean.

### v478b completed resume verification / v480-v481 next bounded pair

v478b completed and published candidates_0008.pth and frozen audit.
CPU recursive comparison against continuous v476: refreshed observation
tensors/history, training order, horizon, layout, counters, torch/CUDA RNG
and Python RNG equal; NumPy array state was not compared by this helper.
Not bitwise identical: maximum candidate position-code difference 3.04e-5,
source opacity-logit 2.53e-5; SH 2.38e-7; Adam first moments <=1.19e-11.
Step8 before/after full losses identical at printed precision. Mean tree
identical; interior/boundary differences <1e-7 dB, worst reported individual
region difference 3.82e-6 dB. This supports numerical resume continuity,
not a bitwise determinism claim. CUDA reduction ordering is a plausible
explanation for tiny differences, not independently proven by this check.

Started v480 fixed-order / v481 per-epoch shuffle, otherwise same clean
640 RGB recipe, 800-step horizon, evaluation/update audit400/800. First
304 visits identical, divergence only from second epoch. GPU1 PID5335,
GPU2 PID5370 confirmed live. Native RGB v474 continues separately GPU2;
v479 native read-only audit on GPU1 completing its second48-view arm.
Schedule tests3 passed. Added native-audit comparison rejecting mixed image
resolution, changed source render and duplicate cameras; tests4 passed.
Archive v480_v481_schedule_sources.tar.gz preserves changed trainer/helpers.
No production checkpoint or native renderer replacement.

### v479 full native-resolution paired audit completed

Both 400 checkpoints evaluated on the same48 cameras at1920x1080.
compare_v479_native_rgb400.json validates identical source render, camera
identity, resolution and audit implementation. Native-trained minus
640-trained mean PSNR: tree +0.08188436, interior +0.08389986,
boundary +0.08828582, rigid -0.00727459, hard -0.03109272 dB.
Largest hard regressions:632 -.24535,674 -.19783,634 -.15200,
633 -.14775,677 -.12302. No promotion.

660 lower leakage ROI at native [1410,615,1890,1005]: control PSNR13.05275,
native13.16655; surface contribution .378146 -> .370706; volume
.602444 -> .611266. Thus there is a small actual occlusion change, not
only color fitting. It is still far from resolving the visible leakage.
Inspected full native panels (display downsizes5760x1080 to2048x384):
large lower-canopy streaks and railing blur remain, no claim of resolved
native-pixel detail from that downscaled preview.

Latest live check: v474600/800, v480 started actual training; v481 still
initializing. All tasks remain restricted to physicalGPU1/2.

## v482 conditional rigid geometry measurements / SH inventory

Source checkpoint CPU check: SH degree3, maximum directional coefficient
norm2.8876574, ZERO rows exceed trainer norm cap4. This does not support an
initial global SH-clipping explanation for the observed blur. Stored support
and observation tables are both N x4; these are not complete observation
design matrices for the15 directional SH coefficients. Do not freeze poorly
observed SH modes solely from this truncated historical table.

Extended training-only rigid MoGe audit with >=1degree source-neighbor ray
angle, on top of5%depth and30degree normal consistency. Exported measurements
require at least two such neighbors; each remains a conditional single-depth
hypothesis, not a fused point or proof that the source model is wrong.
Native pixels, camera-z, world position/normal and neighbor count are retained
for subsequent source-geometry error measurement. CPU-only runv482 launched
at audit_v482_supported_rigid_measurements; output pending at this entry.
No geometry parameters changed. Baseline-angle and self-exclusion tests3
passed. Existing304-view uniform rigid sampling still does not specifically
measure thin railing quality or guarantee independent correspondence.

v482 completed:41676 measurements meet two depth5%/normal30degree neighbors
with each source-neighbor baseline>=1degree. Export artifact
audit_v482_supported_rigid_measurements/conditional_rigid_measurements.npz.
Started v483 GPU1 read-only source-surface median camera-z comparison on16
uniformly selected training cameras, native1920x1080. Foliage contribution
disabled for this surface-only diagnostic; no parameter updates. Only opaque
surface samples(alpha>.9) enter relative-depth summary. Differences are
conditional disagreements, NOT proven source-geometry errors. Run handle62231;
script audit_supported_rigid_geometry.py. Outcome pending.

v483 completed normally:2167opaque surface samples across16training views.
Median absolute relative source/MoGe differences:559 .0071,579 .0196,
598 .0333,618 .0610,641 .3440,665 .9000(only25opaque samples),
701 .1281,737 .2443(ONE sample),764 .1935,789 .0568,809 .0843,
829 .1628,849 .0573,869 .0122,888 .0386,908 .1231.
Overall source/MoGe depth ratio percentiles0/10/50/90/100:
.6565/.9078/1.0214/1.1951/2.1908. Camera665 disagreement is not evidence of
an erroneously NEAR foreground blocker (ratios must be inspected, not assumed).
These conditional comparisons do not distinguish locally biased monocular
depth from source-model error. Nearest-camera consistency with1degree minimum
is too weak to resolve all such ambiguity. Next geometry supervision work
should seek larger-baseline independent evidence before changing buildings;
do not silently use these measurements as ground truth.

## v484 wider baseline and correction to v483 interpretation

Added configurable neighbor count/baseline threshold; launched CPUv484 with
24nearest training cameras and5degree minimum, same depth/normal tests.
Run audit_v484_wider_baseline_rigid_measurements, handle34537, outcome pending.

Visualized v483 sample locations on559/641/665 in disagreement.png.
The large641/665 differences are predominantly broad walls/windows, not
just missing railings. CRITICAL historical-context check: existing v155
canopy profiles are5591.0118179,6411.3450354,6651.8920460,
701 .8720542,8291.1656363. These already explain the corresponding raw-global
ratios. build_dense_canonical_canopy_diagnostic.py fits these profiles from
opaque rigid anchors, then applies them to canopy depth; current candidate
training also uses these profiles. Therefore raw-global disagreement is
NOT a newly discovered uncorrected canopy root cause. It is a warning for
the proposed extension of MoGe supervision to all rigid geometry, which
cannot reuse raw global scaling blindly. Applying profiles fitted to the
same source geometry cannot independently validate that geometry either.
Avoid circular 'correctness' claims based on matching these anchors.

v484 stopped intentionally with SIGTERM(PID8917), NOT OOM:24-neighbor
query with16-frame LRU repeatedly evicts required frames; at4m21s only
21/304progress reported. Retained incomplete output directory. Expanded
CPU cache to max(16,2*neighbor_count),48frames for this audit; identical
geometric criteria. Started distinct v484b directory/handle35295. GPU
trainers untouched. Latest steps:v474700,v480400,v481125; all live.

## v484b / v485 correspondence outcome

v484b completed:43877measurements have at least two neighbors passing5%
depth,30degree normal and5degree source-neighbor baseline among nearest24.
More neighbors changes available support; this is not a matched count
comparison proving higher accuracy versus v482. Geometric self-consistency
alone can accept a displaced plane, especially without texture correspondence.

v485 adds actual training RGB patch reprojection for v483's16cameras at640
RGB resolution. Three hypotheses: raw-global MoGe, rigid-anchor calibrated
MoGe, source surface median. EXACT same valid neighbor set per ray for all
three, >=2unique neighbors; mean cost across those neighbors (not independently
choosing best views per hypothesis). Existing patch helper called with one
duplicated neighbor only to obtain a per-neighbor score; that duplicate never
counts as independent support. Tests2 passed for shared-neighbor selection.

1802comparable rays: winning counts raw124,calibrated418,source1260.
Ray-weighted mean costs .50600108/.33371944/.28188567. Source surface has
lowest mean on all16views, though calibrated MoGe wins more individual rays
on888. This supports protecting current broad rigid geometry and examining
local alternatives, NOT blanket MoGe depth replacement. Costs are conditional
plane-patch photometric evidence, not ground-truth 3D errors; repeated texture
and occlusion remain limitations. Added per-ray depths/costs/pixels to audit
output; v485b rerun(handle49884) records those for future selective analysis.
No geometry or renderer changed; v474775/800, v480600/800, v481225/800.

v485b completed and recorded matched_ray_costs.npz. Calibrated MoGe beats
surface cost by>.01 on231rays,>.05 on71,>.1 on29; respective median relative
depth changes1.016%,1.963%,2.825%. These margins are descriptive, not calibrated
confidence or authorization to modify buildings. Most local opportunities
are small corrections, not a wholesale geometry replacement.

## v473/v474 800-step pair completed

Both supervisors completed returncode0, frozen_parameter_audit unchangedtrue.
Matched640-resolution final delta native-trained minus640-trained:
tree+.02583774,interior+.02289824,boundary+.03600568,
rigid-.01349982,hard-.07218264 dB. Native absolute tree16.23382090;
its+.90890368 versus old source is NOT the incremental resolution benefit.
No adoption. Started sequential native1920x1080 48-view audits for BOTH800
checkpoints onGPU1, v486, handle8690; launcher now accepts --step800
--audit-version486. Final native-resolution result remains pending.

## v480/v481 matched400 / longer convergence check

v480 completed800, frozen audit passed. Matched400 shuffle-fixed differences:
tree+.05253484,interior+.05258516,boundary+.06180378,
rigid-.00118226,hard-.01126965. Short shuffle arm still running, no adoption.
The fixed arm's400->800 tree gain is about.25dB;800image visits are only2.63
passes over304views, not800observations per primitive. This does not establish
convergence. Started clean v487 fixed-order3200 onGPU1(PID13950), identical
protection and representation; a new horizon is NOT an exact800-step resume.
v488 shuffle3200 launcher prepared, but MUST wait for v481 completion/frozen
audit; not launched yet. launch_v487_v488_schedule3200.py explicitly selects
one arm, preservingGPU0 and avoiding another concurrentGPU2 training job.
Long-versus-short gains will include changed exposure budget/LR horizon;
only matched3200-arm comparisons isolate the order policy.

Read-only renderer cutoff check: native MIN_RENDER_ALPHA=1e-6, volume radius
3sigma; optional surfel TIGHTBBOX=0. Thus initial candidate alpha.001 is not
globally killed by an old1/255cutoff. No new cutoff bug found or shader edit.
Latest v481525/800, v487 initializing; v486second native arm33/48views.

v486 native1920x1080 final800 A/B completed: tree+.08223985,
interior+.08322169,boundary+.08661600,rigid-.00780878,hard-.03770747.
compare_v486_native_rgb800.json. Native resolution branch not extended/adopted.
660lowerROI control/native PSNR13.43741/13.57878;
surface contribution .360774/.352409,volume .622824/.632722.
Inspected native final660 panel(display reduced to2048wide): persistent lower
canopy wall streaks and poor railings remain. Small real occlusion gain,
not a solved reconstruction. Long fixed-order v487 continuesGPU1; wait for
short shuffle v481 finish before launching v488GPU2.

## v481 completed / v488 launched / building veto remains

v480/v481 both completed returncode0 and frozen audits true. Matched800
shuffle-fixed:tree+.00650136,interior+.00035226,boundary+.03970122,
rigid-.00739948,hard-.01954559. The400-step ordering advantage largely
vanished. Worst hard deltas shuffle-fixed:738-.217127,909-.184294,
772-.124213,677-.120901. Critically,738 versus source is fixed-.958888,
shuffle-1.176015dB. Inspected738GT/render; no building-safe adoption.
Started scheduled clean v4883200GPU2 after short arm terminal, PID16894;
v4873200GPU1 PID13950 remains live. Longer runs are convergence diagnostics,
not certified safe maps, and must not be promoted if this boundary loss remains.

Rechecked old boundary experiments before proposing another guard: v313/v314
separate source-reference boundaryRGB weight1 improved738 by~.1045dB but
lost mean tree.0655dB and did not solve the problem. Current conservative
visible-rigid alpha mask excludes hard boundary explicitly; that is a designed
uncertainty policy, not newly proven renderer bug. Existing v431/v432 rollback
already showed most738damage comes from candidates, especially their sizes,
not mainly sourceSH/staticXYZ. Do not rediscover these as fresh conclusions.
Future boundary work should distinguish observed RGB risk budget from preserving
all source RGB changes; simply retrying the old boundary weight is not justified.

## v489 observed boundary-risk prefix

Implementation inspection: declared_objective ALREADY switches the independent
boundary term to observed_rgb_risk when noncanopy_rgb_target=observed_risk.
No new objective implementation was required. Current weight0 disables it;
historical v313/v314 used source-reference preservation, not this risk budget.
Started v489 weight1 onGPU1 PID19073, same immutable v480 control code/source/
initialization/config otherwise. Both original800-step LR horizon; treatment
stops after400 to compare against existing v480400, avoiding a duplicate
control training. v489 is a bounded configuration test, NOT yet effective fix.

Comparison now permits differing stop_after only for an executed common prefix
within each declared horizon; steps/LR horizon and all training arguments still
must match except the explicitly controlled weight. Tests cover valid prefix,
too-early stop rejection and mismatched horizon rejection. Comparison/boundary
suite7passed. Additional observed-boundary objective tests verify no penalty
for RGB improvement, no boundary-area dilution and no gradient to target/source;
observed-risk suite8passed,1CUDA skip(CPU-only invocation). Training replay400
will check actual complete-objective update. No live trainer/helper edits.
v487525/3200,v488175/3200 at launch; GPU0 remains unused by these experiments.

Follow-up full CPU regression:1475passed,41skipped,16warnings(20.22s), with
CUDA_VISIBLE_DEVICES empty and environment library path set. GPU-dependent
skips are not evidence of CUDA correctness. v489 published manifest confirms
only boundary weight0->1, output path and stop_after None->400 differ from
v480. Full comparator passed manifest/implementation/camera-input checks but
could not compare step0 RGB because v489 metrics.json is not published yet;
this is pending evaluation publication, not a training failure. No comparison
artifact was emitted by that attempt. Observed GPU1/2 memory9893/6577MiB,
utilization100/99%; trainingPIDs13950,16894,19073 confirmed live.
Latest steps:v487625/3200,v488325/3200,v489pre-first-step initialization.

## Long-horizon800 and reference-render efficiency

v487/v488 matched400 reproduces short-horizon early behavior: tree+.05250714,
hard-.01130094 for shuffle-fixed. Matched800 now published: fixed tree
16.31717279,rigid16.60690095,hard14.78268154; shuffle tree16.31051677,
rigid16.58970465,hard14.74220333. No meaningful order-policy advantage.
Fixed long-horizon800 gains~.109dB tree over short-horizon800 but worsens
hard boundary; same visit budget, DIFFERENT LR schedule, not a pure duration
effect or building-safe improvement. v489 reached actual optimization,
step25boundary loss.0016311, so its enabled term is active; outcome pending.

Reference cache audit: v480 capacity16, hits1/misses800; v481 hits3/misses798.
Trainer separately renders frozen RGB and surface-alpha references, and this
small LRU cannot retain a304-view epoch. Prepared independent
FrozenReferenceBundleCache(unaltered legacy helper/trainer) with bounded CPU
storage and defensive output copies;3CPU tests pass. GPU2 read-only v490
probe on training views559,560,559: old4renders vs bundle2; RGB/alpha maximum
absolute differences ZERO for every result. Timings2.208s/1.143s are a tiny
probe, not end-to-end speedup (new timing also includes equality checks).
Artifact audit_v490_reference_bundle_cache/audit.json. New cache is NOT YET
integrated into training, and does not explain or fix canopy leakage.
Current paired training implementation left unchanged for valid comparisons.

## v489 completed; shaped-component diagnosis

Matched400 observed boundary-risk1 minus control0: tree-.07861471,
interior-.06481214, tree-boundary-.12363724, rigid+.00595186,
hard+.01869486dB.738 hard recovers+.27837086 but remains-.41869545
versus original source. This does not satisfy building preservation and is not
adopted or extended. Frozen audit unchanged=true; actual full-objective update
at400 decreases sampled-view loss .06616165->.06452210, not evidence of
multi-view convergence or visual success.

Extended read-only component rollback to anisotropic candidates. Full initial
size rollback resets BOTH shared scale_code and deviatoric shape_code; separate
code/shape modes keep the other learned component. Due to tanh coupling these
are nonadditive code interventions, not pure volume/selectivity attribution.
22CPU tests pass including actual nonlinear scale reconstruction and exact
restoration after simulated render failure. Trainer and its live helpers remain
unchanged. Started audit_v491_shaped_components_v480_800 on physicalGPU1,
all48 regression views; read-only, no production replacement.

Long matched1200: shuffle-fixed tree+.05214373,hard-.00356332;
source-relative tree fixed+1.09862518/shuffle+1.15076892, but rigid
-.06161034/-.05954383.738hard source-relative-1.45293427/-1.17331886.
Viewed v4881600 RGB pairs660 and738:660lower wall streaks remain;
738canopy blurs/overcovers building. This is not solved by uniform thickening,
nor evidence to accept long-horizon models. v489 supervisor returncode0
confirmed; no abnormal stop or OOM for that run.

v491 completed all48 views, joint_components.json. Relative to full v480800:
disable candidates mean tree-.57186178,738hard+.86728382,660tree-.53785992;
reset BOTH size codes mean tree-.27840694,738hard+.52816963,660tree-.31102085;
reset shared scale_code only mean tree-.18751375,738hard+.35962963,
660tree-.23097706; reset shape_code only mean tree-.01441763,
738hard+.04460907,660tree-.01382542. SourceSH reset738hard-.00895596;
sourceposition reset-.02511024. This confirms the historical candidate-expansion
tradeoff on the current shaped model, not a newly discovered renderer bug.
Blanket scale rollback is NOT adopted: substantial useful canopy contribution
is removed too. Future growth restrictions need current TRAINING-only evidence
of harmful expansion, not738evaluation pixels or old shape eligibility as truth.

## v492 current growth derivatives (running)

Added standalone training-only derivative audit to the read-only contribution
entrypoint; no trainer/helper or production changes. At exact v480800 state,
aggregate per-view tree L1 and rigid observed-risk derivatives with respect to
candidate shared scale_code over all304 training views. Other parameters stay
frozen; no optimizer. Positive derivative means locally shrinking that code
reduces that particular loss, NOT that a finite rollback is safe. Excludes48
regression cameras from signal generation; tracks nonzero-view counts and
restores requires_grad while checking scale tensor unchanged.2CPU sign/input
tests pass. Started audit_v492_current_growth_directions GPU1 PID32197;
confirmed live and first training view completed. This is current-state evidence
collection, not yet a repair or a full-objective/Adam gradient audit.

Prepared finite growth counterfactual (not yet run): exact snapshot/code and
training/excluded-view contracts required. Only positive shared-scale codes
with BOTH net tree/risk derivatives positive and >=2 nonzero training views
per derivative are selected. Probe25% and100% rollback of selected code toward
zero, preserve all other parameters and restore even on failure. These finite
steps require48-view verification; no promotion or deletion from derivative
sign alone.31CPU tests pass across growth and component helpers.

Long matched1600 shuffle-fixed tree+.03085657,rigid-.01167033,
hard-.03495051. Source-relative tree+1.21060675/+1.24146332, rigid
-.10905731/-.12072764. Thus the convergence experiment still does not produce
a building-safe map, despite all48 shuffle tree regions exceeding old source.

v492 completed304training views.168687positive shared scale codes;
127993have net tree derivative<0 and rigid-risk derivative>0 (local expansion
conflict),9006have both derivatives>0 (before >=2nonzero-view filter).
Mean-gradient L1 tree.00086429507,risk.00008629521. This suggests simple
joint-benefit shrink has limited reach; it does not prove the remaining conflict
is geometrically impossible. Started v493 finite shrink probes GPU1 PID3206,
audit_v493_joint_shrink_counterfactual. Exact evidence/snapshot matching,
48regression views,25%/100% selected-code rollback; no model export. Also save
GT/render pairs for660,657,690,738,674. Intervention helper hash now recorded.

v493 completed48 views;8409candidates pass the distinct-view filter.
Quarter/full selected shared-code rollback mean tree-.00023288/-.00235218,
rigid+.00004085/+.00019445,hard+.00023095/+.00102564dB.
660tree-.00002098/-.00012016;738hard+.00013256/+.00109291.
Inspected660full-rollback RGB pair: wall streaks remain. Negligible gains and
small tree losses; NOT adopted or extended. This closes the simple jointly
beneficial shared-scale shrink branch at this state.

v494 CPU semantic-patch census: existing photometric depth score samples nine
points at offsets{-2,0,2}, but only neighbor CENTER semantic membership is
checked, with no source patch semantic weighting. On all valid tree centers in
187seed TRAINING cameras,5.693% have at least one sampled non-tree mask point.
This is a potential boundary matching weakness, NOT established incorrect-depth
rate, not actual priority-selected seed-ray distribution, and not evidence that
it explains dense-interior leakage. Coarse semantic masks can miss mixing.
Artifact audit_v494_training_patch_semantics.json; CPU-only, no geometry edits.

Long matched2000 shuffle-fixed tree+.01656717,rigid+.01489431,
hard+.03462444. Source-relative tree+1.27108775/+1.28765492,
rigid-.14667076/-.13177645. Still not building-safe.

Next geometry-capacity hypothesis (NOT implemented or launched yet): native
RGB alone retained candidate initialization at1.2 runtime pixels per sigma and
fixed1024rays/seed view. Spatial bounds permit .5x--2x scale, so smaller resolved
leaves and denser placement have not been isolated by that resolution test.
Reviewed recent/historical docs: no4096-ray/smaller-initial-sigma paired test
located. A controlled denser/smaller recipe should preserve approximate initial
projected optical area (4xcount,half sigma), explicitly preserve physical motion
extent (radius sigmas doubles), and use native RGB in BOTH arms to avoid the
runtime raster filter confound. This is a coupled representation-resolution
test, NOT an isolated density effect or exact optical equivalence. Old selective
shape cohort cannot be reused across changed candidate identities. Do not
hot-edit active training source or claim this planned experiment has run.

Prepared independent canopy_representation_resolution.py (NOT integrated into
trainer). Actual187seed populations imply185838->637524rays,3.4305x rather
than4x;52views are capped below4096. Uniform half-sigma would give nominal
projected area ratio.857634 overall. Recipe therefore provides per-seed low
initial peak .001/actual_area_ratio (.001--.004), only to equalize the nominal
small-alpha area budget; NOT exact rendered-alpha equivalence or opacity target.
Physical motion extent remains pixel_sigma*radius_sigmas=9.6 in both recipes.
9CPU tests pass including capped populations. Both planned arms require native
RGB/native kernel and no reused shape cohort. No new training launched yet.

Matched2400 shuffle-fixed tree+.03785886,rigid+.01502566,hard+.04059476.
Source-relative tree+1.31549338/+1.35335225,rigid-.16848667/-.15346102,
hard-.01780357/+.02279119. Longer fitting still trades building quality for
canopy mean; no adoption. v488 reached3200 updates but final metrics/frozen
audit were not yet published at check; PID16894 remained live, so this is final
evaluation/publication in progress, not completion or a stopped job.

v488 completed normally: supervisor returncode0,state completed; frozen audit
unchanged=true. Final3200 tree16.71850026,rigid16.43146020,hard14.62238552.
Relative to candidate-initial step0:tree+1.36824495,rigid-.17203726,
hard-.02141082 (this is NOT original-source-relative). Inspected660final:
lower canopy wall streaks and poor railing remain. No adoption.
Started v495 read-only native1920 endpoints on now-freeGPU2, launcherPID9595:
v480800 andv4883200, all48views with RGB pairs. Both horizon/order differ;
comparison is descriptive practical endpoint quality, not causal attribution.
v487 fixed3200 remains runningGPU1; no active trainer edits.

v495 native endpoints completed both48-view audits. v4883200-v480800:
tree+.49740438,interior+.48306215,boundary+.51077608,
rigid-.19292563,hard-.17941254dB. Worst hard674-2.45256901,
694-2.26125717,677-1.70439911.660tree+.44709778,hard-.41229057,
canopy surface contribution-.07682699. This is real native-resolution tree
improvement but not building-safe; duration/order both differ. Inspected674
triptych (viewer downsized5760x1080->2048x384): canopy remains blurred and
lower structure is obscured; not a native-pixel sharpness assessment. No
promotion. Compare artifact compare_v495_native_endpoints.json.

Created separate experimental entrypoint calibrate_canopy_resolution_candidates.py
from the unchanged current trainer. Original trainer hash still matchesv480;
no hot edits to v487 code/dependencies. New entrypoint requires explicit recipe,
native RGB/kernel, photometric-volume search, no old shape cohort; connects
per-seed sigma/count/low-opacity initialization and preserves nominal motion
extent. Explicit experimental capacity ceiling4M total leaves vs original2M.
Both recipes remain unverified candidates; source/building scopes unchanged.
CLI import/help passed. Started v496 coarse8 GPU2 PID12343 via supervisor,
maxrestarts0, memoryfraction.75; originalv487 still liveGPU1. v497fine not yet
launched. No quality claim from startup. This independent entrypoint allows
verification without modifying the still-running convergence implementation.

Representation comparison now recognizes the explicit coupled policy only
after validating each arm's derived rays/motion settings, per-seed sigma/peak/
area budget and candidate count. Seed populations/images, native sources,
implementation hashes and all other optimizer args remain matched. It does
not silently permit arbitrary count/scale changes.15CPU tests pass across
recipe, contract and existing comparison tests. v496 initialization progressed
through seed_views301 with no error at check; final update/frozen checks pending.

v496 coarse8 completed normally, supervisor completed and frozen unchanged=true.
Actual full-objective deltas at4/8 are-.001839586/-.0000815131; first native
update peak allocated4635.54MiB. These demonstrate execution, not quality.
Full CPU regression1503passed,41skipped,16warnings(21.84s); CUDA-dependent
skips are not CUDA verification. After coarse process exited, launched v497
dense_fine8 onGPU2 using the same independent trainer. v487 remains running
GPU1 at3075 at check. No source/production edits or model promotion.

v487 fixed3200 completed, frozen unchanged=true, supervisor completed.
Final paired3200 shuffle-fixed tree+.03079446,rigid+.00899506,
hard+.03067933. Source-relative tree fixed+1.36278858/shuffle+1.39358304,
rigid-.18584416/-.17684911,hard-.04680026/-.01612093. Both convergence arms
complete and rejected for building-safe adoption; no restart or OOM observed.
v497 fine initialization completed and first native update logged, peak
allocated6271.74MiB; full8-step/frozen checks still pending. Prepared but did
NOT launch v498/v499400-prefix pair: requires bothsmokes completed/frozen,
full matched smoke contract validation and unchanged experimental source hash.
Shared800-step LR horizon, stop400,eval/replay200,coarseGPU1/fineGPU2.

v497 completed normally and frozen audit passed (validated by formal launcher).
Actual full-objective4/8 deltas-.001975037/-.000100948. Matched8 fine-coarse:
tree+.00872978,rigid+.00318921,hard+.00610218; too short/small for a quality
claim. Full pair validation passed including per-seed recipe and original-source
RGB identity. Started clean v498 coarse400GPU1 PID15568 and v499fine400GPU2
PID15584, both maxrestarts0. Separate experimental trainer; original remains
unchanged. Archive v498_v499_representation_sources.tar.gz contains scripts/
outdoor code at launch (not a complete environment/native-binary archive).
No further trainer/helper changes while this pair runs.

v498/v499 both reached native training step1; manifests confirm557514 vs
1912572candidates, initial peak ranges.001 vs.001--.004. Matched step0 contract
passed. Fine-coarse initial tree+.00920880,rigid+.00335354,hard+.00570786.
Thus the8-step smoke tree difference+.00872978 is already explained in scale
by initialization differences; it is NOT evidence of better learning. Future
reporting must show initial versus trained differences as well as final quality.
Artifact compare_v498_v499_representation0.json. No accepted quality result yet.

Matched200 runtime640 fine-coarse tree+.00203802,interior-.00277734,
boundary+.01370800,rigid+.00363874,hard+.00860131. Tree difference is smaller
than initial+.00920880, so no demonstrated learning advantage at this scale.
Started v500 matched native1920 audits of both200snapshots,48views, onGPU1/2
alongside continuing bounded400 training. No extra training branch. Launcher
PIDs19737/19738. Need native output before deciding whether finer geometry
benefits were hidden by runtime downsampling; do not assume they exist.

v500 both native200 audits completed: fine-coarse tree-.02009680,
interior-.02321406,boundary-.01133712,rigid+.00009974,hard+.00034976.
660tree-.00245762,canopy surface contribution+.00005308;738hard+.02022362,
674hard+.04171371. No hidden native-resolution tree benefit demonstrated.
Keep only the existing400 horizon; do not extend from these intermediate data.

Rechecked depth-motion history before proposing a wider range: v260 ray-only
logradius.15 actually moved candidates median.50032 scene units but lost
.01895851dB versus v259. Therefore wider depth freedom alone was tested and
failed, not a new assured fix. Current CPU code-bound audit: at v4883200,
40530/557514 rows have any |tanh(position_code)|>.9,1430>.99; median/.9/.99
quantiles .5981/.8751/.9748. At v498200 zero rows>.9;v499200 only5/1912572.
This does NOT support the claim that most current candidates are stopped by
their position bounds. Depth uncertainty/material-size coupling remains a
modeling question, not an established dominant bug or permission to inflate
bounds without a fresh controlled hypothesis.

v498/v499400 both completed normally (returncode0), frozen unchanged=true.
Runtime640 fine-coarse tree+.01986440,interior+.01818448,boundary+.01263988,
rigid-.00342935,hard-.01733017. Source-relative tree+.63534017/+.65520457.
This is small, not an accepted fix, and worse building boundary than control.
Started v501 final native400 audits on now-freeGPU1/2 using unchanged render
audit implementation; generalized only the launcher step/version. No extended
training. Need final native evidence to close the representation-resolution pair.
