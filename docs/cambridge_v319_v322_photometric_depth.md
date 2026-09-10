# Native size audit and training-only photometric depth proposal

v317/v318800 completed normally. Size learning gives additional tree+.113411,
interior+.127283 and tree boundary+.101919 dB. Absolute tree gain+.453445.
Rigid mean-.008082, boundary mean+.000368 versus source, but738 boundary
-.565689 (another-.171252 versus fixed size). Not approved for production.

v319/v320 matched read-only native audits also completed.660 lower ROI:

| Metric | Fixed size | Learned size |
|---|---:|---:|
| RGB PSNR |12.692867|12.810066|
| Actual structural contribution |.401426|.394295|
| Actual volume contribution |.581347|.588658|
| Fixed-state surface-only error floor fraction |.726280|.723205|
| Candidate contribution sum |413.925|1051.215|
| Contribution-weighted candidate size ratio |1|1.924498|

93.2636% of learned-size candidate ROI contribution comes from ratios>1.8.
This confirms modest real occlusion gain, but scaling alone does not remove
the residual building light. Do not increase the size ceiling to mask the problem.
These are limits of the current geometry/opacity state, not of all methods.

## New diagnostic hypothesis

Current seed depths are hypotheses around calibrated MoGe. Test whether selecting
depth by neighboring TRAINING RGB improves their placement. Source and four
nearest training cameras only; no48 evaluation views used.24 log-depth candidates
over [.4,1.2] times prior, in3 strata. Compare normalized3x3 sparse patch RGB
(offsets -2,0,2) under exact-K reprojection. Require two valid neighbor scores,
texture, parallax, a bounded best score and a distinct minimum in each stratum.
Unsupported, flat or ambiguous hypotheses retain their original deterministic
stratified proposal BITWISE. Chosen center and sigma scale together, preserving
source angular footprint. Colors and total candidate count remain unchanged.

These matches are initialization proposals ONLY, never depth truth, verification
metadata, opacity targets or extra training supervision. Wrong or occluded
neighbor projections are unsupported, not certified empty-space evidence.

Tests cover known plane parallax, no parallax, non-tree neighbors, ambiguous
minima, bitwise fallback and world-frame rotation/translation invariance.
v321/v3228-step paired smoke runs onGPU1/2: stratified_volume versus
photometric_volume, otherwise same as v318 (joint optics, bounded position/size,
guard3, extra boundary0, feasibility auxiliary0). Full557514 seed budget.
Archive v321_v322_photometric_depth_smoke_sources.tar.gz. Check real accepted
depth-slot counts and native frozen/gradient audits before full training.

Both v321/v322 smokes completed with frozen audits true. Photometric mode
reselected289870 of557514 depth slots. This indicates active proposal selection,
not correctness. A source-image edge padding issue was then found in the NEW
helper: incomplete source patches could yield finite matching costs from zero
padding. Added an explicit complete-source-patch validity requirement and test.
Seven photometric tests pass after this correction. The actual accepted count
on this scene is unchanged; the guard matters for general edge inputs. Old smoke
sources remain archived and are not relabeled as the corrected helper.

v323/v324800 launched onGPU1/2 using the corrected helper. Only depth policy
differs: stratified_volume versus photometric_volume. Other settings identical
to the smoke (800 steps/eval400). Archive v323_v324_photometric_depth800_sources.tar.gz.
Both camera JSON fingerprints match, as do camera-intrinsics contract hashes.
Comparator now also checks camera inputs when present, with a regression test
that changes a TRAINING camera while leaving evaluation identities unchanged.
No geometry/opacity pseudo-target or extra boundary loss enabled.

## v323/v324 completed800 and longer follow-up

Both completed normally. Photometric depth versus stratified: additional
tree+.033715, interior+.038773, boundary+.031430 dB. Absolute tree gain+.466096;
rigid-.006098 and rigid-side boundary+.002156 versus source.738 boundary
-.565984 (another-.141788 versus stratified control). This is a modest gain,
not a successful resolution of canopy transparency. Full reports at400/800.

800 optimizer steps cover the304 training cameras only about2.6 times; short
controls screen mechanisms but do not establish convergence. Next matched pair
uses3200 steps/eval800 and the photometric + spatial/size + slow-SH recipe.
The new controlled variable is source_opacity_lr .02 versus0. Both keep original
tree geometry and all building parameters fixed. With source opacity frozen,
new candidates must supply extra occlusion; source colors remain trainable.

Implemented an exact no-op source-opacity projection when learning rate0, plus
before/after fingerprint of ALL source logits. A zero learning rate alone must
not leave a post-step clamp enabled. Positive-rate projection is unchanged.
Tests cover repeated Adam0 steps with extreme logits and unauthorized rows.
The joint optical authority helper was not changed, preserving old audit loads.

v325 frozen-opacity8-step smoke launchedGPU2; v326 normal-source-opacity3200
control launchedGPU1. After v325 succeeds, launch v327 frozen-opacity3200 onGPU2.
Archive v325_v327_source_opacity_control_sources.tar.gz binds this implementation.
No source-geometry update, opacity target, or production adoption. Keep following
through shared checkpoints instead of ending at launch. Storage check:5.1TB free.

v325 completed successfully: all frozen audits passed, source-opacity update
remained false and its full fingerprint matched the original source. v327 is
now running on GPU2 alongside v326 on GPU1; GPU0 remains reserved for the user.

## v328 read-only candidate-gradient follow-up

The contribution auditor now loads diagnostic payloads on CPU and transfers
only render parameters to CUDA, avoiding unnecessary GPU allocation of saved
Adam state. This is an audit memory improvement, not evidence of a historical
training OOM cause. Candidate-only gradient auditing now accepts a joint-optics
checkpoint with its refined source parameters frozen, while retaining the
original immutable source for background RGB targets. Unsupported auxiliary
losses remain rejected. Saved candidate opacity moments are matched by parameter
name rather than optimizer group position. Moment disagreement with the
all-training-view gradient is diagnostic only, not proof of an optimizer bug.

v328 audits v324 step800 across the original48 views and304 training views,
with building radiance floors and candidate contribution weights. It is running
read-only on GPU1 alongside v326. Source archive:
v328_joint_candidate_gradient_sources.tar.gz. Focused tests:8 passed.

Project test suite after these changes:1348 passed,23 skipped. Unrestricted
repository discovery additionally collects a third-party tetra-triangulation
test that fails to import its unbuilt `tetranerf.cpp` extension (1352 other tests
pass). This dependency failure has not been linked to the active training path;
do not report the unrestricted run as fully passing or alter the third-party
test merely to hide it.

Visual review of v324800:632 still blurs fine branches;660 still exposes a large
building patch in the lower dense canopy;738 retains foliage/building edge
mixing. These images do not support a claim that transparency has been solved.

## v326/v327 first matched checkpoint800

Both native evaluations and snapshots are complete; training continues3200.
Frozen-source-opacity versus trainable source: tree-.102748, interior-.112177,
boundary-.068358 dB; rigid+.006074, rigid-side boundary+.016102. Absolute tree
gain is+.557980 for trainable source and+.455232 for frozen source.738 hard
boundary remains-.547123/-.514269 versus original source.711 tree regression
improves from-.131714 to+.080292, but660 tree loses.121661 versus control.
Thus blanket source-opacity freezing is not an established solution; it trades
canopy fit for a small average background improvement and does not remove the
worst building-boundary damage. Keep both runs through matched later steps.
Report:compare_v326_v327_0800.json. Main hashes match and the only non-output
argument difference is source_opacity_lr .02 versus0.

At1600, tree gains+.766309/+.643855 (trainable/frozen); freezing loses.122454
versus control. Mean rigid changes-.050249/-.041611 and hard-.050453/-.038388.
674 rigid collapses-.996773/-.831852; its hard boundary-1.095078/-.879926.
738 hard-.805946/-.786261. Thus longer fitting
raises canopy metrics but worsens protected building renders; neither arm is
eligible for adoption.660 visual still has lower-canopy building leakage.
Report:compare_v326_v327_1600.json. Continue the bounded diagnostic horizon,
but do not call higher tree PSNR a successful repair.

At2400, tree+.834592/+.700256; freezing loses.134336. Rigid-.071901/-.066368,
hard-.077156/-.080421. The mean hard-boundary benefit of freezing has disappeared
and reversed slightly. Neither arm meets the building-preservation requirement.
Report:compare_v326_v327_2400.json.

v327 completed3200 normally(returncode0). Frozen checks passed; source opacity
fingerprint1ff06ff21dec560245dc08bea7bba2d0fa6c3d2fe17930744cdbac0243e1e08f
equals original source. Final tree+.742740, interior+.809301, boundary+.592090;
rigid-.075502, hard-.087811. Not adopted. v326 still finishing for matched final.

v326 also completed3200 normally; both final frozen checks pass. Trainable/frozen
tree+.874831/+.742740, rigid-.079909/-.075502, hard-.081469/-.087811.674 rigid
-1.364841/-1.324816;738 hard-1.019643/-1.083924. Thus freezing is not a solution,
and the stronger tree score of longer training is bought with unacceptable
local building degradation. Neither promoted. Final matched report:
compare_v326_v327_3200.json.
