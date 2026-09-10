# Fixed-budget training RGB proposal allocation

Historical256->1024 rays/view increased canopy quality modestly but did not
remove transparency. Current557514 candidates plus1239507 source foliage stay
below the enforced2000000-row cap. Do not silently raise that cap or solve the
experiment by multiplying primitive count. Current lower660 ROI contribution
is dominated by candidates near the allowed size maximum, motivating a check
of allocation before widening individual footprints.

Optional seed_ray_policy=rgb_floor_priority (default uniform) retains the same
per-view ray count, three depth hypotheses, initial peak opacity.001, candidate
parameter bounds and all304 training cameras. Only training tree-interior
pixels with the already-required finite search depth are eligible. Render the
unchanged source with black foliage radiance and no sky; score positive
surface-RGB excess over observed training RGB. Sampling without replacement
mixes25% uniform probability and75% normalized excess-score probability. This
is not a guaranteed25/75 count split. All-zero scores or selecting every pixel
use the EXACT historical permutation. Invalid scores fail rather than silently
create geometry. Scores do not establish leaf existence or depth truth.

No evaluation image/ROI selects points. No inference mask or new verification
metadata. Sampling audits record training pixel population, positive-score
counts and selected pixel hashes separately from immutable image identities.
The2M capacity guard remains unchanged. Six CPU tests cover deterministic
sampling, unchanged zero-score/uniform paths, fixed counts/uniqueness, invalid
scores, and preservation of global RNG state. Comparison supports a single
explicit seed_ray_policy difference and checks the new helper hash.

Implementation prepared while v338/v339 continue; not yet scene-tested or
enabled in those already-running experiments. Choose an identical auxiliary
weight for the next matched sampling pair only after the current tests finish.

Project suite1363 passed,24 skipped before adding the metadata-only population
score mean. After v338 completed, v342 full557514-candidate8-step priority-seed
smoke launchedGPU2 alongside its lightweight read-only v340 audit. Auxiliary
weight remains0 in the smoke; this checks proposal routing and frozen-scene
invariants independently, not a combined efficacy claim. Archive:
v342_priority_seed_smoke_sources.tar.gz. No budget increase or map replacement.

v342 completed normally with all frozen checks passed and actual optical,
position and size updates. Candidate count557514 unchanged. Selected-ray mean
RGB-excess score.0532338 versus uniform expectation.00720161 under the same
per-view allocations, ratio7.3919. This verifies allocation, not efficacy or
leaf existence. No candidate promotion metadata was fabricated.

v343 uniform(GPU1)/v344 RGB-floor-priority(GPU2) launched800 images/eval400.
Both auxiliary weight0: isolate allocation from the approved auxiliary loss,
which so far trades additional tree fit for building damage. Same source,
camera cohort, rays/view, three depths, initial opacity and parameter bounds.
Archive:v343_v344_fixed_budget_priority800_sources.tar.gz. Keep following until
paired metrics, native leakage regions and building views have been checked.

At400, priority adds mean tree+.077941 dB, interior+.088958 and boundary+.063622
versus uniform; rigid-.001003 and hard-.005410. Absolute priority tree+.392793,
hard+.001211 versus original. Native660 still visibly leaks in the lower canopy;
do not equate sampling concentration or mean PSNR with solved occlusion.
Report:compare_v343_v344_0400.json. Current project suite1363 passed,24 skipped;
seed sampling6 and matched comparison4 tests also pass. Continue800 and native
transmitted-surface audit before deciding whether to retain this allocation.

Both800 runs completed returncode0 and frozen audits pass. Priority adds tree
+.079115 dB, interior+.081881, boundary+.077241 versus uniform, but rigid-.002576
and hard-.006055. Absolute tree+.545159, rigid-.008676, hard-.003836.46/48 tree
views improve versus original (not a blind heldout claim). Native660 still has
obvious lower-canopy see-through;738 still shows tree/facade contamination.
Report:compare_v343_v344_0800.json. v345 uniform/GPU1 and v346 priority/GPU2
read-only native surface-floor/candidate-capacity audits launched. No adoption.

v345/v346 completed.660 lower ROI uniform->priority: PSNR12.825838->13.027851,
surface contribution.394653->.383089, volume.588316->.600805, floor-MSE
.03789635->.03589319. This is real but small occlusion improvement.657 ROI RGB
improves16.669119->16.804941 while surface contribution slightly worsens
.216283->.217700;690 ROI PSNR16.815798->16.781996. No uniform benefit claim.

v347 priority+authorized feasibility weight1 launchedGPU1 against exact-code
v344 weight0. Same800-image horizon/400 evaluation, no stronger auxiliary.
Archive:v347_priority_authorized_auxiliary_sources.tar.gz. v348 read-only
priority capacity/near-opaque persistent+candidate probe launchedGPU2, never a
training target or accepted visualization. Awaiting user direction on a
separate bounded per-leaf static source-position experiment; not implemented.

v348 priority read-only near-opaque capacity completed.660 lower ROI candidate
peak1 counterfactual surface contribution.153492 (actual.383089), only113 pixels
with intrinsically projected alpha<.25;19207 with intrinsic alpha>=.95, of which
3268 still retain surface contribution>.25 in joint rendering. Near-opaque
existing persistent leaves as well only lowers surface to.150280; black-floor
MSE.00514753. This separates substantial unlearned extinction from residual
depth/coverage conflict; it is not a physical opacity prescription or accepted
render. v349 uniform identical capacity audit launchedGPU2 for matched context.

v349 completed. Uniform candidate-peak1 lower660 surface.246516 versus priority
.153492; missing projected pixels879->113, intrinsically dense15544->19207,
dense-but-surface>.25 pixels3587->3268. The candidate allocation improves this
counterfactual coverage but actual training realizes only a small part of it.
User subsequently APPROVED supported per-leaf static source-position control;
implementation/smoke tracked in cambridge_supported_static_source_position.md.

v347 priority+weight1 completed normally. At800 adds tree+.051412 dB versus
priority weight0 (absolute+.596571), rigid-.003770 (absolute-.012446), hard
-.009461 (absolute-.013297). At400 additional tree+.031606. Reports:
compare_v344_v347_0400.json and compare_v344_v347_0800.json.46/48 tree views
improve versus original, but building regressions remain; no production adoption.
Native transmitted-surface audit will use GPU2 after v352 finishes, while v353
source-position experiment runsGPU1. Do not conflate auxiliary with position.

v354 completed native auxiliary audit.660 lower ROI priority0->priority1:
PSNR13.027851->13.158184, surface contribution.383089->.375635, volume.600805->
.608581, floor-MSE.03589319->.03434608. Actual occlusion gain remains limited;
71.07% of its current ROI MSE remains below the nonnegative-color floor.
738 hard is-.922130 versus original, another-.120481 versus priority0. No
adoption. GPU2 proceeds with v355 source-position control native audit.
