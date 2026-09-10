# Native canopy occlusion: v167–v173

Status: investigation and matched experiments in progress; no accepted solution.
Resource constraint updated by the user: all subsequent experiments, renders,
and CUDA diagnostics must use physical GPU 1 or GPU 2 only. GPU 0 is reserved
for the user's other work. At the resource handoff GPU 0 had no compute process
and 1 MiB allocated; the completed v173 pair and descriptor audits had exited.
All numbers below use 640-pixel native rendering, without semantic inference
gates. The 48 reconstruction validation views are excluded from these diagnostic
updates and new evidence construction, but are not historically held-out RGB.

## Completed continuation diagnostics

Both v167/v168 start from v159 iteration 1000, freeze all rigid parameters and
sky, and optimize leaf position, scale, opacity and SH3 appearance for 800 steps.

| Arm | Tree mean-view PSNR | Rigid mean-view PSNR |
| --- | ---: | ---: |
| Source v159 1000 | 15.177877 | 15.713120 |
| v167 native rigid extinction weight 3 | 15.363775 | 16.356733 |
| v168 native rigid extinction weight 0 | 15.687447 | 16.133093 |

These gains are not sufficient: the difficult view 660 still shows severe
translucency and loss of detail. Rigid geometry being frozen does not guarantee
rigid image quality: leaf contributions can still contaminate rigid pixels.
The higher tree score without negative extinction also costs rigid quality.

Qualification caveat: these older diagnostic arms allow exact-support-camera
measured single-view leaves to refine geometry/high-order appearance, as well
as persistent leaves. Production's high-bandwidth authority is narrower.
The v173 pair explicitly restricts geometry and high-order SH to persistent
leaves, with exact restoration and Adam-moment clearing outside authority.
Single-view DC/opacity updates remain restricted to their exact support camera.

## Read-only attribution checks on v159 1000

Dense regions are post-hoc diagnostic ROIs recorded in
`docs/canopy_dense_regions_v149.json`, not optimization targets.

| View | Native leaf alpha | Leaf alpha with rigid gate disabled | Difference |
| --- | ---: | ---: | ---: |
| 657 | 0.878625 | 0.904703 | 0.026078 |
| 660 | 0.840007 | 0.908339 | 0.068332 |
| 690 | 0.988609 | 0.989251 | 0.000642 |
| 713 | 0.940328 | 0.966874 | 0.026547 |
| 768 | 0.987282 | 0.997052 | 0.009770 |

Removing rigid visibility is an attribution counterfactual, not a repair or
evidence that the rigid geometry is wrong. View 660 has both visibility conflict
and insufficient intrinsic leaf extinction/coverage. View 690 is already
nearly opaque but lacks detail. A single global opacity explanation is false.

Contribution-weighted audit of the 0.35 leaf opacity ceiling found only 1.65%
of visible leaf responsibility at the ceiling in view 660 (1.50% in 657).
The ceiling is not the principal bottleneck at this checkpoint.

## Production pilots: early checkpoints, not final conclusions

| Arm at 500 | Tree | Rigid | Hard rigid |
| --- | ---: | ---: | ---: |
| v163 atomic measured split | 14.829335 | 16.044776 | 13.969591 |
| v166 broaden verified optical authority | 14.737747 | 16.407310 | 14.397508 |

Atomic split lifecycle correctness has only a tiny early visual effect.
Broadening optical authority improves rigid pixels but does not fix canopy
underfill at 500. v163 is scheduled for 3000, v166 for 1000.

## v173: explicitly experimental foreground occlusion prior

The evidence builder processes all 187 canopy-bearing cameras among the 304
calibrated training cameras; all 48 validation views are excluded as both source
and witnesses. The geometry-only cache contained 1,736,258 consistent pixels,
but semantic/depth agreement does not prove binary opacity. It is not accepted
by the new training evidence contract.

The revised cache contains 650,094 pixels and additionally vetoes backgrounds
that already explain observed RGB. Certificates require stable eroded canopy,
the full measured thin interval before independently opaque rigid geometry,
two other calibrated cameras with sufficient parallax and compatible depth,
and rigid-only RGB disagreement above the camera's rigid-anchor residual q90
plus 0.02, in both source and witness observations. Insufficient rigid anchors
produce unknown evidence. This remains a conservative hypothesis, not measured
ground-truth alpha; real-hole preservation must be inspected in validation.

Cache: `evidence_v172b_moge_occluder_rgb_veto_allcanonical` under the run root.
Manifest validates source checkpoint/seed hashes and complete matching camera
coverage. It does not yet fingerprint source images or masks individually.

Matched arms:

- `diagnostic_v173_persistent_joint_control800`: prior weight 0.
- `diagnostic_v173_confirmed_occluder_joint800`: prior weight 0.2.

Both use the same v159 1000 checkpoint, frozen rigid/sky, native rigid extinction
weight 3, persistent geometry authority, and identical 800-step schedules.
The added loss is mean negative log native leaf alpha only on certified pixels.
It does not change inference composition. It cannot create absent primitives
where alpha is exactly zero, and must not be described as a topology repair.

The actual CUDA loop has passed 100 steps with nonzero certificate loss and
matching camera sampling. Final quality is pending. The prelaunch full test
suite passed 1227 tests. Launch sources are archived separately for each arm;
later file edits do not retroactively alter live training modules.

## Follow-up: coverage bottleneck and 400-step results

v163 at 1000: tree 15.192294, rigid 15.709339, hard 13.741244.
Versus v160 at 1000 this is only +0.014194 tree, not a substantial repair.

v173 at 400: control tree 15.325716 / rigid 16.284697 / hard 14.334873;
occlusion-prior arm tree 15.330384 / rigid 16.278008 / hard 14.315240.
The added prior has only +0.004668 tree and slightly worse rigid scores so far.
Both continue to their scheduled 800-step endpoint.

Full native attribution audit `audit_v174_occluder_native_coverage_source`:

| Reporting region | Pixels | Pixels with native leaf alpha below 0.9 |
| --- | ---: | ---: |
| All observed canopy in 187 training views | 8,859,310 | 3,481,960 |
| Source candidates | 2,241,454 | 541,656 |
| Multi-camera confirmed certificates | 650,094 | 89,799 |

Certificates cover only 2.579% of low-alpha canopy observations. Conversely,
86.19% of confirmed pixels were already at alpha >= 0.9. This explains why a
large raw certificate count does not imply useful missing-occlusion coverage.
Low alpha includes legitimate leaf gaps and is NOT an error label. Training
neighbors 658/659/661/662 have 4,401–6,086 low-alpha certified pixels each, so
the certificate is not entirely irrelevant there, but its coverage is limited.

Next read-only test: native-grid MASt3R reciprocal RGB correspondences on nine
training-only pairs, with fixed-camera triangulation and rigid controls. This
checks whether depth-independent matches offer useful alternative geometry;
it does not change cameras, import predicted poses, or promote matches to
training evidence. Native pixels are preserved by bottom/right patch padding,
not rounded rescaling of match coordinates. Synthetic exact-K triangulation
including a rotated/translated camera passes to 1e-11. Descriptor hypotheses
still need correspondence and multi-view validation before any geometry use.

## v173 endpoint and v176 follow-up

Both v173 arms completed normally at 800. Control tree/rigid/hard scores:
15.363732 / 16.356852 / 14.399512. Confirmed-prior scores:
15.371376 / 16.348432 / 14.380226. The +0.007644 tree gain is negligible and
comes with slightly worse rigid scores; this is not a successful solution.

v166 completed normally at 1000; native tree/interior/boundary/rigid/hard:
14.986022 / 15.447762 / 13.113286 / 16.497747 / 14.511823. Broad optical
authority primarily improves rigid pixels, at a canopy cost relative to v163.

`diagnostic_v176_source_candidate_occluder800` is running on physical GPU 2.
It matches v173's source/schedule/rigid protection and persistent geometry
authority, but uses the explicit `source_candidate` evidence scope. Source
candidates retain erosion, stable measured depth entirely before opaque rigid,
and rigid RGB disagreement; they do NOT have the additional two-camera
certificate. This is a deliberately weaker prior experiment, not new ground
truth and not permission for globally visible single-view geometry. It tests
whether the observed 6x larger low-alpha coverage has useful effect. Weight
remains 0.2; validation is unchanged; launch sources are separately archived.

Descriptor cycle audit `audit_v175c_descriptor_cycles` closes native pixel
matches within one pixel BEFORE epipolar selection. Difficult triples
658/659/661 and 659/661/662 have only 22 and 9 canopy cycles, respectively;
their median maximum pair epipolar errors are 1.730 and 3.844 pixels, while
rigid controls are 0.317 and 0.319. No canopy track meets all three geometric
tests there. Conversely 686/687/688 has 95 canopy cycles, median 0.258 pixels,
and 56 geometric candidates (no valid rigid cycle controls in this triple).
These sparse descriptor hypotheses support localized inconsistency, but do
not by themselves prove wind, establish a dense motion field, or justify
dropping whole cameras. No descriptor geometry has entered training yet.

v176 at 400: tree 15.370230 / rigid 16.264930 / hard 14.300082.
Compared with v173 control this is +0.044514 tree. View 660 improves +0.137690
tree dB and native whole-canopy leaf alpha rises 0.621572 to 0.646676, but the
image remains visibly translucent. This is insufficient for acceptance.

`diagnostic_v177_source_candidate_occluder_w2_800` brackets the prior strength
at 2.0 (versus v176's 0.2) with all other objective/schedule/source parameters
unchanged. It is on physical GPU 2, alongside v176. This is a strength diagnostic
of a weaker source-only prior, not evidence that filling these pixels is always
correct. Native rigid extinction remains 3, geometry/high-order appearance
authority remains persistent, single-view permissions remain camera-local, and
all rigid parameters are frozen. Both endpoints are 800 with 400-step evaluation.

Latest full suite after the audit/selector changes: 1231 passed, 15 warnings.

v163 at 1500: tree 15.358049 / interior 15.794484 / boundary 13.511008 /
rigid 15.493685 / hard 13.622810. Tree continues improving, but rigid declines
monotonically through 500/1000/1500. This is a quality tradeoff, not acceptance.
The immutable full 1500 checkpoint and lossless render-only derivative are
preserved; native 51-view evaluation completed successfully on physical GPU 1.

v176 completed normally at 800: tree 15.418842 / rigid 16.335692 /
hard 14.359786. Versus the matched v173 control, +0.055110 tree remains modest.

`formal_v178_native_optical_resume1k_to3k` continues v166's immutable full
iteration-1000 checkpoint on physical GPU 2, preserving its 30k phase horizon
and all objective settings, with retained 1500/2000/3000 checkpoints. This is
a continuation, not another clean initialization. v163 remains the 3k control
on physical GPU 1. v177 is the short diagnostic also on GPU 2; prior measured
formal allocator usage leaves headroom for its approximately 4.2 GiB usage.

Launcher incident: passing an external checkpoint via ordinary trainer
`--resume` was rejected before a training child launched. Corrected to the
supervisor's existing `--initial-resume-checkpoint` mechanism; no validation
was disabled and no checkpoint was overwritten. The new child must still
pass trainer contract validation before continuation is considered successful.

## User-approved controlled tree deformation

The user explicitly approved a tree-only deformation experiment. Physical
GPU 0 remains reserved for the user's unrelated work. v178 has passed restore
and emitted iteration 1175; this is actual resumed optimization, not merely
a live launcher.

New isolated diagnostic: `calibrate_canopy_center_field.py` with
`BoundedCanopyCenterField`. Matched static rank-0 and temporal rank-8 arms
start from v159 iteration 1000, fit the same 304 canonical training cameras,
and exclude all 48 validation views. Only persistent tree centers can move;
all base leaf parameters, rigid parameters, sky and appearance are frozen.
The spatial grid spacing is 2 world units, with displacement norm bounded by
0.1 world unit. Colors, opacity and covariance have no deformation parameters.
This is a center-displacement diagnostic, not a full physical elastic model.

Canonical evaluation uses the shared spatial field alone. Temporal evaluation
linearly interpolates codes belonging only to training frames and is reported
separately; it cannot substitute for the static-map acceptance result. Better
temporal fitting alone would not prove wind motion or solve static occlusion.
Both arms use identical native RGB, canopy SSIM, sky and spatial regularizers.

Four unit tests pass, including an actual Adam update with immutable source
positions and exactly zero displacement of ineligible rows. The first CUDA
smoke correctly rejected an evaluator mismatch: raw composed RGB had not been
clipped to the deployment API's [0,1] output range. The diagnostic evaluator was
corrected; production rendering was not modified. A second smoke is in progress.

v177 completed 800 steps: tree 15.426592 / rigid 16.254426. Compared with
v176's weak prior, +0.007750 tree costs 0.081266 rigid PSNR. Increasing this
prior's weight is not an accepted solution to the canopy problem.

Second CUDA smoke completed: all 48 zero-field views agree with native API;
the first field gradient L1 is 0.0001377898643 and an Adam update produces
different rendered outputs. Final hashes confirm all base parameters unchanged.
Full suite after the diagnostic addition: 1235 passed, 15 warnings.

Launched detached paired 800-step diagnostics (400/800 evaluation), with
per-process CUDA allocator fraction 0.35 and no automatic restart:

- GPU 1: `diagnostic_v179_static_center_field800`, supervisor 10663 / child 10664.
- GPU 2: `diagnostic_v180_temporal_center_field800`, supervisor 10669 / child 10670.

Launch sources archived as `v179_v180_center_field_launch_sources.tar.gz` in
the run root. Results remain pending; the source checkpoint and production
training implementation are not changed by these experiments.

v179 finished normally at 800, with every base-parameter fingerprint unchanged:
tree 15.171848 / interior 15.631712 / boundary 13.273985 /
rigid 15.719515 / hard 13.762262. Source tree is 15.177877: shared center
refinement has no useful canopy gain. View 660 visibly retains wall leakage.

v180 at 400: shared static tree 15.177726 / rigid 15.715900; interpolated
temporal tree 15.167561 / rigid 15.725761. No positive motion-model result yet.
This bounds only this frozen-appearance, small-amplitude, coarse spatial field
experiment; it does not prove foliage motion absent or every deformation model
ineffective. Keep this branch out of production pending useful evidence.

Expanded read-only visibility audit with near-zero and low-alpha bins plus an
optional native volume-only counterfactual. v181 is running on GPU 1 over all
187 canopy training cameras from the same immutable source. This separates
missing intrinsic coverage from native rigid occlusion, without mutating any
geometry or treating disabled rigid visibility as a candidate solution.
Two new threshold/empty-region unit tests pass.

v181 completed all 187 cameras. Native tree alpha below 0.5: 1,587,841 pixels;
intrinsic (no rigid visibility) below 0.5: 897,383. There are 399,685 pixels
with native alpha below 0.5 and intrinsic alpha at least 0.9. These remain
attribution counts, not error labels; genuine holes are included. Source
candidate masks cover only 122,060 of the native-below-0.5 pixels, and confirmed
masks only 2,557. Only four candidate pixels have native alpha below 1e-6;
thus total zero forward coverage is not the primary explanation for the weak
prior's gradient within its selected source masks. Selection coverage itself
is strongly biased toward already-opaque regions.

A separate v182 read-only audit on GPU 1 now attributes attrition to individual
source gates, with cumulative and overlapping-independent rejection counts.
It emits no candidate/confirmed training files, and asserts that reconstructing
the cumulative gates equals the existing candidate mask exactly. No gate has
been relaxed in training.

Approximate geometric scale check (projected persistent centers in dense ROIs,
not visibility-weighted): 0.1 world-unit lateral displacement corresponds to
median 4.79/3.87/2.67/4.98/3.39 pixels in 657/660/690/713/768 respectively.
The current negative deformation result only covers this small-amplitude model.

v180 completed normally at 800, frozen base-parameter hashes unchanged:
static canonical tree 15.178271 / rigid 15.715785 / hard 13.750040;
interpolated temporal tree 15.173346 / rigid 15.727981 / hard 13.765993.
Neither this nor v179 provides a material canopy gain. Do not integrate the
diagnostic deformation branch into production on this evidence.

## v183 spent positive birth-witness repair (code, not yet trained)

Reproduced a distinct birth bug: `_available_support` removes spent first-hit
witnesses, but the depth solver and priority still included their positive
depth slabs. A consumed near hit at z=[0.10,0.12] incorrectly rejects two
remaining hits at z=[0.18,0.20] in the same coarse cell. Removing only the spent
positive constraint restores feasibility. This is not a claim that the spent
camera supplies no free-space evidence; separate negative-ray checks remain
necessary and unchanged.

The solver and depth priority now use available support cameras only. If
another selection spends a witness during the same drain, the candidate is
re-solved. An aggregate center containing spent evidence is replaced by a
neutral voxel-center starting point before the same feasibility check. An
irreversibly mixed color is returned as unknown (NaN), not measured RGB;
the existing append routine uses its finite nearby appearance fallback for
such rows. No new color authority is inferred. No rigid parameter is touched.

Two reproductions failed before the patch. Three new regression tests pass,
including same-drain witness consumption; 38 targeted birth/checkpoint tests
pass. The original solver remains bitwise equivalent on unspent inputs.
v163/v178 already imported the previous implementation and are intentionally
not modified or restarted. This change requires a separately identified
subsequent training experiment; final-image effectiveness is still unknown.

Real v163 immutable iteration-1500 checkpoint: 2,659,408 pending hull cells;
304,615 include spent support, and 52,723 still have at least two available
cameras. In an ordered bounded sample of 5,000 such cells, the initial patch
rescued 140 but lost numerical convergence for 7 previously feasible cells.
Added a historical-position warm-start fallback certified ONLY against the
remaining cameras, without relaxing feasibility thresholds. Repeating the
same sample: 4,760 previously feasible versus 4,903 now feasible; 143 newly
feasible, ZERO newly infeasible. This sample is not a random population estimate
and feasible proposals are not yet verified/published leaves.

An extra nearly-parallel-slab test and a real CPU foliage-append test pass:
unknown birth colors produce finite fallback features, keep all existing
parameters bitwise unchanged, and grant no verification count. The earlier
full suite was 1240 passed before these two extra tests.

v182 source-gate attrition completed 187 views. Native-alpha-below-0.5 pixels
remaining after each cumulative gate: canopy 1,587,841; eroded core 1,203,344;
valid depth 1,162,536; stable depth 1,161,270; background disagreement 553,964;
opaque rigid 376,267; entire depth interval before rigid 122,060. The depth
ordering gate removes 254,207 of the last group. This motivates checking depth
evidence quality, not blindly turning these rejected pixels into opaque targets.

v184/v185 are the next matched controlled field-scale pair, on GPUs 1/2:
spacing 0.5, radius 0.25, LR 0.008, rank 0/8, 800 steps, 400/800 evaluation.
Relative to the first pair, the grid resolves smaller branches and the bound
covers larger image displacements. Initial physical Adam displacement is kept
approximately matched; magnitude and spatial-gradient regularization are
rescaled to retain the previous physical strength (radius factor 6.25,
spacing factor 16). All colors, opacity, covariance, rigid and sky remain
frozen. This is a scale-controlled follow-up, not evidence of improvement.
Launch archive: `v184_v185_finer_center_field_launch_sources.tar.gz`.

Both finer-grid arms completed normally, exit 0, all frozen-parameter hashes
unchanged. At 800:

| Arm / evaluation | Tree PSNR | Boundary | Rigid | Hard rigid |
|---|---:|---:|---:|---:|
| Source | 15.177877 | 13.319877 | 15.713120 | 13.741587 |
| v184 static | 15.177292 | 13.280151 | 15.719926 | 13.760534 |
| v185 shared canonical | 15.178113 | 13.304151 | 15.717409 | 13.752636 |
| v185 interpolated temporal | 15.172036 | 13.178298 | 15.741603 | 13.796668 |

Still no meaningful canopy improvement; the temporal view is not promoted to
static-map acceptance. This second field has 25,139 nodes / 681,185 temporal
arm parameters. An asynchronous user question asks whether to proceed to a
matched shared-color/opacity plus deformation experiment; no such experiment
has yet been implemented or started. The approved position-only studies are
complete. Do not infer an answer to that follow-up question.

v178 iteration 1500 native 51-view evaluation completed from a lossless
render-only derivative (148 storages, 77.72 seconds extraction). Canonical48:
tree 15.220349 / interior 15.673865 / boundary 13.344385 /
rigid 16.572337 / hard 14.605951. Compared with v166 iteration1000, both tree
(+0.234327) and rigid (+0.074590) improve. View660 still visibly shows wall
leakage, so this is progress, not completion. Training continues towards 3k.

Latest complete tests after warm-start and finite-color append regressions:
1242 passed, 15 warnings. Existing `--allow-trainer-repair-resume` explicitly
allows a state-compatible `static_ray_birth`-only hash change while continuing
to reject CUDA/model changes; no new blanket resume bypass is needed for v183.
Any actual resume using this permission must still pass unchanged contract,
schedule, source and optimizer checks and be identified as a repaired run.

## User-approved joint optical / position-field control

The user answered "继续联合对照" to the follow-up. Implemented an opt-in
`--joint-shared-optics` mode in the isolated field diagnostic. Both arms will
optimize the same shared static center field, persistent leaf SH colors and
opacity; only the experimental arm adds temporal position codes/basis. Base
XYZ, covariance, nonpersistent leaf optics, all rigid parameters, sky and
appearance remain immutable. No temporal color or opacity parameters exist.

Shared SH receives only canopy RGB/SSIM gradients. Opacity and center fields
receive canopy/rigid RGB plus sky negatives and native rigid-extinction weight
3. The shared opacity cap is 0.35 and persistent high-order SH norm cap is 4,
applied only to eligible rows. These caps are identical in both new arms; the
old v173 unconstrained diagnostic opacity results are not an exact new control.
SH LR 0.0025 / opacity LR 0.02, field LR 0.008, with identical annealing and
0.5-spacing / 0.25-radius physical regularization.

An actual-Adam unit test confirms that gradient hooks and projection leave
ineligible rows bitwise unchanged. Full CUDA smoke of three optimization steps
and both 48-view rendering modes is underway before launching the paired runs.
Saved diagnostic field checkpoints also contain the shared optical tensors;
they remain explicitly non-production artifacts with source checkpoint identity.

The three-step CUDA smoke completed successfully, including both 48-view
evaluations and the final bitwise frozen-parameter audit. At step3/view607,
shared color gradient L1 was 0.107166 and opacity gradient L1 was 0.020683;
8,967 opacity rows increased. Full tests: 1243 passed, 15 warnings.
This establishes an active update path, not an image-quality improvement.

Launched the matched 800-step pair with detached, fail-closed supervisors:
`diagnostic_v186_joint_static_center800` on physical GPU1 (supervisor28938,
child28939) and `diagnostic_v187_joint_temporal_center800` on physical GPU2
(supervisor28944, child28945). Both evaluate at 0/400/800; all 48 validation
views remain excluded from fitting. GPU0 is not used. Source snapshot:
`v186_v187_joint_center_launch_sources.tar.gz`. Existing v163/v178 training
was not interrupted or modified by this launch.

Added `scripts/compare_canopy_field_controls.py` for matched per-view reports:
it rejects different fitting cameras, missing/duplicate evaluation views,
nonfinite metrics and mismatched initial native metrics. It reports both
change from initialization and experiment-minus-control, including medians
and worst views; alpha increase is explicitly not declared a quality gain.
Five new tests pass; full suite is now 1248 passed / 15 warnings.
Applying it to completed v184/v185 confirms the static mean tree difference
of +0.000821 dB hides 34/48 negative per-view differences (median -0.003278).
Output: `v184_v185_paired_view_report.json` under the experiment root.

### Joint controls: intermediate iteration400

| Mode | Tree | Interior | Boundary | Rigid | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| v186 static control | 15.455127 | 15.945821 | 13.413899 | 16.207523 | 14.215036 |
| v187 static canonical | 15.462207 | 15.942183 | 13.458021 | 16.197041 | 14.168026 |
| v187 interpolated temporal | 15.418256 | 15.944852 | 13.233872 | 16.233183 | 14.267320 |

The shared-optics static control improves tree +0.277250 and rigid +0.494403
from the source, but visual inspection of view660 still shows wall leakage.
Mean canopy native alpha declines from 0.746445 to 0.720807; the PSNR gains
must not be described as increased canopy density. The temporal arm has no
clear incremental canopy benefit: its static result is +0.007080 tree but
-0.047011 hard versus control (40/48 hard views lower). Its interpolated
render is -0.036871 tree / -0.180027 boundary versus control. Both experiments
continue to800; no production deformation integration is justified so far.

v186 completed800 normally (exit0, frozen tensors unchanged): tree15.471869,
interior15.965986, boundary13.415759, rigid16.287195, hard14.296924,
mean native tree alpha0.714853. This remains visually insufficient.

After v186 freed GPU1, launched `diagnostic_v188_static_joint_native1500_800`
(supervisor3250, child3251), applying the SAME static-only joint optimization
to immutable v178 iteration1500, whose rigid baseline is substantially better.
No temporal parameters or topology changes; all rigid parameters remain frozen.
`scripts/build_canopy_refinement_cohort.py` generated a SHA-bound camera-only
manifest with304 fitting /48 excluded cameras, explicitly NOT a depth/opacity
prior cache. It uses current dataset camera metadata and asserts checkpoint
stability during reading. No old v159 evidence is relabeled as v178 evidence.
Sources: `v188_static_joint_native1500_sources.tar.gz`. v187 continues on GPU2.

v187 completed800 with frozen-parameter audit unchanged. Static canonical:
tree15.484888 / interior15.967605 / boundary13.470106 / rigid16.273032 /
hard14.231142. Interpolated temporal: tree15.449606 / interior15.976820 /
boundary13.258663 / rigid16.310218 / hard14.345684. Against v186, the static
tree gain is only0.013019 while hard drops0.065782 (41/48 hard views lower).
Temporal tree is0.022263 lower and boundary0.157096 lower. View660 remains
visibly translucent. This field configuration is not selected for production;
these results do not prove all possible motion models ineffective.

Started read-only `audit_v189_front_interval_partition` on GPU2 after v187.
It extends source-gate attribution with a disjoint partition: full interval
before wall, center before but interval crossing, center behind but interval
crossing, full interval behind, invalid interval. Two CPU regression tests
pass. Same v159 source as v182, all187 canopy training cameras,48 exclusions.
No relaxed training masks, posterior probabilities, or renderer gates are
emitted. This tests whether conservative interval rejection excludes plausible
front centers; it does not label rejected pixels as errors or opacity GT.

v189 completed187 views. Of the254,207 low-alpha pixels rejected at the final
front-depth gate, only8,722 have a center before the wall but a crossing
interval;14,319 have a center behind and crossing interval;231,166 have their
entire interval behind the wall clearance. Simply relaxing the interval gate
is therefore not a compelling main fix. Launched read-only
`audit_v191_depth_conflict_panels` on GPU2 (supervisor19343, child19353)
for views771/661/765/659/762/773/
662/757, explicitly selected from the training set. Panels show GT/native/
rigid-only above, low-alpha behind-interval pixels over GT / relative depth
gap (red behind, blue before; saturation at50%) / native leaf alpha below.
It additionally reports depth gap quantiles, without choosing either depth
source as ground truth.

CPU solver optimization v190 groups a homogeneous batch by available depth
constraint count only if padded slots exceed twice the true slots. Projection
order/tolerances remain identical. Mixed-width/spent-witness regression matches
the existing padded solver bitwise. Synthetic20k timing:0.841s padded vs0.773s
grouped (~1.09x, not a main-training speed claim). Real immutable1500 candidate
metadata has widths2:468018,3:67092,4:1134,5:41,6:1 after available-camera
filtering, reducing padded slots3,217,716 to1,142,059. This does not establish
the cause of long full-training topology steps. Full tests1251 passed/15 warnings.
The v183 audit remains143 newly feasible/0 newly infeasible on its5000-cell
ordered sample. Neither running formal trainer has imported v183/v190 changes.

## v178 iteration2000 and follow-up diagnostics

v178 iteration2000 was retained and extracted losslessly (148 storages,
147.53s, render derivative1.445GB; original3.075GB unchanged). Native51-view
evaluation completed: canonical48 tree15.324917 / interior15.775400 /
boundary13.426374 / rigid16.608309 / hard14.638506. Tree and rigid remain
better than its1500 checkpoint, but canopy leakage persists.
v188 static refinement of1500 completed800 and frozen audit passed:
tree15.333092 / interior15.814632 / boundary13.316488 / rigid16.608898 /
hard14.714944. Small improvement, not a solved reconstruction.

v191 panels completed. Median MoGe-minus-rigid camera-z gaps on the selected
behind-interval low-alpha pixels are2.47–3.61 scene units across8 training
views, far above the0.25 center-field bound. View661's problematic pixels
coincide with shallow structure in the rigid-only proxy, including tree-like
artifacts. Neither proxy nor monocular depth is independently established as
ground truth here. Next diagnostic: native per-surfel rigid versus canopy
visibility responsibility across every non-validation training camera, then
only a read-only counterfactual for rows with zero rigid support and canopy
support in at least2 views. No production pruning or semantic rendering gates.
`scripts/audit_rigid_canopy_support.py` is in3-view CUDA smoke before full scan.

The v191 process was also observed waiting in `rpc_wait_bit_killable` while
opening MoGe runtime arrays; `/mnt/pool` is a hard-mounted NFS4.2 filesystem.
It later completed normally. This establishes an observed I/O wait, not a
GPU-memory failure or proof that all long training intervals have that cause.
No checksum checks were bypassed and no source assets were relocated.

Implemented and CPU-tested `mixed_order_darkening_bound`: for fixed background
contributors with per-channel radiance bounded by M and nonnegative leaf RGB,
insertion removes at most A_leaf*M background RGB, so visible alpha must obey
A_leaf >= max_c((B_c-C_c-slack)/M_c). This works with interleaved depths but is
weaker than the existing front-layer bound using B as denominator. Tests cover
a concrete counterexample to applying the front-layer bound to mixed order,
random12-layer insertion, detached targets and invalid ceilings. Actual audit
uses RAW RGB and the maximum over ALL surface SH colors plus sky (no quantile
ceiling), with rigid-anchor q95 residual+.02 slack. Read-only
`audit_v192_mixed_order_bound_2000` launched on GPU2 (supervisor22958,
child22959); no training loss or production renderer change has been enabled.

## v193–v195: authorized local rigid contradiction investigation

The user explicitly authorized local controlled rigid geometry/opacity
correction only after independent multi-view contradictions, with verified
buildings protected. No rigid corrections have been applied at this point.

v193 scanned all1439 non-validation views. Of1006817 rigid rows,1006261 had
some native known-rigid visibility. Only75 had zero rigid support and at least
two canopy-support views. Suppressing those75 in a read-only counterfactual
changed tree PSNR by exactly0 and rigid PSNR only by floating-point noise.
They were NOT production-pruned. Semantic visibility is not geometric proof.
The first three-view smoke failed because the diagnostic caller detached DC;
the caller was fixed, not the renderer. The complete audit checks per-view
responsibility mass conservation and restores all six rigid parameter hashes.

v192 global-contributor radiance bounds completed187 views:40939 deficit
pixels,31035 outside the old strict candidate gate and27966 wholly behind its
MoGe interval gate. Mean required/native visible alpha on deficits was
0.09533/0.03959. This is a weak necessary bound, not an opacity measurement.

v194 implements a stronger bound by capping EACH fixed surface/sky contributor
before blending at0.25/0.5/1/2, then taking the maximum of the four bounds.
Capping final blended RGB is mathematically incorrect and has an explicit
counterexample regression test. Actual per-view SH colors are baked into DC
only for diagnostic renders, with uncapped-native equivalence and exact color
restoration checks. The661 smoke found3828 deficits, required/native alpha
sums904.01147/381.00952. Full187-view construction launched onGPU2, child30531,
in `evidence_v194_capped_mixed_bound_2000`; no bound training loss is enabled.
The standalone builder does not use MoGe and thus does not load its large
runtime arrays. Source checkpoint, mask, RGB and frozen rigid identities are
recorded. These bounds remain background-model-dependent, not ground truth.

`audit_rigid_track_depth.py` now screens actual reciprocal-descriptor tracks
against native rigid median depth, independently of MoGe. Tracks require
three distinct allowed cameras, descriptor cycle support, subpixel archive
reprojection and at least1-degree low-percentile triangulation angle. Any
track touching a validation or unavailable camera is excluded in full.
Native reprojection, opacity and diagonal position-uncertainty checks further
screen observations. The output explicitly does NOT authorize removing a
primitive: aggregate median depth does not identify that primitive, descriptor
matches can be wrong, and these old tracks may have participated in fitting.
The two new selection tests and nine foreground-bound tests passed11/11.
Full track audit launched onGPU1, child31781, `audit_v195_rigid_descriptor_depth`.

v163 retained iteration3000 was extracted losslessly:148 storages,86.54s,
render state1466709721bytes. Its51-view native evaluation is running onGPU1.
Training child31453 remained live in finalization when the extraction finished;
checkpoint availability alone was not reported as successful process exit.
GPU0 remains reserved for the user and unused by these experiments.

v163 subsequently completed with exit0. Its retained3000 canonical48 results
are tree15.462016/interior15.892343/boundary13.596139/rigid15.540191/
hard13.765401. The tree gain does not offset the rigid regression versus the
better optical-supervised branch; this arm is not accepted.

v194 completed187 views:116050 deficits, required alpha sum32971.73279 versus
native13977.50119 (means0.28412 versus0.12044). v195 completed194 participating
cameras/260 eligible tracks, with1111 usable rigid observations,5 shallower
rigid discrepancies and ZERO usable canopy observations. Each of those five
belongs to a different track; this does not establish a multi-view contradictory
tree-region primitive. No rigid correction is justified from this archive yet.
v199 therefore reuses the separately computed three-edge descriptor cycles,
rechecks actual native reprojection in all three cameras and estimates a local
one-pixel-noise information covariance, then audits native rigid depth. No new
matching or GPU0 use. This is still a read-only contradiction screen.

Matched v196/v197 static joint diagnostics use v1782000, the same304 training
cameras/order,48 excluded views,800 steps and all six rigid parameters frozen.
Both read the same fully covered v194 cache; the ONLY objective difference is
bound weight0 versus0.2. Cache checks cover checkpoint/masks/RGB/rigid hashes,
actual eligible camera coverage and pixel shapes. A one-sided log alpha deficit
acts on native visible leaf alpha; no renderer masks, geometry evidence relabeling
or opacity-ground-truth claim. Initial48 metrics match exactly. Supervisors:
v196441/child442 onGPU1;v197445/child446 onGPU2. Actual training updates started.

v198 mainline repair comparison launched onGPU1 after v163 exited, supervisor
1614/child1618. It resumes v178's immutable full2000 checkpoint to3000, preserving
the training objective/schedule/optimizer contract, with the existing explicit
trainer-repair resume mechanism for spent-witness and exact grouped-solver fixes.
It is not a fresh30k run and has no new bound loss or rigid geometry correction.
Launch sources for v194–v198 are archived separately; v199 source is later.

Project tests:1259 passed/15warnings before the later cycle adapter test.
Unscoped repository-wide collection additionally found one unrelated optional
tetra-triangulation extension import failure (`tetranerf.cpp` not compiled),
with1263 other tests passing. This is not claimed as a canopy renderer failure
and no unrelated dependency installation or source mutation was performed.

## v196–v203 results: no accepted correction yet

v196 and v197 completed800, both frozen-parameter audits passed. Native48:

| arm | tree | interior | boundary | rigid | hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| v196 control |15.405412|15.878590|13.415535|16.645339|14.760002|
| v197 bound0.2 |15.427251|15.897638|13.446860|16.641869|14.756810|

Incremental tree gain0.021839 (46/48 views positive), boundary0.031325,
rigid−0.003469. Visually660 still has conspicuous leakage/bright streaks.
This is not a substantive solution and was not promoted to production.
The paired report CLI now supports explicitly selecting static-only modes;
its previous default expected a temporal arm and failed for this static pair.
`v196_v197_paired_view_report_800.json` records the actual paired deltas.

v199 independently closed cycles produced58 usable canopy observations, all
with the rigid background BEHIND the triangulated canopy (686–688), not proof
of a shallow incorrect proxy.553 rigid observations had only one shallower
discrepancy. v200 separately examined TWO-view descriptor hypotheses without
claiming cycle verification:61 tracks had shallower rigid depth at both ends.
Some661/662 candidates had triangulated depth~6.4 versus rigid~5–5.7.
This motivated NEW matching to training view664, not held-out663.

v201 new661–664 and662–664 descriptor edges ran only onGPU1 under0.25 allocator
fraction and completed normally. Their661/662/664 closed cycles had29 canopy
tracks, median third-view reprojection28.63px, and ZERO all-three geometric
candidates. Rigid controls:98 tracks, median third reprojection0.726px,69
geometric candidates. Thus two-view tree depths are not safe correction
authority; repeated texture/motion remain ambiguous. No wall was moved or
deleted from those hypotheses. `--pairs` and `--triples` flags make follow-up
neighborhoods explicit and continue to exclude validation cameras.

v202 screened original MAtCha (NOT MoGe-adaptive chart) and MoGe agreement on
known-rigid interiors across available chart cameras. Both sources must agree
within10%, with a full7x7 valid window and a conservative20% depth interval.
Surfel center plus SIX plane standard deviations must lie before the measured
near depth. Any consistent geometry support protects a row. Result:54541 rows
had consistent support; no rows passed the conservative free-space screen.
This screen does not test every off-center surfel footprint or establish
individual native responsibility; its negative result is not proof that all
rigid proxies are correct. New plane-extent and track/cycle tests passed5/5.

v203 read-only attribution removed globally selected canopy-dominant rows
(NOT a per-pixel mask, NOT an exported correction). Selection used v193's1439
non-validation visibility observations:canopy mass>=10,>=3 canopy views;
rigid mass share<1% selects32 rows and<5% selects236. Counterfactual48 metrics:
32 rows:tree15.328198/rigid16.607535/hard14.635235;
236 rows:tree15.340980/rigid16.604948/hard14.624383;
baseline15.324917/16.608309/14.638506. Still a small tradeoff, not the central
fix. No parameters changed, native baseline equivalence checked in all48 views.

Also rechecked the suspected cross-sequence negative-supervision leak:
the mainline ALREADY uses a surface-disabled canonical footprint for unknown
protection and masks deletion factors; isolated detail cameras are restricted
to the canonical sequence. Do not report this as a newly found/fixed bug.
Intra-sequence tree disagreement remains distinct from this already-fixed case.

v198 is actually training (observed2175), not merely a launched PID. v178 was
observed2675. Both remain onGPUs1/2. Extracting v178's rolling2500 state now
preserves an earlier matched render comparison before it is replaced by3000;
the extractor checks iteration and source immutability. No30k result is claimed.

## v204–v207 continuation and allocator bottleneck

The deformation diagnostic now optionally anchors temporal code displacement
to an ACTUAL training time. `reference_index` is persisted in the field state;
at that time temporal and static geometry are exactly equal. Legacy captures
without this field remain mean-centered and cannot be silently reinterpreted
as a physical snapshot. Tests cover exact equality, saved-state restoration
and rejection of an unobserved reference. Old diagnostic results remain labeled
shared-reference shape, not retrospectively called a physical timestamp.

Matched v204/v205 use v1782000, physical reference frame104(view661), same304
training cameras,3200 steps (~10.5 visits per camera), eval every800, static
versus rank8 temporal center field. Both use radius1 scene unit, spacing0.5,
reference regularization radius1/spacing0.5 and the same shared leaf optics,
native rigid-alpha weight3, all rigid/sky/covariance parameters frozen. These
are a larger-capacity/weaker-prior and longer-duration pair, not a one-variable
comparison against v186. No mixed bound loss. GPU1 child10273;GPU2 child10277.
Sources archived in `v199_v205_launch_sources.tar.gz`. No result claimed yet.

Actual v1982200 birth event confirms the spent-witness repair is exercised:
depth-consensus rejected cells15256(old v178)→12050, eligible cells751183→754393;
902 births correctly mark spent-witness aggregate color as unknown. Both still
append EXACTLY2048 due the allocator cap. Verification debt was0.001305, and
retained additive mass0.13377 versus budget0.69149, so this event is not limited
by the configured debt or additive-mass allowance. Counts alone are not quality.

Implemented an EXPLICIT future-trajectory allocator ablation resume flag.
It may change only `maximum_births_per_topology_event`, with every evidence,
first-hit, optical and global-capacity field unchanged. Ordinary resume rejects
2048→8192; explicit ablation accepts it. Synthetic tests reject changes to
camera requirements/initial opacity/noninteger or nonpositive caps; the same
check passed on real v1782000 metadata.340 trainer/ablation tests passed.

v206 is QUEUED, NOT TRAINING: queue supervisor12727/child12728 waits for v178
to exit successfully and GPU2 to have at least12500MiB free. It then launches
`formal_v206_birth8192_resume2k_to3k` from the SAME immutable full v1782000
checkpoint as v198, changing only this event cap under the explicit ablation.
It never stacks two full trainers onto one GPU. Waiting is bounded7200s,
predecessor failure prevents launch, and any trainer/outdoor source change
while queued prevents launch pending review. **Do not edit those source files
without accounting for this intentional queue invalidation.** Tests for the
queue passed2/2. Sources archived in `v206_budget_ablation_sources.tar.gz`.

v178 rolling2500 extraction completed85.75s/148 storages, render state1453952665
bytes. Native48:tree15.288139/interior15.731145/boundary13.439167/
rigid16.630839/hard14.660416. Tree is lower than2000, not a monotonic improvement.

v207 expands ONLY the read-only global-surfel attribution range. Rigid mass
share<25% selected5206 rows:tree15.360170/rigid16.567192/hard14.505374;
share<50% selected16577:tree15.318105/rigid16.460813/hard14.235597.
Broader suppression trades away rigid quality and still does not solve canopy.
No source parameters or production render gates changed. Do not promote it.

Latest project suite1271 passed/15 warnings (before the v207 CLI-only change).
Rechecked deployment geometry handling: checkpoint loading ALREADY applies
baked live-atlas xyz/scales/rotation before these audits. The free-space screen
did not inadvertently test stale raw optimizer geometry. No new bug there.

## v208 relative rigid preservation diagnostic

Added an optional diagnostic-only objective to `calibrate_canopy_center_field`:
cache the initial model's raw per-pixel RGB error and native leaf alpha for all
304 TRAINING views, then penalize only increases on known-rigid pixels.
Existing absolute mode is unchanged. This is a refinement preservation
constraint, not proof that initial spill is geometrically correct. ReLU has
zero boundary derivative so exact initial predictions do not receive forced
retirement gradients. Tests confirm zero initial gradient, penalty for added
spill only, and detached references. The first training camera has no canopy,
so a legitimate zero field gradient there is allowed in this mode; the run must
still exercise a nonzero field gradient before completion. Frozen parameters
are checked on every update and by final hashes.

v208 compares against v196 with the same source2000, static0.25-radius field,
spacing0.5,800 steps, shared optics and otherwise identical settings (bound
weight remains0). Only absolute rigid optimization/retirement changes to
initial-model preservation. GPU1 allocator cap0.20 limits the additional
diagnostic independently of both running trainers. Supervisor18712/child18713;
it first builds training-only reference rasters, then trains/evaluates. No new
mode is enabled in the production trainer. This script-only edit does not
invalidate the queued mainline allocator source check.

At v205800, the larger anchored temporal arm is NOT yet an improvement:
static tree15.234404/rigid16.654118; temporal tree15.101559/rigid16.681026,
with temporal boundary12.811828. The declared3200-step test remains running;
do not call a larger displacement alone a reconstruction improvement.

## v209 depth-scale provenance recheck and v210 diagnostic projection repair

The seed's canopy depths have an additional per-camera rigid-anchor multiplier:
661=1.701731,662=1.764939,664=1.767109. Thus v189/v191 depth conflicts concern
CALIBRATED canopy MoGe, not just the globally converted network prediction.
This is not automatically a bug: the v130 provenance already documents large,
coherent per-view metric-scale errors and improved agreement after calibration.
The initial v209 reader rejected an older audit lacking `evidence_kind`; v209b
accepts that legacy schema only after verifying the actual three-edge track
file hashes. It rejects two-view hypotheses. CPU-only v209b compares saved
descriptor depths against global MoGe, calibrated MoGe and native rigid depth;
no parameters are edited and no proposed depth is promoted to ground truth.

A separate real DIAGNOSTIC implementation confound was found in v208: step1
has exactly zero loss/gradient yet 4687 opacity rows retire. The shared-optics
helper imposed absolute alpha<=.35 after every Adam step, including rows already
above this cap in the initial checkpoint. New `initial_upper` projection instead
uses max(initial_logit, logit(.35)) as the per-row ceiling. It is NOT a lower
bound, does not restore legitimate retirement, and never changes ineligible rows.
The historical absolute mode is explicit for reproducibility. Tests cover zero
gradient identity, accepted retirement, forbidden growth, and ineligible identity.
This does not change production training or the native renderer. Existing
v204/v205/v208 processes keep their imported historical implementation.

v210 on physicalGPU2 repeats v208's800-step relative-rigid-preservation control,
same source2000 and settings, changing only this projection policy. Allocator
cap0.20; supervisor22917/child22918. It first caches initial rigid references.
The v209/v210 archive was collected immediately AFTER launch (not a prelaunch
archive). No visual improvement is claimed from this repair before evaluation.

The unlaunched v206 queue was deliberately stopped (child12728 SIGTERM) and
recreated as `queue_v206b_birth8192_supervisor` (22513/22514), because its broad
source fingerprint includes the changed diagnostic helper. No running CUDA
trainer was stopped. It still waits for successful v178 completion and GPU2
memory headroom, uses the same immutable2000 checkpoint, and keeps its exact
8192-versus2048 experimental question. This was not a spontaneous crash.

v204800 static tree15.353668/rigid16.660691/hard14.794770; its boundary13.269797
is worse than the source13.426374. Together with v205800, larger motion has not
yet established a useful reconstruction change. Both continue the declared3200.

v209b completed13 camera comparisons. At problematic661,44 independent closed
rigid track observations have median absolute log-depth error0.54414 for global
MoGe,0.01249 after the1.70173 calibration, and0.01323 for native rigid depth.
At662 the corresponding errors are0.57664/0.00849/0.00739. This supports keeping
the per-camera gauge correction; dividing canopy depths by1.7 is not a verified
repair. No closed canopy tracks exist in these particular views. The58 canopy
observations in686–688 have unchanged scale1.0 and MoGe errors0.056–0.129, with
rigid much farther behind; they do not explain the shallow-proxy conflict in661.
Project tests after these changes:1275 passed,15 warnings. The first focused
test invocation omitted the environment library path and failed importing
libstdc++; rerunning with the existing conda LD_LIBRARY_PATH passed. No runtime
dependency was installed or changed, and no formal trainer failed from it.

## Original observed-track evidence, v211/v212 (not SfM coverage supervision)

The prepared runtime COLMAP images explicitly contain zero feature observations,
but the original `/mnt/pool/sqy/Cambridge_stdloc/StMarysChurch/reconstruction.nvm`
retains419837 sparse points and their actual measured image features. A new
CPU-only v211 reader retains tracks observed ENTIRELY in permitted canonical
training cameras, excluding every48 validation view and every other sequence.
It checks original/current camera centers agree within1e-4, current native
pinhole reprojection<1px at EVERY observation, distinct cameras>=3, all known
semantic pixels and p10 triangulation angle>=1degree. Unsupported raw distortion
is rejected by native reprojection, not refitted to manufacture correspondence.

Counts:6854 all-allowed tracks,1291 native-reprojection-valid,1281 angular-valid,
103 tracks with at least2 canopy observations,324 total observations across64
views. This includes tracks198739/200242/204017 in664/665/666 at depth7.3–7.6.
These are historical SfM observations, possibly used in rigid initialization;
they are NOT independent camera poses or proven descriptor cycles. The adapter
keeps cycle rank0 and explicitly uses the original-observation contract.
No SfM coverage priors, DAV2 supervision or new training losses were introduced.

v212 is queued on GPU1 AFTER successful v208 completion and>=6500MiB free,
to compare these observations with native rigid-only median depth from exact
v1782000. Queue supervisor26536/child26537; it is not yet a running GPU audit.
Source fingerprints are checked again at launch. No rigid parameter has been
corrected or approved for retention on the basis of track counts alone.
The adapter's test explicitly verifies three original observations do not get
relabelled as descriptor-cycle evidence (focused suite5 passed).

v208400 tree15.452432/interior15.922270/boundary13.483378/rigid16.615740/
hard14.703225. Source-upper v210 step1 verifies exactly zero growth AND
retirement for zero loss/gradient, versus4687 retired rows before the fix.
Its final image/metric effect still requires paired evaluation.

## v212–v214 separate the depth-prior mismatch from rigid obstruction

v212 completed64 views:191 valid canopy observations,4 shallow-rigid and148
deeper-rigid discrepancies. Only one track (original NVM320497, adapter index82)
is shallow in multiple views:678/679/681, actual track depths9.304/9.353/9.555
versus native rigid8.640/8.578/8.831, margins0.465/0.468/0.478. Its responsible
primitives still need identifying; a median discrepancy is not permission to
remove building geometry. At new nearby664–667 tree tracks, native rigid is
much farther BEHIND (median gaps−11.5 to−31.0), not in front of the tracked point.

v213 verifies all four USED mmap arrays (depth, valid, refinement std/delta)
against their content hashes and source index; unused normals are not read.
At318 valid actual canopy observations, only41 lie in the production thin-hit
interval,213 are in front and64 behind.53 tracks are in front of the thin prior
in at least3 views;72 in at least2.19 points precede the conservative negative
bound BEFORE any rigid clipping. Do not claim those19 received actual negative
gradients: the live rigid clipping/state policy must still be checked.
The median absolute log-depth discrepancy is0.09729. These sparse features can
include gaps/background and are not dense leaf-depth truth.

Direct raw-archive checks of tracks198739/200242/204017 in664 give actual
depth7.29–7.47 versus calibrated MoGe8.48–8.98. At665,6 observed canopy points
give7.19–7.61 versus9.11–9.61;666 gives7.11–7.54 versus9.33–9.71;667 gives
7.27–7.69 versus10.41–10.88. The positive AND birth interval has at most0.40
metric half-width, so it cannot contain these observed positions. This is a
serious observation-model limitation: refinement stability does not establish
monocular depth accuracy. Widening optical material to cover the error is not
the proposed fix. Rigid anchor calibration itself remains well supported.

v214 read-only regional source-bearing association uses the same immutable
v1782000 checkpoint, correct calibrated camera IDs, actual recorded source UV,
and persistent eligible leaves only. Within2 native pixels,45 of53 three-view
front-of-prior tracks associate with152 distinct leaf rows. Per-track median
required displacement has median1.34894 and maximum2.58254 scene units. These
are NEIGHBORHOOD associations, not exact SIFT-to-leaf identities; no rows moved.

v208 and corrected-projection v210 both finished800. v210 tree15.475946/
interior15.946901/boundary13.504037/rigid16.616646/hard14.710390. Difference
from v208 is only8e-6 tree PSNR: the zero-gradient projection bug is fixed but
had no meaningful visual impact. Do not advertise it as the canopy solution.

## v215/v216 static, observed-region-guided controlled experiment

Added script-only optional regional centroid guidance. Both arms load the same
observation/association hash chain, same source2000, same304 training views and
48 exclusions, shared optics, relative rigid preservation and source-upper
opacity policy. The experiment adds a0.02-weight Huber centroid term with0.1
scene-unit robust scale; this scale is NOT claimed to be calibrated noise.
Each observed source camera contributes a separate centroid. Within-region
offsets are not individually collapsed to a feature point. Sparse field
evaluation matches the full field exactly in value and gradient in unit tests.
No guide reaches rigid/sky/appearance, covariance, ineligible optics or cameras.

Both fields have rank0,spacing0.5,radius5,physical regularization reference5/0.5,
800 steps/evaluate400. The existing tanh cube allows each component up to
radius/sqrt(3); radius5 allows multi-unit corrections suggested by the new
observations. The control gets EXACTLY the same larger capacity. This remains
an exploratory static-field test, not a replacement production initializer or
ground-truth leaf correspondence model. Native48 RGB evaluation decides whether
the observed geometry prior helps or merely distorts the crown.

v215 `diagnostic_v215_observed_wide_static_control800` onGPU1, supervisor32629/
child32630,weight0; v216 `diagnostic_v216_observed_wide_static_guided800` onGPU2,
supervisor354/child357,weight0.02 after successful v210 completion. Both allocator
caps0.25. The source archive was collected BEFORE launch. Focused guide/field/
rigid-preservation tests12 passed. Mainline v178 reached its3000 topology event
at17:42 but retained checkpoint/final successful exit still require checking;
v206 remains predecessor/memory-gated, not a completed or running result.

Both v215/v216 validated93 source-camera regional centroids. v216 step1 has
only geometry-guide loss0.12336, field gradientL1=0.56224 and maximum actual
shift0.039999; color/opacity gradients and changed-opacity rows are all0.
At200 its guide loss falls6.16794→0.06920 and maximum displacement2.85710;
this establishes fit to the guidance, NOT reconstruction generalization.
Full project suite after the guidance changes:1278 passed,15 warnings.

v198 rolling2500 was safely extracted (148 storages,83.51s,1,453,959,001B,
source3,185,756,934B unchanged) into `render_only_002500.pth`. Its51-view native
evaluation is queued on GPU1 after successful v204 completion and6500MiB free
(`queue_eval_v198_2500_supervisor`,4165/4166). v178 retained3000 exists at17:53;
render-only extraction supervisor4153/child4154 is running. Successful final
v178 exit remains distinct from the existence of that retained checkpoint.

## September 9, 18:30: completed results and local depth attribution

v178 completed normally (supervisor returncode0). Retained3000 extracted without
changing source: full3,238,470,598B, render-only1,466,034,393B,148 storages.
Native canonical48 at3000: tree15.35748722, interior15.79477264,
boundary13.50608792, rigid16.64744407, hard14.66806777. Still not a substantive
solution. v198 at2500: tree15.28859438 versus old2500 15.28813918, only+0.0004552.

v204 static3200: tree15.36690833, boundary13.24036117. v205 temporal3200:
STATIC tree15.24484084, boundary13.27614748; temporal tree15.49340747,
boundary13.10194393. Temporal quality is not static-map acceptance.
v215/v216 both800 completed with frozen-parameter audits. Wide static control:
tree15.32954446, interior15.88204980, boundary13.12553304,
rigid16.64365069, hard14.75984959. Guided counterpart: tree15.35146077,
interior15.91269288, boundary13.08196352, rigid16.64570804, hard14.75060064.
Only+0.02192 tree versus control; boundary worse. Neither field is adopted.

Deployment-only export bug fixed for future launches: final inference state
previously serialized the large training candidate pool and surface Adam state,
although the inference loader uses neither. New deployment helper omits these,
preserves native surface tuple layout and all learned parameters, and marks
the output render_state_only_nonresumable. FULL training checkpoints continue
to retain optimizers and candidate pool unchanged. Tests explicitly distinguish
the two save sites. Full suite1282 passed before the later handshake tests.

v206c queue dispatched the actual v206 supervisor8891/child8992 onGPU2 at18:13.
Its OUTER queue nevertheless reported failure: 15-second acknowledgement timed
out during32 seconds of startup work. Actual training is alive and has a fresh
trace; this was not CUDA OOM or a stopped trainer. No relaunch performed.
Supervisor handshake now distinguishes launch_unconfirmed (timeout with live
pipe; may still launch; check heartbeat, do not relaunch) from true pipe EOF or
explicit failure. Dispatch success is never called running without heartbeat.
Two regression tests added; supervisor suite28 passed. Future explicit launches
also use120-second acknowledgement timeout, with tool yielding asynchronously.

v217/v218/v219 image-only audits compare identical photometric rules at actual
observed track neighborhoods, not at a model-rendered synthetic correspondence.
Source665,target664/666, track198739: narrow observed-centered sweep54 joint
matches versus MoGe9; track200242 72 versus4. Wider MoGe sweep returns53/80
matches with depths7.60/7.65, not priors9.16/9.60. Track204017 has0 joint matches
despite many individually good pairs; it is NOT promoted. Source666 corroborates
some other neighborhoods. These sparse overlapping patch counts are not dense
coverage or independent material counts.

At the only three-view shallow rigid conflict, track320497, source678 and targets
679/681: observed-centered narrow sweep15 matches, median depth9.30427 and
minimum-target NCC0.95927; MoGe-centered narrow sweep2 matches at5.81653.
Wide MoGe search recovers depth9.22919. Bias thus changes sign spatially; no
global scale replacement is justified.

v220 local native DC-gradient attribution completed, exact surface restored.
At views681/679/678, wholly shallower native responsibility is0.6695/0.9661/
0.8548. Forty surfels contribute conservatively wholly in front in>=2 views.
They ALL have some known-rigid view support; they are candidates for causal
testing, NOT authorized deletions. v222 runs native48 counterfactual alpha
factors0.5/0 on these exact40 global rows, with no per-pixel deployment gate and
no model export. Known building quality must be examined before any retained
correction, and a visible feature behind a real foreground branch is possible.

v221 is a dense IMAGE-ONLY sweep on source665 with training targets664/666,
physicalGPU1, allocator0.15, stride2. Search radii0.15/0.4 share the SAME0.005
log-depth bin spacing, reject boundary peaks and retain only common unambiguous
two-target peaks. No NVM depth is used in this dense search; no interpolation,
opacity, or model changes. Outputs remain unverified depth hypotheses pending
independent reverse-cycle and native visibility checks. The historical thin
MoGe interval is a demonstrable location bottleneck in some neighborhoods,
but neither denser geometry nor final leak improvement has yet been established.

## v221–v232: distinguish location evidence from density and rigid edits

v221 finished:35,090 source665 training canopy samples,162 shared narrow-range
matches versus525 wide-range matches. Wide median selected/prior depth0.77492.
Both ranges use0.005 log-depth spacing, not different coarse agreement bins.
The legacy matcher retained only the adjacent depth bin, so finer grids shrink
the positional acceptance neighborhood. Added OPTIONAL score-supported peak
width inside the unchanged physical uniqueness neighborhood. Legacy production
callers are unchanged; no material layer is widened. Synthetic sampling-density
and ambiguity tests pass. v223 controlled rerun:314/848 matches, wide ratio
0.78270. Still sparse, no reverse-cycle/native verification, not dense truth.

v224 moved only existing persistent source-bearing neighbors of these image
hypotheses, preserving SH and opacity; a separate arm also preserves source
screen footprint by explicitly rescaling covariance with depth. Only19 rows
qualified. Native48 tree15.3250674 center-only /15.3250155 footprint-preserved
versus15.3249166 source. All leaf and rigid tensors restored exactly. This is
NOT a useful map repair and produces no deployable model.

v222 finished40-row rigid counterfactual: half alpha tree15.32541476 /
rigid16.60899282 /hard14.64039395; zero alpha15.32570503 /16.60970245 /
14.64251226. Gains remain negligible. No deletion retained, and whole-scene
known-building acceptance was not inferred merely from mean48 improvement.

The original NVM screen discarded a whole track whenever any observation lay
outside the training cohort. Added an explicit training-subset refit: discard
those observations first, then triangulate solely from the remaining>=3 actual
training pixels, with no original NVM xyz initializer. Exact rotated-camera
projection and zero-parallax tests pass. Reprojection<1 native pixel, parallax
p10>=1degree and known observation masks are retained. Historical track grouping
and camera poses still originate from existing SfM; this is NOT a newly blind
matching benchmark or a resurrection of projected SfM coverage supervision.

v225 CPU refit:419837 input tracks,75109 training-only three-view candidates,
29138 pinhole-valid,28371 angular-valid,2311 canopy-related tracks,8634 total
observations,130 training views. Canopy semantics may include visible gaps or
branches/background; these tracks are not automatically optical leaf matter.
v226:8421 valid canopy observations,1108 inside current thin prior,3368 before,
3945 after;709 tracks are before in>=3 views.225 observations precede the
unclipped negative bound; actual negative gradient is still not established.
v227 native depth screen:11 tracks have shallower rigid conflicts in>=3 views,
24 in>=2. Source662's newly available point418562 has real depth5.51846 versus
MoGe7.74417 but its native opaque-rigid test is invalid; it does NOT prove a
wrong building there. This distinction remains important.

v228 source-bearing association:647 of709 front-of-prior tracks associate with
2867 distinct existing leaves. Median per-track required motion1.31436, p90
2.13827, maximum6.07909. v229 guided /v230 zero-guide matched800-step tests
started onGPU1/GPU2 with the same old radius5,spacing0.5,weight0.02/0,source2000,
shared optics, source-upper projection and relative initial-rigid preservation.
Both validate1555 source-region centroids. New control explicitly checks that
the larger zero-weight evidence does not change the experiment. Both exclude48
validation cameras from updates. Archivev229_v230_prelaunch_sources.tar.gz was
captured BEFORE launch. Guide membership validation now uses an equivalent
CPU boolean active-row lookup, avoiding repeated million-row membership scans.

v231 actual patch comparison at the11 rigid-conflicting tracks favors the
observed depth strongly for several tracks; native-depth median NCC is often
negative. Four tracks (319149,391297,392555,320497) have every ordered camera
pair NCC>=0.8 and at least0.1 better than the native-depth hypothesis. v232
sequential native contribution attribution only uses these mutually corroborated
observation subsets and preserves all source tensors. It remains a candidate
audit, not a retained opacity/geometry edit; known building support is protected.

## September9 19:12 follow-up

v229/v230 completed800; final frozen-parameter checks passed. Guided tree
15.37983960/interior15.93646268/boundary13.08475182/rigid16.62742472/
hard14.65098377 versus zero-guide control15.33726956/15.89052097/
13.12649091/16.64785107/14.77543300. Tree+0.04257 is modest and costs hard
quality0.12445; no field adopted. The wider position evidence is useful for
diagnosis but is not a demonstrated reconstruction fix.

v232 attributed88 union surfels. v233 native48 suppression: half-alpha tree
15.32951504/rigid16.60932267/hard14.64192822, zero-alpha15.33362613/
16.60896466/14.64158156. Zero-alpha hard view674 regresses0.10046; half-alpha
regresses0.02827 there. Mean improvement is not enough to authorize the group.
No surface opacity or geometry correction has been retained.

Added an explicit DC learning-rate ablation to the diagnostic only: scale the
ACTUAL Adam DC parameter delta, not its gradient (which Adam normalizes).
Higher SH, opacity, geometry rates and authority remain unchanged. A separate
Adam-parameter-group reference test verifies equivalent DC updates, bitwise
unchanged higher SH and ineligible rows. This tests color adaptation versus
fast opacity retirement after relocation; it is a hypothesis, not a confirmed
production bug. v234 fast-DC guided800 startedGPU2 after successful v230; its
DC multiplier10 gives initialDC LR0.025, higherSH stays0.0025, opacity0.02.
v235 zero-guide fast-DC counterpart is planned after v1983000 evaluation.
Sources archived BEFORE launch in v234_v235_prelaunch_sources.tar.gz.

v198 reached final3000 topology and wrote retained checkpoint3,262,316,294B.
Its CPU finalization is still distinct from successful process exit. Extraction
to render_only_003000.pth started (supervisor27061/child27062), with native51
evaluation queued onGPU1 after extraction success and6500MiB headroom.

External nonblocking Python stack sampling was denied by container SYS_PTRACE
policy. No attach, stop signal, capability change or training restart occurred.
py-spy0.4.2 was installed ONLY in /tmp/g4splat-profiler-apFSvP, not the project
environment. Exact CPU hot stack remains unmeasured. Future production drains
now emit stage/candidate/selection progress to supervised stdout so CPU work is
visible without attaching. Also fixed a definite zero-capacity inefficiency:
drain previously solved/sorted all candidates before breaking at row0. At zero
capacity it now returns unchanged state immediately, marking eligibility NOT
evaluated (zero placeholder is not evidence that no candidate exists). Positive
capacity selection is unchanged; progress-enabled equivalence and zero-capacity
no-scan/no-consumption tests pass.43 focused birth/export tests pass. Running
v198/v206 already imported old code and are not claimed to have this new logging
or fast path. Their current positive-capacity events are not explained by the
zero-capacity bug. Full suite before these last changes:1289 passed15 warnings.

## September9 19:32 follow-up

v198 completed with return code0. Render extraction completed83.214s,
1,466,123,929 bytes. Native canonical48 at3000: tree15.3580974142,
interior15.7953658303, boundary13.5060219566, rigid16.6479242047,
hard14.6687533657. Tree gain versus v1783000 is only0.0006102dB: the
allocator correctness fix is not a substantive canopy reconstruction solution.

v235 fast-DC control completed800: tree15.35801017, interior15.92163537,
boundary13.11795473, rigid16.64194401, hard14.75433560. v234 guided400:
tree15.25525633, boundary12.92261646, hard14.60763727. Await paired800;
these intermediate results do not establish a useful improvement.

v236 local-UV audit initially failed its shape assertion: the new diagnostic
mistakenly expected three output channels for three input audit fields. Native
outputs contain total contribution first, then the supplied fields (four
channels). Corrected ONLY the new diagnostic's shape and field offsets, not
the renderer. Fresh v236b supervisor31819/child31820 started GPU1 at19:30;
1439 training views exclude all48 validation views. Each view also checks
atlas-node contribution sums against native per-surface responsibility. No
surface tensors changed or model exported. This tests a narrowly scoped local
counterfactual after whole-row suppression damaged validated building view674.

Full suite after DC learning-rate and drain progress changes:1292 passed,
15 warnings. This predates the new local-UV diagnostic and does not claim
coverage of its initial channel-contract error.

## September9 local-support and reciprocal-depth follow-up

v236b completed all1439 views. Among22,528 atlas nodes on88 independently
attributed surfaces,87 had exact-zero known-rigid contribution, four had
multi-view negative evidence, and their intersection was EMPTY. The native
atlas test now explicitly verifies total-channel semantics and atlas-to-row
conservation; targeted CUDA test passes. v237 temporarily suppressed only the
four negative nodes on three surfaces, including supported nodes. Their total
known-rigid contribution was0.00389261. Native48 tree mean was EXACTLY unchanged
15.324916621; rigid/hard changes were floating-point scale (worst hard
-9.54e-7dB). No useful correction or model export. This does not justify broad
rigid deletion and does not establish all rigid geometry as correct.

v234 fast-DC guided800 completed: tree15.41716609/interior15.98911287/
boundary13.09582311/rigid16.62245615/hard14.64302814. Versus v235 fast-DC
control: tree+0.05916 but hard-0.11131 and boundary-0.02213. Not adopted.
v206 main2675 is still active. Its2500 native48 tree15.29225167,
interior15.73545452/boundary13.44244623/rigid16.62871291/hard14.65755483;
tree+0.00365728 versus same-correctness v1982500, not a substantial gain.
Extracted render state1,460,413,465B in74.94s, confirmed iteration2500 and
unchanged source during extraction. Initial extraction launch used an incorrect
CLI flag and exited before reading; corrected fresh extraction completed.

v238/239 independently swept sources664/666 using the same three training
images as v223 (no validation images). Wide search produced875/871 hypotheses
versus848 at665. New reciprocal checker demands all six directed matches,
nearest-grid matches chosen without depth cherry-picking,2px and0.02 log-depth
tolerance:91 three-way cycles. Broader2.5px/0.05 tolerance gives102, diagnostic
only. Camera metadata, images, masks and raw depth hashes agree. Plot inspected:
many candidates lie on crown texture and are nearer than the canopy MoGe
profile, but gaps/branches remain possible and coverage is sparse. No leaf
opacity or rigid correction is authorized by these hypotheses alone.

v241 eight-step smoke startedGPU1 for a new diagnostic native volume-depth
guide from these91 cycles. No nearest-owner assignment or interpolation is
used. Additional depth guidance blocks opacity and color gradients, only moves
the shared static tree position field; normal RGB optimization is unchanged.
Expected volume depth is a diagnostic moment, NOT proof of a unique leaf depth.
Guidance excludes unknown/near-zero-alpha pixels and allows0.02 log-depth
position tolerance. First active guide step audits nonzero geometry gradient
and exactly zero optical gradients; all rigid tensors stay frozen. Await smoke
before any800-step run. Four focused reciprocal/depth-loss unit tests passed.

v241 completed returncode0, frozen audit passed. At the first guide update all91
pixels contributed, geometry gradient L1=5.05108309, optical gradient EXACT0.
Read-only v243 reconstructed the saved8-step field and checked native depth:
mean absolute log depth at665/664/666 decreased slightly; all91 pixels remained
visible. This is a gradient/path smoke, not a quality result. v242 weight0.2
800-step run startedGPU1 (supervisor4352/child4353); v244 weight1.0 counterpart
startedGPU2 (4935/4936). Both compare to unchanged v230 zero-guide control;
regular RGB optics schedules, static field capacity and rigid protection match.
Sources archived before v242 launch in v242_dense_depth_prelaunch_sources.tar.gz.
The auxiliary post-hoc depth reader was added later and is not in that archive.
Current full suite1296 passed15 warnings, before that auxiliary reader.

At the strict91 cycles, observed/prior depth medians665=.767206,664=.843665,
666=.759572. Respectively87/83/86 of91 are nearer by at least10 percent. These
are localized, mutually consistent position hypotheses; they neither certify
every point as a leaf nor authorize scaling the entire canopy or all profiles.

## September9 prehit-authority repair and remaining bottleneck

v242 completed800: tree15.33490215/interior15.89312388/boundary13.10888904/
rigid16.64336212/hard14.75873234. Versus zero-guide v230 tree15.33726956,
this is NOT an improvement. v246 native training-depth reader at400 confirms
the position objective actually worked: mean absolute log depth665/664/666
was .22031/.27513/.23387, became .03584/.03366/.03587 with geometry alone,
and .03628/.03471/.03701 with learned optics. Yet canonical48 RGB is unchanged
relative to control. Sparse local geometry correction is insufficient to fill
the canopy. v244 strong1.0 guide400 tree15.16152108/hard14.71343478, also not
an improvement; final800 still pending at this note.

Confirmed a legacy DIAGNOSTIC mismatch: diagnose_canopy_thin_coverage and the
support-expansion branch of audit_pending_canopy_verification configured only
the global MoGe scale, not seed-bound per-camera canopy profiles. Both now
require matching initialization when using canopy depths. New shared helper
checks checkpoint initialization-manifest and foliage-seed SHA256 and applies
both scales, reading seed metadata without materializing all tensors. Native
v1782000 metadata check succeeds with304 profiles. New unit tests reject
mismatched global scale and missing profiles. This does NOT invalidate recent
v226/v240 analysis, which explicitly used the correct canopy profiles. The
separate raw-depth consistency tool has an explicit rigid-anchor calibration
comparison and was not blindly changed.

v245 initially failed BEFORE model load because the new probe script imported
the editable-installed CUDA module before binding this worktree's native root.
The implementation hash guard correctly rejected it. Fixed only import order;
no CUDA source/binary changed and no compatibility check was bypassed.
v245b used source-bound profiles, production rigid clipping and actual task
fields. Synthetic small volumes at the91 reciprocal hypotheses received
retirement gradients in72/17/76 cases at665/664/666.88/87/87 hypotheses remained
query-valid after rigid clipping. This is causal evidence of the negative
LOSS PATH, NOT a count of actual retired model rows or certified leaves.

Implemented production default --moge3-prehit-authority observed_rigid:
prehit negative authority requires explicit rigid semantics, no canopy mixture,
finite positive rigid depth and opaque rigid alpha>=.95. The original
--moge3-prehit-authority legacy_moge remains an explicit control. Positive hit
loss, birth authority and native clipping are unchanged. Kept the ORIGINAL
negative normalization denominator, so removing canopy terms does not amplify
remaining wall gradients. CPU gradients prove zero canopy negative, identical
retained wall gradients and identical hit gradients. v247 native probes confirm
72/17/76 legacy retirement cases become EXACT0 in the repaired domain.

The policy is recorded in the training contract. Missing historical domain key
means legacy_moge. Changing it across resume requires the explicit
--allow-moge3-prehit-authority-ablation-resume plus trainer repair; every other
optical contract field and other training contracts still compare exactly.
The helper file has its own implementation hash and is a Python-only compatible
repair, never a CUDA/render migration. Full suite1301 passed15 warnings.

v248c GPU1 supervisor11213/child11214 starts a paired2000->3000 run from the
same v1782000 checkpoint as v198, changing only the negative domain (and already
tested runtime progress/zero-capacity diagnostics). It is NOT a clean retrain.
The initial v248 command omitted the supervisor's separately stored external
resume path and was rejected by the parser before training; v248b supplied it
in the wrong interface and was rejected by the supervisor. v248c uses the
documented --initial-resume-checkpoint interface. Neither rejection was OOM
or a mid-training interruption. Prelaunch sources archived as
v248_observed_rigid_prehit_prelaunch_sources.tar.gz. v206 remains GPU2 on its
older already-imported code, unchanged by the new default.

Asked asynchronously whether low-initial-opacity multi-depth canopy candidate
experiments may be tested using only multi-view RGB, not promoting hypotheses
to ground truth or changing buildings. No such candidate experiment has begun
pending that answer. Additional read-only check: existing source high-SH norm
max2.887657<diagnostic cap4 for all1,239,507 rows; SH clipping is NOT the cause
of the observed retirement in these paired runs.

### v252: authorized static multi-depth RGB candidate pilot

User explicitly approved low-opacity multi-depth canopy candidates with frozen
buildings. Implemented isolated `canopy_candidate_diagnostic.py` and
`calibrate_canopy_candidates.py`; no production renderer/trainer changes.
Existing foliage, rigid geometry/covariance/opacity/SH, sky and appearance are
frozen. Hypotheses use only non-evaluation canonical training canopy rays,
seed-bound calibrated MoGe depth as a search center, never depth truth.
Each candidate starts at opacity .001, fixed isotropic angular sigma1.2px,
DC-only trainable color and opacity; only known-pixel multiview RGB loss.
There is no positive-hit loss, opacity floor, image-space canopy visibility
mask, fabricated camera verification or production checkpoint export. Native
joint depth sorting is retained. Candidate-off renders must match the source
within1e-6 on all48 views; immutable parameters are hashed before/after.
Planned matched pairs use three depths each: [.98,1,1.02] vs [.65,.8,1], same
training rays, point count, initial opacity and angular footprints.
v252 is an8-step causal smoke with32 rays/view on GPU2, not an efficacy run.
Three CPU unit tests pass; actual native gradients/updates still pending.

v244 strong depth-guide800 completed: tree15.3170731664, interior15.8713395596,
boundary13.0914980570, rigid16.6448969046, hard14.7550406059. It does not beat
the matched v230 control; do not adopt. v206 reached3000 and retained its full
checkpoint (3,282,295,238 bytes); final native48 evaluation pending. v248c has
actually reached2125 (not merely occupied GPU), with unchanged prelaunch
trainer SHA53685dd91afab946cc2017cde5ebf05fd411c49991748c4d0ccc594b095efa85.
Independent dense maps near held-out660: source659 has142 hypotheses, source661
166; source662 v251 launched on GPU1 to complete reciprocal checks. All three
sources are training-only. None of these results solves canopy see-through.

v252 completed8 steps, return0:17,952 candidates; native candidate-off equivalence
passed on48 views; all original parameters bitwise unchanged. First update
opacity gradient L1=9.27247e-6,1,529 changed rows; step8 had11,995 changed rows.
Candidate-enabled initial tree15.325451 vs disabled15.324917, only.000534dB.
This validates the causal plumbing, not reconstruction improvement.
CPU suite1284 passed/20 skipped (CUDA hidden); separate GPU2 native+candidate
tests22 passed. Additional paired-report test passed (four candidate tests total).

v254 narrow [.98,1,1.02] and v255 wide [.65,.8,1] launched800 steps onGPU1/2,
256 rays per eligible training view, three candidates per ray. All304 canonical
training views receive RGB supervision;187 have candidate seeds. Same DC Adam
LR.025, opacityLR.05; hold LR first half, decay tenfold second half. This schedule
differs from the8-step plumbing smoke, but is identical between paired800 runs.
Source archive `v254_v255_candidate_prelaunch_sources.tar.gz`.
v253 independent reciprocal659/661/662 check retains only4 strict triplets
(5 under looser diagnostic tolerance); this does not establish wind as cause.
v206 native3000 canonical48 tree15.3723467191, interior15.8098717332,
boundary13.5187574824, rigid16.6446048419, hard14.6633433302: versus v198,
tree+.014249305, hard-.005410035. Not a substantive canopy fix.
New `audit_canopy_candidate_contribution.py` measures actual native candidate
T_before*alpha responsibility and surface-alpha changes, distinguishing canopy
contribution from rigid/sky spill. Evaluation support NEVER becomes training
verification; v256 is its read-only v252 smoke audit onGPU2.

### v254/v255 completed; improvement with unacceptable building regressions

Both800-step runs completed and preserved all source parameter fingerprints.
v255 wide800: tree15.5089566112, interior15.9421154857, boundary13.6316506068,
rigid16.6270707250, hard14.6831525962. Relative to the immutable source,
tree+.184039394, boundary+.205276688, hard+.044646144. Relative to v254 narrow,
tree+.086890817, boundary+.093363941. However44/48 trees improved and some
buildings materially regressed: hard909 -1.195127487,748 -.251064301,
743 -.197742462,738 -.175075531. These candidates are NOT adopted/exported.
Viewed660 at400/800: strong background-building see-through remains despite
tree+0.0852108 at800. Fixed geometry does not guarantee preserved building RGB.
v257 actual native responsibility atwide400:142,731 candidates,21,352 with
opacity>.01,1,951>.1; canopy mass37,112.2734,rigid23,349.0801,sky2,139.7581.
76,642 candidates contribute>.001 mass in≥2 evaluation views; this is
evaluation-only attribution, NOT verified training support or leaf truth.

Found an implementation defect specific to these NEW DC-only candidate
experiments: v255800 has142,692/428,193 negative color channels (59,315 rows
with≥1),10,506 channels>1; pre-clampRGB range[-.1583,2.0715]. Native clamp_min
then gives negative DC-only channels zero recovery gradient in EVERY view.
This does not prove the same global death for the existing view-varying SH.
Added candidate-only in-gamut DC projection; no source SH/opacity changes.
The first mathematical endpoint implementation failed the recovery-gradient
unit test due float32 roundoff slightly below0. Fixed with one representable
step inward using nextafter. The test now demonstrates actual recovery.

New independent helper `canopy_candidate_ray_refinement.py` supports bounded
log-depth adjustment along each seed's original camera ray, preserving its
source angular footprint; it creates no verified evidence. An optional separate
known-building RGB-reference loss preserves the frozen source render, without
treating its depth as ground truth. v258 combined8-step plumbing smoke completed:
11,994 actual depth codes updated, maxabs.08297, geometrygradient nonzero;
all source parameters unchanged. It preceded the new color-domain repair and
is NOT the matched efficacy baseline. Candidate helper and trainer source
changes occurred only after v254/v255 had recorded their original manifests;
their imported numerical code remained unchanged and old archives are retained.

v259 unit_rgb baseline800(GPU2),v260 unit_rgb+ray_log_radius.15(GPU2),v261
unit_rgb+rigid_RGB_preservation_weight3(GPU1) launched from the SAME2000 source,
same142,731 wide-depth candidates and800-step schedule. No combined efficacy
change: depth and building guard each have a separate same-code corrected-color
baseline. Source archive `v259_v261_unit_rgb_candidate_sources.tar.gz`.
Focused candidate/native tests27 passed. Full suite rerun pending at launch.
Formal v248c remains healthy (2275 confirmed); noGPU0 usage, no adopted fix yet.

Full CUDA-enabled suite after v259-v261 launch:1309 passed,15 warnings in34.38s.
Additional CPU SH-bound test passed subsequently. v262 read-only source audit
uses SH addition theorem/Cauchy-Schwarz to obtain a sufficient all-direction
color upper bound. Source1,239,507 leaves has19,292 guaranteed dead channels in
13,405 rows (19,291 below a1e-6 margin), but ZERO among881,782 persistent rows.
13,395 affected rows are measured-single state3,10 unverified state0. This does
not justify blanket projection of existing SH or identify the global canopy
bottleneck. No existing foliage SH was changed.
v263 is a read-only .99 candidate-opacity geometric coverage counterfactual
from v255800; candidate logits are restored in memory and no model is exported.
It is NOT a training target, physically justified leaf material or a fix.
An optional further intrinsic candidate-only render will distinguish absent
projected coverage from coverage obscured by the existing scene.

### v259-v261 finished; guarded search/density controls continue

v259 corrected-color800 tree improvement+.185312231dB vs source; v260 ray-depth
adds no benefit (tree-.018958509 vs v259). v260400 moves all142,731 candidates:
source-depth ratio range[.86675,1.15979],medianworld motion.50032,90th1.34084,
max5.75604. Hence absence of improvement is not an inactive geometry optimizer.
v261 RGB guard800 tree15.3869170547/interior15.8443368077/boundary13.4680277308,
rigid16.6082545320/hard14.6443448265. Tree+.061999838,47/48 improved; hard
mean+.005838374, worst748-.02265,738-.02240,707-.01947,671-.01631.
The large909 building regression is suppressed, but canopy see-through remains.
The DC repair itself only changes tree mean by~.00127 vs v255; not the cure.

v264 coverage probes demonstrate both missing coverage and scene occlusion.
At660,55,934 initial high-surface-contribution canopy pixels:15,378 have
candidate-only alpha<.25 even at hypothetical opacity.99;15,719 have intrinsic
alpha≥.95, of which9,910 still retain surface alpha>.25 in joint rendering.
43,437 total retain high surface contribution. These thresholds diagnose the
CURRENT candidate geometry, not a claim that every canopy-mask pixel is leaf.
Both this ceiling and the intrinsic probe are read-only; no hardened model saved.

New `blocking_aware_search_depth` clips only an unverified search center to.98
times a finite opaque native blocker when closer than the MoGe prior; it never
labels that center as true leaf depth or modifies a building. Source ray samples
and low opacity are unchanged; only RGB learns candidates. v2658-step smoke
completed with source fingerprints unchanged and actual opacity updates.
1,283/5,984 sampled source rays had a closer opaque blocker. Its source archive
matches its recorded main-script SHA. CPU candidate/math tests10 passed.

At21:21 Shanghai, v266 guarded MoGe256-rays baseline(GPU1),v267 guarded
blocker-aware256-rays(GPU2),v268 guarded MoGe1024-rays(GPU2) were launched800
steps. All use same source, RGB guard3, DC projection, fixed ray positions,
same3 depth ratios and optimizer schedule. The density test remains within the
unchanged2M global foliage budget; no branch fabricates verified leaf status.
Baseline is repeated with the exact same new source for controlled comparisons.
Archive `v266_v268_guarded_search_and_density_sources.tar.gz`.
No accepted production/map change; original building parameters remain protected.

v248c2500 checkpoint saved successfully (3,185,227,462 bytes); capture/encoding
and NFS save explain its multi-minute pause, not an interruption. Render-only
extraction1,453,919,065 bytes/148 storages completed79.97s with source unchanged.
Native canonical48: tree15.2866520882, interior15.7296470602, boundary13.4358631571,
rigid16.6292753220, hard14.6570877433. Versus v1982500 tree-.001942297,hard-.003510237;
the independently proved negative-authority repair is not a visible cure here.

Same-code contribution controls v270(v259) versus v269(v261) show RGB guard
reduces new-candidate rigid responsibility44,171.2148→1,365.6444 (~96.91%), while
canopy78,691.0938→24,140.3711 (~30.68% retained),sky5,305.6577→777.5216.
This is real native contribution reduction, not merely recoloring a wall proxy.
v267400 blocker search adds only+.008286dB tree over v266; v268400 density adds
+.056011dB (total+.085642dB vs original,48/48 trees improved,hard+.010146dB).
Dense case557,514 candidates,totalfoliage1,797,021<unchanged2M cap.

Identified and repaired an additional NEW candidate-diagnostic ownership
mistake: the original candidate loss allowed known-rigid/sky GT residuals to
fund tree-branch opacity. `candidate_rgb_loss` now defaults to true image RGB
only on canopy, original immutable scene RGB on rigid/sky; legacy control is
explicit. Tests show background residual previously grows wrong opacity but
now retires it, with canopy gradients EXACTLY unchanged. Existing production
trainer already separates structural_prediction from mixed canopy RGB in
`_branch_isolated_rgb_losses`; this is NOT grounds for changing that renderer.
v266-v268 were already imported with the prior loss and keep their archived
protocol unchanged. New optional footprint optimization bounds candidates to
half-to-double initial sigma, learned only by RGB; no forced opacity increase,
no source parameter changes. v2718-step CUDA smoke is checking this and the
corrected ownership loss before efficacy trials. CPU footprint/ownership tests3
passed; a fresh full suite is running. No scene or map has been adopted as fixed.

v271 native smoke completed: frozen source unchanged; 11,862 footprint codes
actually updated (max |code| .0937544), nonzero scale gradients. Full suite
1,314 passed,15 warnings. v272 fixed versus v273 learned footprint were launched
on GPU1/GPU2 with identical557,514 low-opacity candidates and corrected
noncanopy source-reference RGB ownership; 800 steps,48-view checks0/400/800.
Both manifests recorded main SHA40a112b5dcfbdaf808d783193ff1e60b40feb1441775ff4b56ba209486f2ea91.
Archive `v272_v273_owned_dense_footprint_sources.tar.gz`.

v266/v267/v268 completed800 with unchanged frozen-source fingerprints.
Blocker search adds only+.011571dB tree over its matched control. Dense sampling
adds+.105231dB over control,total+.167231dB versus source,47/48 tree views
improve; interior+.194283,boundary+.110849,rigid-.000141,hard+.013702dB.
Mean building scores conceal hard-view regressions:738-.155018,674-.082623,
748-.068596dB. Native image660 still clearly shows light building through
lower canopy. This is NOT an accepted cure. v274 read-only native contribution
and geometric coverage ceiling audit launched on GPU2; preset dense ROIs are
diagnostic only, never fitting masks or replacement for the48-view evaluation.

v274 completed. Dense candidates native canopy responsibility70,837.18,
rigid4,547.59,sky2,547.12; only30,187/557,514 opacity>.01. At660 actual canopy
surface alpha .362426→.357088, versus .151526 under the nonphysical .99-opacity
coverage probe. Missing projection pixels15,378(old sparse)→3,171(dense), but
14,648 intrinsically dense pixels still lie behind native blockers. These
semantic-mask counts include natural gaps and are NOT leaf existence truth.
v275 high-density blocker-aware search launchedGPU2 with same source-owned
RGB protocol as v272; only search_depth_center changes. Original source remains
fully frozen. Low-density blocker search alone was ineffective.

v272/v273400 matched: footprint learning adds+.083484dB tree over fixed scale,
total+.156308dB over source,47/48 improve; hard mean+.011885 but view738-.260728.
Do not adopt based on a mean. Original dense ROIs sometimes already have nearly
opaque volume (690/768 alpha~.98), missing visible lower-crown leakage. Preserve
them and add supplemental evaluation-only ROIs657(420,170,610,270),
660(470,205,630,335),690(40,110,160,250), selected by inspecting complete native
images, never used for training. v276 repeats completed v268 contribution
audit with these extra ROIs; main48-view metrics unchanged. Density comparison
regression test additionally verifies seed-image identity and capacity accounting;
comparison tests2 passed.

v272/v273 finished800: fixed tree+.141812dB, footprint+.265150dB versus source;
footprint improves46/48 tree views but hard738-.423289dB,674-.083287,707-.071297.
All source parameter hashes unchanged; this does not guarantee unchanged
building pixels. Do NOT lengthen the footprint branch simply for mean PSNR.
v276/v278 supplemental ROI660 confirms actual source surface contribution
.424097→.418496(dense fixed legacy ownership)→.409952(learned footprint owned
RGB); source volume .553912→.559863→.569121. Severe leakage is still present.
These latter two protocols differ in ownership AND footprint, not a single
causal ablation. v272/v273 is the valid matched footprint control.

Added optional `SpatialCandidates`: static XYZ displacement bounded per-axis
by8 initial sigmas (config cap16), zero-code exact initial XYZ, no footprint/
opacity change by construction. Learned only from all304 training-view RGB;
old foliage/buildings remain frozen, all48 evaluation views excluded. Distinct
from prior ray-only displacement: permits lateral alignment as well. Position,
ray-depth and footprint axes cannot be combined in this diagnostic. v2778-step
smoke launchedGPU2; first native position-gradient L1 .000131848, opacity
updates1494 rows. CPU spatial/comparison/ownership/footprint tests6 passed.
Archive `v277_spatial_smoke_sources.tar.gz`. No accepted production-map change.

v277 completed8 with original source unchanged,11,863 position-code rows
actually changed,max |code| .0935711. Fresh full suite1,316 passed15 warnings
in35.18s. v279 fixed versus v280 spatial candidate control launched3200steps
(GPU1/GPU2,supervisors29212/29213),same557,514 candidate initial points,
source-owned background RGB+guard3,unit DC,48 excluded views. Only position
radius0 versus8sigmas differs. Checkpoints/evaluation every800, LR holds until
half the3200 horizon then decays10x; consequently its800 is NOT the old800-step
schedule. Archive `v279_v280_spatial3200_sources.tar.gz`. Do not adopt a candidate
map merely because the frozen building parameter fingerprints pass.

New independently reproduced production optimization defect: scale backward
held peak opacity fixed, but post-Adam `restore_integrated_optical_mass` changed
opacity to preserve tau*pre-step area. Missing chain rule can make the computed
scale descent an ascent after retraction. CPU analytic and native CUDA tests
both show this sign reversal; corrected derivative matches retracted finite
differences. Native rasterization itself remains correct and UNCHANGED.

Added `mass_preserving_scale_alpha` and training-only `MassConsistentFoliage`.
Forward alpha is bit-identical (zero-valued gradient correction); the added
geometry Jacobian uses the same gated RAW scales and same world area proxy as
post-step mass compensation. Opacity's own gradient is unchanged. Restricted
opacity-only backward cannot acquire geometry gradients; blocked geometry rows
remain zero. Static compensated training now defaults to consistent gradients,
with `--no-mass-consistent-scale-gradients` for explicit legacy control.
No model/Adam tensor schema changes or renderer/CUDA edits. Resume requires
`--allow-optical-mass-gradient-repair-resume` plus trainer repair permission
when switching; missing historical contract key meansFalse. Other contracts
remain strictly compared. New helper hashes recorded. Full suite1,323 passed,
15 warnings,38.12s. Initial targeted test invocation missed required LD_LIBRARY_PATH;
rerun corrected environment; fixture replacement groups also fixed to valid
contiguous canonical IDs. Neither was a training crash.

v281 actual-canopy read-only Jacobian audit launchedGPU2(supervisor1653),
training cameras664/665/666, persistent rows only. It checks forward identity,
opacity-gradient identity, forbidden-row zeros, and physical scale+mass finite
differences on the real checkpoint; perturbed scales/logits are restored and
no map exported. Archive `v281_mass_gradient_sources.tar.gz`.
This repair is NOT retroactively applied to v248c, whose old code and hashes
remain in its existing process/archive; v279/v280 candidate runs do not use
production mass compensation and are also unchanged. Real quality benefit of
the production repair still needs controlled validation.

v281 finished with source restored/buildings unchanged. Active persistent rows
664:294,344,665:266,701,666:286,833; opposite raw versus retracted scale-gradient
row directions127,621/123,002/129,119 (~43–46%; counts, not optical weighting).
Along the audited correction direction, old predicted derivative at664
-.00290674 but real finite difference(h=.003)+.0000255863; corrected+.0000273763.
665 old-.00321627,corrected-.0000394922,FD-.0000407518;
666 old-.00220790,corrected-.000173861,FD-.000176841.
These are directional derivative tests, NOT claims every full Adam step ascends.

Added isolated `calibrate_canopy_mass_coordinates.py`: persistent-leaf scales,
SH and opacity only; xyz/rotation, all buildings, sky/appearance, source reference
and ineligible leaf rows frozen. Both intended controls perform the same
post-Adam tau*old-area compensation and half-to-double scale bounds; only the
training-only Jacobian differs. RGB canopy GT, immutable background reference
and guard3; exact original48-view masks/metrics preserved. LRscale .0004,
opacity .02,color .0025; no candidates or forced alpha target. Diagnostic deltas
are not production checkpoints. v2822-step smoke launchedGPU2(supervisor2920),
archive `v282_mass_calibration_smoke_sources.tar.gz`; wait for frozen audit and
actual updates before the paired efficacy experiment. v275 final800 adds only
.009764dB tree versus v272; total+.151576dB versus source,hard+.004341.

v279/v280800 (both on3200 LR horizon): fixed tree+.215192dB versus source,
spatial+.251300dB; position adds+.036108 but hard delta versus source-.009433.
Still not accepted. Later1600/2400/3200 checks pending.

v2822-step smoke finished with protected fingerprints unchanged, but first
two sampled cameras had no positive canopy loss. First loss0/scalegrad0, then
loss6.1e-13/scalegrad2e-6 arose from needless sigmoid/logit mass-restoration
roundoff on otherwise unchanged rows; L1 reference loss can turn such tiny
residuals into non-negligible gradient signs. This is not canopy recovery proof.
Added `IdentityPreservingMassFoliage.restore_integrated_optical_mass`: restore
exact original logits on already-correct mass rows (unless a physical alpha
bound requires change), optional explicit row mask preserves ineligible rows.
Both calibration controls use this same no-op repair, isolating only the
Jacobian. Production corrected subclass inherits it; legacy base remains an
explicit old behavior. Also completed the Jacobian for the training model's
`opacities` property so integrated-mass regularizers have zero scale derivative
at fixed mass. Renderer-conditioned alpha still uses its own geometry gate;
no double compensation. New tests verify zero integrated-mass scale derivative,
bitwise no-op restoration and row protection; focused tests8 passed.

v2838-step smoke launchedGPU2(supervisor4338) with these repairs; archive
`v283_identity_mass_smoke_sources.tar.gz`. Freeze helper/script edits until its
manifest/source hashes are recorded. No final mass-coordinate efficacy pair
launched yet: wait for genuine canopy gradients/updates and frozen audit.
v248c reached iteration3000, final checkpoint save pending (still old code).

v283 completed8, frozen audit passed. Genuine step8 canopy loss.0387143,
scale gradient L1 .0549564, maximum actual log-scale shift.00097847; first
zero-loss step remains an exact no-op. v284 legacy versus v285 consistent mass
coordinates launched800steps,400/800 evaluation,GPU1/GPU2(supervisors5822/5823).
Both share identity-preserving mass restore, bounds, same RGB ownership and
all protected parameters; only Jacobian differs. Archive
`v284_v285_mass_coordinates_sources.tar.gz`. Strict comparator
`compare_canopy_mass_controls.py` rejects other argument/source changes and
missing cameras; unit test passed. Do not edit their main/helpers before startup
manifests record hashes.

v248c finished successfully with full retained3000 checkpoint3,261,226,438bytes.
Render-only1,465,965,209bytes extracted70.63s,148 storage records, source values
unchanged. Native51-view evaluation completed;canonical48 means:tree15.3544421991,
interior15.7918325861,boundary13.5016590158,rigid16.6450032592,hard14.6618762612.
Versus v1983000 tree-.003655215,hard-.006877105; no substantive canopy gain.
This run used the old mass Jacobian, not the new repair. No training OOM or
interruption occurred; topology processing and checkpoint capture/save explain
the long final phase. Full checkpoint remains resumable; render-only is not.

Latest full suite1,326 passed15 warnings39.66s (before later joint-optics tests).
Added non-identity exact-K ray reprojection test (rotation, translation, unequal
focals/off-centre principal point); candidate geometry tests4 passed. No camera
transform bug found. v284/v285400: corrected tree+.209342 versus source, legacy
.210558; corrected-minus-legacy-.001216dB,hard-.000286dB. Correctness repair is
proved, but incremental quality benefit is NOT demonstrated by this result.
v279/v2801600: spatial tree+.417388 versus source,+.087214 over fixed; hard mean
-.018241 versus source, worst738-.756293 and909-.559187dB. Native660 remains
clearly see-through. Do not adopt on aggregate PSNR. v288 native contribution/
failure-ROI audit of v2801600 launchedGPU1(supervisor12509).

Added explicit joint existing-persistent-color/opacity candidate diagnostic.
`JointOpticalCandidateView` permits ONLY source features/opacity, with exact
eligible-row hooks; all old xyz/scales/quaternions, buildings, sky/appearance
and ineligible source rows stay frozen. New main flags joint_persistent_optics
and candidate_gain permit a matched source-optics-only(0) versus joint-new-
candidates(1) comparison. Immutable background reference is a SEPARATE frozen
source clone, never the changing fitted source. Candidate source support remains
unverified; no promotion fields fabricated. Audits restore optional source optics
into a separate refined base while preserving original reference rendering.
CPU joint/candidate/comparison tests7 passed. v286/v2878-step paired smoke launched
GPU1/GPU2(supervisors11831/11832),32rays/view, no position or scale learning,
archive `v286_v287_joint_optics_smoke_sources.tar.gz`. No joint efficacy pair yet.

Further production review caught an authority risk in the preliminary global
`opacities` property extension: direct false-positive/complexity priors enter
unrestricted canonical backward and must not gain ungated shape authority.
NARROWED the corrected class before any new production training: ordinary
`opacities` remains the base getter; conditioned RGB uses gated mass Jacobian;
explicit `integrated_optical_mass` detaches area in local mass coordinates,
giving exactly absent scale gradient (not a floating-point cancellation).
Regression test confirms ordinary opacity priors cannot update scales. Focused
tests9 passed. v284/v285 use only conditioned RGB plus no-grad restoration, so
their already-imported prior wrapper has the same numerical training path;
their archived/hash-recorded source remains authoritative. Do not relabel old
wrapper hashes as the new implementation. No renderer/CUDA file was changed.

### Joint candidate continuation, v289–v291

Full regression after authority narrowing and joint-optics changes: **1329
passed, 15 warnings, 31.77s**. v286/v287 smoke8 completed successfully; both
frozen audits passed. Existing source opacity changed in both, new candidate
opacity only in the enabled arm, as intended. No efficacy claim from smoke8.

v284/v285800 finished: corrected tree +.303902805 versus source, legacy
+.305676023; corrected-minus-legacy -.001773218 dB. Hard difference -.001453380.
This is a verified gradient correctness repair, NOT demonstrated incremental
image-quality improvement. Both frozen audits passed.

v288 read-only native audit of spatial1600, lower660 ROI (470,205,630,335):
20,780 canopy pixels; PSNR12.20467949 to12.64112854, surface contribution
.424097389 to.397267073, volume .553912222 to.583383083. Some real improvement,
but substantial background building contribution remains; not solved.

v279/v2802400 matched spatial-minus-fixed tree +.123802145; spatial versus
original tree +.479213834, interior +.524750729, boundary +.373036206. Hard mean
-.024163524; worst738 -.951612473,909 -.522386551,677 -.322746277. Do not adopt.
Here historical `hard` is the non-tree side of the canopy interface (5-pixel
band at640x360), NOT the entire building interior. Both full rigid and this
boundary-sensitive metric remain required, with unchanged evaluation masks.

Launched v289 source-persistent-optics-only and v290 joint-new-candidates,
1600 steps/every400,1024 rays per training seed view,557,514 hypotheses;
physical GPU1/2 only. All geometry and building parameters frozen. Source
features/opacity authority limited to previously verified persistent leaves.
Immutable original background reference remains separate from fitted source.
Archive `v289_v290_joint1600_sources.tar.gz`; smoke-derived matched commands.

Added v291 onGPU1, identical to v290 except extra rigid preservation weight
3 becomes0. Base source-reference RGB supervision remains weight1. This is
an explicitly labeled hyperparameter hypothesis, not a proven bug or removal
of building protection. Current regional-mean objective gives total rigid/tree
per-pixel weight ratio 4*Ntree/Nrigid: training quantiles0/10/50/90/100 percent
are0,0,.0863581,5.3188042,52.7202402. Eval examples660=4.98971,690=39.43703,
738=2.22350,909=.27674 (evaluation data not used for updates). Compare v290/291
as a one-argument guard control; never compare three changing axes as causal.
All new runs remain diagnostic hypotheses, not production map exports.

Read-only `canopy_rgb_region_audit.py` splits full rigid, historical tree-side
interface, and its rigid complement; 2 unit tests passed. v292 native48 audit
of spatial2400 completed:738 full rigid18.550779 to18.186272; interface14.028181
to13.076567; away-from-interface18.987404 to18.726889. Thus degradation is NOT
merely the boundary band's small denominator.909 away-from-interface21.258286
to21.249935, primarily boundary damage.660 away-from-interface15.610953 to
15.610720; lower-canopy ROI PSNR12.204679 to12.699286, surface contribution
.424097389 to.391965240. Still substantial see-through. Archive
`v292_rigid_audit_sources.tar.gz`; original acceptance masks unchanged.

v2881600 per-depth candidate contribution (canopy,rigid):.65=(39039.738,
3273.450),.8=(89057.781,3884.553),1=(76731.367,1369.450). Nearest layer does not
dominate useful contribution and has higher rigid/canopy contamination. This
does NOT motivate indiscriminately translating every candidate toward camera.

Added optional READ-ONLY `--rgb-feasibility-probe`: fixed xyz/scales/opacity,
native render candidate DC black and white, preserve original remainder and
native interleaving, restore colors in finally. RGB interval [black,white] is
a per-pixel relaxed color space. Distance of GT to this interval is a necessary
color-only error lower bound, NOT realizable shared colors, opacity supervision,
or ground-truth foliage geometry. Checks scalar nonnegative optical weight and
actual unit-RGB containment. Helper tests2 passed, region+feasibility4 passed.
v293 spatial2400 read-only native48 probe launched GPU2; all training untouched.

v293 completed, numeric containment/scalar-weight checks passed. At660 lower
ROI, current raw MSE .053712018 and best per-pixel candidate-only color bound
.051046949:95.0382% residual remains,90.5005% pixels outside the unit-RGB
reachable interval by more than1/255. At690 supplemental ROI:95.4141% error
remains,97.6964% pixels unreachable. This rules out **only changing new
candidate colors at current geometry/opacity/frozen source** as the cure; it
does NOT by itself distinguish old-source color errors from opacity/geometry.
Archive `v293_color_feasibility_sources.tar.gz`. Full suite1333passed15warnings
46.08s before the subsequent nonnegative-floor helper test.

Added `--persistent-black-floor-probe` to separate that ambiguity: temporarily
black out all previously verified persistent leaf SH (including rest terms)
and candidate DC, leaving opacity, geometry, buildings, ineligible leaves and
sky untouched. Restore every modified feature in finally. The remaining RGB
is a lower floor under **any nonnegative leaf radiance**, with no upper RGB
cap assumption. Any GT below that floor cannot be reached by leaf-color-only
optimization. Residual includes all immutable branches, not automatically only
building light. v294 spatial2400 native48 probe launchedGPU2(supervisor22287).
Three feasibility helper tests passed. No training objective or checkpoint
changed by either counterfactual audit.

v294 completed:660 supplemental ROI fixed-opacity nonnegative persistent-color
floor leaves82.0474% current error,66.7998% pixels below floor.690 supplemental
ROI leaves only1.60275%; do NOT assume the same local bottleneck everywhere.
Archive `v294_black_floor_sources.tar.gz`.

v295 completed: `BlackRadianceCandidateView` returns black SH in an ephemeral
render state, preserving original xyz/opacity and ALL surface parameters. Sky
radiance excluded. Native alpha unchanged within2e-5. Thus remaining render is
the actual surface branch contribution through current foliage, not an opaque
wall-only rerender. At660 supplemental ROI, this building/surface-only floor
already leaves70.0826% current error,65.3272% pixels have a channel below floor.
At690 supplemental ROI, only.0329461% error remains (.761905% pixels). This
directly confirms excessive surface transmission in660, not merely foliage
color contamination. Does NOT identify which surface geometry is wrong or
authorize deleting it. Archive `v295_building_floor_sources.tar.gz`.

v279/v2803200 both completed and frozen audits passed. Fixed tree+.368528009;
spatial+.514990747 (+.146462739 versus fixed), interior+.559893350,
boundary+.405970951. Spatial rigid-.018915613,hard-.028166691. Native660 still
shows lower crown/background structure. No adoption, no production export.

v289/v290400: source-optics-only tree+.207058867; joint+.261904538; incremental
candidate+.054845671. Joint rigid-.007971247,hard+.004335284; early efficacy
only, not final acceptance. v291 still running; compare equal steps only.

Added read-only `--training-gradient-conflict-probe`: re-evaluates each of the
304 training views at a fixed snapshot, separates tree GT L1 gradients from
immutable-background L1 gradients (original total rigid weight4). Keeps
positive/negative sums separately; no optimizer, no parameter update. Three
evaluation ROI contribution vectors are REPORTING weights only, never fitting,
pruning or birth authority. v296 spatial2400 audit launchedGPU2, includes48
native reporting views then304 training-gradient views. Source scope frozen;
explicitly rejects joint-source snapshots until their authority is supported.
Latest full suite **1336passed,15warnings,40.73s**.

v289/v290800: existing-optics-only tree+.287968973; joint+.467845281,
incremental+.179876308. Joint interior+.529723704,boundary+.312377572,
rigid-.014628331,hard-.003299296. Worst hard738-.515817; do not adopt on means.
v290/v291400 matched guard control: removing EXTRA weight3 adds+.073142290
tree, but rigid mean further-.011656364; worst full rigid634-.160374 versus
source. Base background RGB remains active. No success claim on guard removal.
v297 read-only joint800 native48 building-floor audit launchedGPU1 to assess
actual transmission, not merely color change.

Implemented OPTIONAL `--surface-rgb-feasibility-weight` in candidate diagnostic,
default0; **NO RUN HAS ENABLED IT** pending explicit user response to the async
controlled-loss question. New `BlackRadianceView` wraps existing optical
authority without modifying/fabricating metadata or parameters. Auxiliary is
mean positive excess squared of actual surface-only RGB over GT on training
canopy pixels. No opacity target, depth pseudo-truth, inference mask or sky
penalty. Uses unchanged native mixed ordering. Unlike older scalar front-alpha
or contributor-ceiling bounds, directly differentiates actual transmitted
surface radiance. This is a proposed optimization aid, NOT a newly proven
renderer derivative bug or adopted production fix.

CPU example proves nonconvex appearance/opacity coupling: background.7,
leaf.8,alpha.2,target.1 gives ordinary RGB opacity gradient+.1 (thins), while
surface feasibility gradient-.644 (needs extinction). Native CUDA fixture
confirms wall-front opacity gradient negative and wall-back exactlyzero,
finite-difference match. Default branch remains off. Full latest suite
**1340passed,15warnings,36.67s**. One earlier test invocation used a nonexistent
test filename (no tests ran); rerun with actual comparison filename passed.
No training crash/OOM or production renderer/CUDA mutation occurred.

v296 finished all304 training views. At spatial2400, contribution-weighted
candidate net opacity gradient still requests INCREASE on96.8065% of the
currently contributing660 ROI weight (69093.2671%,65784.6641%).660 sums:
canopy positive2.32683e-6,negative5.39075e-6; background positive4.53115e-7,
negative7.60521e-9. This does NOT support blanket background-guard starvation
as the main explanation there. These weights select CURRENT contributors,
not missing/fully occluded candidate opportunities, and are not row fractions
or actual Adam-step statistics. No optimizer state was saved in these old
diagnostic payloads, so historical Adam updates cannot be reconstructed from
this gradient audit. Opacity was unchanged throughout.

v297 joint800 ROI:660 PSNR12.204679 to12.823427, surface.424097 to.394977;
surface-only error floor still73.6017% of current ROI error.690 supplemental
ROI PSNR15.770042 to17.042259 with surface only.011277 to.010790; this area's
improvement is mainly appearance, not removal of substantial building light.

v298 spatial2400 capacity audit:660 candidate contribution2154.209 across
7891 contributing rows, weighted peak opacity.720031;42.3835% contribution
from peak>.9, only.7702% from peak>.99. Existing weighted peak.246496.
Candidate-only .99 opacity counterfactual still leaves660 lower ROI surface
.274648 (actual.391965). Increasing existing verified persistent peaks too
in v299 leaves surface.244492,volume.755187;47.2185% pixels still below the
surface-only RGB floor,34.0558% current error remains. This near-opaque test
strongly points to coverage/depth/footprint limitations; it is NOT a training
target or a solved reconstruction.690 counterpart can suppress the residual
surface to effectivelyzero. Archives v298/v299 preserve each audit version.

Added optional exact peak1 read-only ceiling, using finite float32 logit20
(not infinite parameters); default.99 preserves earlier probe behavior. This
is for checking geometry capacity, never actual model optimization.

v289/v2901200 joint tree+.599397699 versus source,+.221918186 versus optics
control; interior+.676386178,boundary+.391678254,rigid-.012444258,hard+.015805503.
v290/v291800 removing extra guard gives tree+.624495228 but rigid-.043456137
and hard-.035080850 versus source: no adoption.

Under the already-authorized RGB-only candidate search, implemented matched
`depth_search_policy`: fixed layers versus deterministic log-depth stratified
volume, three samples per source ray over[.4,1.2] times calibrated prior. Same
training ray pixels,557,514 total candidates, initial opacity.001, source
parameters ALL frozen, no added positive depth/opacity targets. Broader
stratification is a SEARCH hypothesis, not a measured volume or evidence
promotion. Exact-K rotated/translated reprojection and RNG-isolation tests
passed; comparison tests passed. v300/v3018-step capacity smoke launchedGPU1/2,
archive `v300_v301_stratified_smoke_sources.tar.gz`. Check near-opaque native
coverage before committing to long RGB training. The pending RGB-feasibility
auxiliary remains OFF in every run; it cannot by itself solve missing geometry.

### v302–v310: capacity limits, source-color regression, and controlled follow-up

v302/v303 exact-peak-one READ-ONLY capacity probes (finite logit20) compare
fixed versus stratified candidates. In the660 lower ROI, persistent-plus-new
surface contribution ceiling improves .243370 to .209323, surface-only RGB
MSE floor .0179603 to .0124721. This is capacity, NOT fitted image quality.
Missing projected pixels increase397 to690; depth stratification is no cure.
v304/v305 closer ranges [.2,1.2]/[.1,1.2] worsen this floor to .0140073/
.0143943 and increase holes1599/2184. No forward-shifted model retained.

v289/v2901600: source optics control tree+.400747; joint candidates+.634894
(increment+.234147), rigid-.012155, boundary-hard+.021164.711 tree still
regresses-.731060 and738 hard-.583671. v291 reduced guard gives tree+.816813
but909 hard-1.477074: rejected. Native660 lower crown visibly still leaks.
v279/v2803200 fixed/spatial candidates (all source frozen) finish with
tree+.368528/+.514991; spatial rigid-.018916, hard-.028167. Not adopted.

v308 read-only source-only1600 component swaps isolate711 regression:
original tree17.439201, DC-only17.427233, directional-SH-only16.948353,
opacity-only17.439245, joint16.671598. Nonlinear effects must not be summed.
738 boundary-hard regression is instead largely opacity-linked. No source
parameter edit retained. All881782 eligible original SH-rest norms <4,
so the norm cap did not initially clip them. Directional-SH learning-rate
conditioning is a hypothesis, not a proven universal renderer bug.

Added optional source-rest actual Adam step multiplier (default1), preserving
DC and unauthorized rows; multiplying gradients alone would be normalized
away by Adam. Separate-Adam-group equivalence and exact-zero restoration tests
pass. v309/v310 source-optics-only800 compare multiplier1 versus.05, GPU1/2,
all geometry/buildings frozen, candidate_gain0, auxiliary0, eval400/800.
Archive v309_v310_source_sh_step_sources.tar.gz. No efficacy claim yet.

v306/v307 main spatial1600 compare fixed/stratified hypotheses,557514
candidates, initial opacity.001, all original parameters frozen, guard3.
At800, tree+.249501/+.229274; stratified does not yet outperform fixed.
Rigid-.004334/-.003795, hard-.008163/-.006368. Both continue. All48 original
evaluation views excluded from new fitting, but not blind historical holdout.

Diagnostic checkpoints now publish atomically before metrics and include
Adam state, parameter layout, schedule horizon, training order and RNG.
Verified v306 candidates_0800.pth loads with these fields. This does not add
a production/diagnostic CLI resume feature. CPU image cache384 fits352 RGB
images instead of cyclic8-entry thrashing; no extra GPU image cache.
Latest CPU suite:1328 passed,22 CUDA-dependent skips,15 warnings. GPU tests
are separate; do not report skips as passing. No training interruption/OOM
observed in these runs, and GPU0 remains excluded from every experiment.
