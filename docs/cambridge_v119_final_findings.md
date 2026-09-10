# v119 final evaluation and implementation audit — 2026-09-07

Production revision: `c664646`. This audit does not change the trained model or production implementation.

## Final images

The retained 30,000 checkpoint completed its fixed19 evaluation with `exact_implementation_match`. Sixteen seq2 views have canonical canopy authority; the three seq1 views are diagnostic only. Values below are mean per-view float-image protocol PSNR, not historical uint8 PSNR.

| Region | v117 30k | v118 30k | v119 21k | v119 30k |
|---|---:|---:|---:|---:|
| Tree | 12.46408 | 11.76292 | 12.38332 | 12.47154 |
| Inside boundary | 12.30650 | 11.92725 | 12.28894 | 12.33009 |
| Rigid boundary proxy | 14.81031 | 14.88171 | 14.63784 | 14.67535 |
| Non-tree rigid | 16.78153 | 16.85633 | 16.67649 | 16.71317 |

v119 recovers v118's tree regression but provides almost no gain over v117 (+0.00746 dB tree), with worse rigid boundary (-0.13496 dB). View 690 still visibly shows the church through sparse foliage at 30k. View 707 rigid boundary drops from 14.008 to 13.491 dB versus v117.

Surface-only non-tree PSNR is 16.89189 (v117) versus 16.88536 (v119). Mixed-render non-tree quality drops by 0.06836 dB while the surface-only difference is 0.00653 dB. This counterfactual implicates foliage contributions in most of the observed difference; it does not prove surface geometry accuracy.

Artifacts are under `/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1/formal_v119_material_local_canopy_30k/parallel_eval_fixed19_30k/`.

## Confirmed implementation/method defects

### P0: verification is incorrectly sufficient for shared optical ownership

`_persistent_optical_ownership_states` (trainer:16172) unions positive demand with *any durable detail* and broadcasts the union across replacement_group. Consequently, a single verified detail with negative evidence, no positive demand and no other group members is classified as shared. A direct CPU reproduction returned conflict=True, pure_negative=False, shared=True. The primitive supplies its own exemption from negative evidence.

`_apply_persistent_rigid_front_conflict_policy` clears all shared opacity gradients and momentum. `_apply_strict_rigid_front_opacity_policy` then zeroes the strict source on those same rows. Thus the exact negative loss exists and passes renderer tests but is deliberately removed before optimization. This is a policy correctness problem, not a CUDA numerical failure.

At 30k, 98,043 of 98,092 conflict rows (99.95%) are shared. During 18,001–25,500, summed strict source magnitude retained is 0.00126345 versus 4.55387331 deferred: approximately 99.97% is deferred. Summed row-events are not unique primitive counts.

### P0/P1: the shared-state exit condition is not sufficient or universally reachable

The freeze policy accepts group-wide durable detail as evidence, but the localization selector requires an envelope with debt, recorded positive demand, two canonical support cameras, and eligible detail bandwidth. A detail-only conflict group or a group with no recorded positive demand can therefore be frozen without qualifying for the recovery path. Clearing successful children's demand/debt ledgers does not remove the durable-detail shortcut; the next conflict can immediately classify the row as shared again.

Localization performed 30 events, 2,851 factorizations and 2,856 detail splits. These are events, not unique successfully repaired groups. Handoff mean retired fraction was only 0.004475 at 25.5k. Completion, localization and handoff then stop/freeze while approximately 96,700 shared conflicts remain. Verification maintenance works, but its success is not proof of resolved optical conflict.

### P1: the claimed surface-opacity freeze has an explicit bypass

`appearance_only` freezes gradient-based surface geometry/opacity updates. The independent `_replacement_audit` (trainer:20561) still directly writes structural opacity, including when the selected view is noncanonical; the run uses retirement budget 0.0025. Its proof uses isolated/mixed layer depth means and cross-sequence RGB/semantic accumulation, with no equivalent scene-canonical unknown gate at this call site. This is incompatible with a claim of fully immutable rigid optical geometry. Prior checkpoint checks found 761,759 decreasing opacity rows from 3k to 21k while xyz/scale/rotation hashes were identical. Whether particular retirements are physically wrong requires per-row evidence; the bypass itself is confirmed.

## Optical coverage bottleneck

During 18,001–25,500 there were 619 scheduled MoGe events with supported hit pixels. Their event-mean hit alpha was 0.007028 versus target 0.715912. The comparison uses the loss's intrinsic thin-depth interval channel, not mixed contribution alpha. It indicates severe insufficient optical support at the supervised depth; it does not prove whether missing geometry, contributor eligibility or frozen opacity is dominant per pixel.

The same deficit already exists before 18k: 1,271 supported events have event-mean hit alpha 0.003297 versus target 0.716541. It is not solely a late-polish regression. Ordinary volume birth stops with the topology window at 18k; subsequent opacity optimization cannot create a missing primitive in the supervised interval.

Total volume integrated mass grew from 10.1713 at 18k to 12.6273 at 25.5k, then fell to 12.3625 at 30k. Total mass therefore cannot substitute for correct-depth canopy coverage. A large loss or nonzero gradient count cannot establish successful geometric/optical reconstruction.

## Final 16-view front-contribution diagnostic

The separate 30k diagnostic completed using the v119 repository and records the implementation hashes in `parallel_exactfront16_30k/rigid_pixel_optical_ownership.json`. Mean per-view intrinsic volume alpha in front of the supplied surface depth is 0.026633 on authoritative rigid pixels and 0.063639 on the hard-interface mask. The corresponding fractions exceeding alpha 0.001 are 29.98% and 59.92%. Envelope-only alpha is 0.019504 and 0.049493 respectively; detail-only alpha is 0.006482 and 0.013129. Role alphas cannot be added linearly because of transmittance.

Important limitation: event membership is exact relative to the supplied isolated surface **mean-depth** bound, not necessarily a physical first-hit or authoritative surface-owner depth. These numbers demonstrate residual contributions under that diagnostic, not a proven CUDA sorting defect or a direct measurement of the training median-depth negative mask. They must not be used to justify indiscriminately removing all foliage at those pixels.

## What is implemented and what remains unproven

- Native per-pixel surfel ordering and depth-query regression tests exist; no new CUDA ordering defect was reproduced by this audit. Volume events still use primitive centre depth across a footprint.
- Exact ray/MoGe source separation, material-tail filtering, independent topology maintenance, and successful-family ledger rearming are present and exercised by the run.
- The v119 shared-freeze rule introduces a new method defect and blocks most strict retirement. Passing the existing tests is insufficient: the shared-owner fixture explicitly expects group-member verification alone to freeze a row.
- Fixed19 final rendering is now complete. It does not constitute all-scene/held-out evaluation, and the rigid-boundary mask is a semantic proxy rather than building-instance ground truth.

## Isolated repair order

1. Replace group verification as a positive permission with measured per-row canopy responsibility. Reproduce the self-exemption and unrelated-group-member cases as negative tests before changing policy.
2. Give every shared state an attainable exit: independently check canopy and rigid responsibilities after local refinement; successful visual verification alone cannot erase a geometric contradiction.
3. Establish a fixed-rigid baseline with surface retirement explicitly disabled, and compare surface-only output before/after. Treat any desired surface replacement as a separate experiment.
4. Diagnose thin-hit failures per pixel: no in-interval primitive, contributor excluded, source blocked, or insufficient realized optical thickness. Only the first case needs new geometry; valid contributors need a usable optical update.
5. Compare each change from a common retained checkpoint; use a clean run only for changes affecting pre-18k geometry. Do not combine another set of unmeasured policies into a formal training claim.
