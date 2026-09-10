# Optical ownership repair — first implementation batch

Date: 2026-09-07. Based on v119; no new formal training or image-quality claim.

## Implemented

- Shared opacity protection requires measured positive demand on the same row. Neither verified-detail identity nor a different replacement-group member grants protection.
- A verified, supported detail with its own positive demand and rigid conflict can enter the existing mass-conserving split path without an envelope. Split planes prefer the detail's own conflict camera.
- Whole-surface replacement retirement runs only under `joint`. It cannot bypass `appearance_only`, `frozen`, or the immutable prefix of `atlas_residual`. Runtime ownership metadata follows this guard.
- MoGe optical audit now distinguishes zero rendered hit coverage, positive alpha below 0.01, below-target pixels, and target-met pixels. Zero coverage is explicitly NOT equated with missing geometry: filtering can also cause it.

## Validation

The first three changes pass the complete suite: **1057 passed**. An additional coverage-audit regression was added afterwards and is separately checked.

On an in-memory CPU copy of the real v119 30k checkpoint:

| State | Rows |
|---|---:|
| Rigid conflicts | 98,092 |
| Old shared exemption | 98,043 |
| Same-row measured-demand shared exemption | 96,907 |
| False exemptions released | 1,136 |
| Remaining shared detail | 41,915 |

A synthetic separated retirement source on the 1,136 released rows produced an actual Adam opacity decrease on all 1,136 rows and changed zero unrelated rows. This is an optimizer-path test, not a rendered-gradient experiment. Original checkpoints were not modified.

## Remaining work — do not call the canopy fixed

The exemption correction alone affects only a small fraction of the conflicts. A positive-demand observation is still historical evidence, not proof that the row currently supplies useful canopy coverage. The remaining 96,907 rows cannot simply be deleted without risking canopy holes.

The detail-only repair entrance is now reachable for eligible rows, but low-bandwidth, insufficient-support, unresolved, and post-localization-window states remain restricted. This batch does not claim every protected state has an exit.

Next causal experiments must measure eligible contributors and thin-interval coverage on multiple difficult views, separate absent geometry from excluded or update-blocked contributors, and test local repair under an explicitly fixed rigid baseline. Pre-18k geometry changes require a clean run. A fresh 30k run is not yet justified by these unit tests alone.

## Second batch: independent evidence and spatial birth allocation

- Fixed `StaticRayBirthAccumulator._accumulate`: repeated rays/visits from the same camera cannot repeatedly shift a cell's centre, colour or priority. The camera set already survived capture/restore, so the guard also works across resume. Historical weighted centres are preserved, not retroactively corrected; a clean initialization is required to test removal of historical bias.
- Replaced MoGe's globally concentrated top-k allocation with one best candidate per occupied image tile followed by a global score top-up. This preserves the exact proposal budget, depth/semantic gates and independent-camera requirement. It improves spatial allocation but does not guarantee all pixels are eventually selected or prove better reconstruction.
- Added regressions for duplicate evidence across resume and budget monopolization by a high-score patch.
- Full test suite, including native CUDA rasterizer tests with GPU 2 visible: **1060 passed**, no skips reported.

Real v119 18k pending visual-hull evidence has 33,182 cells; 24,303 have accumulated weight greater than the number of independent cameras. Maximum weight/camera is 662.2166. Since each confidence is capped at one, this directly confirms repeated-camera weighting in the historical state, but does not by itself quantify image degradation.

### Clean pilot

Run: `/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1/pilot_v120_camera_balanced_birth_fixed_rigid_3k_r1`.

Uses the v119 base command and seed, clean initialization (no resume), unchanged 30k schedule horizon, requested stop at 3k, retained checkpoint at 3k, surface retirement budget zero, view cache 128, GPU 0. The detached supervisor records the complete resolved command. This is a combined repair pilot against the retained v119 3k reference, not a single-variable attribution experiment or final 30k acceptance.

The first launcher attempt retained v119's checkpoints above the requested 3k stop; argument validation correctly rejected it before training. The corrected `r1` run limits retained iterations to 3k. The failed attempt directory is retained as evidence; no old outputs were overwritten.

Acceptance still requires multi-view thin-hit coverage, spatial birth distribution and fixed-rigid renders. The post-18k shared-owner repair is not exercised by a 3k pilot.
