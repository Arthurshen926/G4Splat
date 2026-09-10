# v129–v131: geometry coverage, photometric evidence, and per-view depth scale

Status: active diagnostics, **no accepted solution / no production initializer replacement yet**. All native validation uses task=None / conditioned=False with no mask/depth rendering gates. The 48 canonical evaluation views are excluded from new seeding, verification, and calibration optimization. They were not held out from the historical input model's RGB training.

## Completed canonical48 results

Mean per-view PSNR: canopy / rigid / adjacent rigid.

| Experiment | Step | Canopy | Rigid | Adjacent rigid |
|---|---:|---:|---:|---:|
| Input v125 | 2000 | 10.93692 | 16.65432 | 14.80528 |
| v128 consensus2048 all304 | 1200 | 13.85748 | 16.64025 | 14.62849 |
| v128 consensus8192 opacity-only | 1200 | 13.90581 | 16.62063 | 14.59621 |
| v129 consensus8192 DC-color / bounded-scale | 1200 | 14.13894 | 16.60660 | pending exact aggregation |
| v129 consensus2048 radius .30 / 17 bins | 400 | 13.87014 | 16.63095 | 14.60044 |
| Same + observed-sky negative weight .1 | 400 | 13.80249 | 16.63348 | 14.62275 |

These do not solve the problem: inspected 690 has much less large-scale church transmission but blurred leaves; 657 still loses much of its crown. More samples and local DC color / scale refinement yield only modest extra quality. Sky negatives improve the silhouette tradeoff but do not fix missing foreground geometry. Rigid parameters remain frozen, but native rigid image quality can still deteriorate due to incorrectly placed foliage.

The isolated opacity script previously excluded observed sky from its optimization losses. It now has an explicit `--known-sky-weight`; this is a defect of that **new diagnostic**, not a claim that the production trainer had no sky-negative path (the production trainer already includes sky evidence). Defaults remain unchanged for causal controls.

## Counterfactual coverage separates geometry from opacity

`audit_v129_native_coverage400`, from the v129 refined 8192 step400 full foliage capture:

| View | Native tree volume alpha | Counterfactual native, 100x opacity | Counterfactual intrinsic, no rigid layer |
|---|---:|---:|---:|
| 657 | .2985 | .5384 | .6484 |
| 660 | .3108 | .5528 | .6873 |
| 690 | .8581 | .9370 | .9681 |
| 707 | .8091 | .9133 | .9793 |
| 713 | .7778 | .8533 | .9809 |
| 717 | .8175 | .8844 | .9791 |
| 768 | .5476 | .7634 | .9147 |
| 772 | .4132 | .6460 | .8274 |

100x is a read-only counterfactual, never written back or proposed as a fix. The native counterfactual uses per-leaf alpha clamped to [.000001,.99], including its scale1 entry; exact unclamped baseline images are in the ordinary native evaluations. These attribution alphas are not opacity ground truth. They nevertheless show that existing projected support / depth ordering is insufficient on 657/660 even when opacity is made extreme. Changing a global opacity scalar cannot create missing geometry.

### Absolute-opacity follow-up (stronger than a multiplier)

A multiplier alone cannot exclude leaves starting at exceptionally small alpha. `audit_v129_native_absolute_ceiling400` therefore sets every leaf's base alpha to .99, preserving geometry and native ownership. Native canopy contribution on 657/660 remains .53885/.55325 (intrinsic without rigid: .57459/.62935). Thus the original geometry / projected support limit is confirmed by an absolute counterfactual, not just the 100x test. The intrinsic 100x experiment scales the kernel's opacity coefficient and can extend support beyond physical per-leaf alpha=.99; it is not itself a physical upper bound.

`audit_v131_raw_anchor_native_ceiling400` replays the source-matched pure-opacity snapshot on calibrated 2048 v1 geometry. Native learned / all-.99 / intrinsic-all-.99:

- 657: .51013 / .82665 / .98110.
- 660: .57879 / .81516 / .99773.
- 690: .86138 / .96428 / .99784.
- 713: .79874 / .91303 / .99623.
- 768: .47580 / .81027 / .99639.

Calibration materially increases geometric coverage, but some support is behind the rigid layer and the learned alpha remains lower than its absolute counterfactual. This is not solved, and setting all alphas high is not a proposed deployment fix. Updated diagnostic scale1 uses the exact original logits without clamping.

## Patch-stereo diagnostic

`outdoor/canopy_patch_stereo.py` warps 5x5 RGB patches on a source-camera tangent plane using exact intrinsics / poses, across alternative depths for one leaf. Flat patches and ambiguous depth peaks cannot provide authority. Up to six independently positioned, similarly facing canonical cameras vote; all 304 cameras still supply conservative free-space negatives. Selected points must acquire two actual native contribution witnesses after relocation. There are no layered uncertainty slabs and no target-view mask gate in deployment.

- `diagnostic_v130_patch_stereo2048_radius030_r1`: 16,968 of 361,082 rays retained, all reverified. 17 depth bins.
- `diagnostic_v130_patch_stereo2048_radius030_fine61`: 15,783 retained / reverified. Finer sampling did not resolve low coverage.
- The initial new-script launch failed on PyTorch 2.0's unsupported tuple argument to Tensor.all; fixed to chained dimension reductions. This was a diagnostic implementation mistake, not an interruption of the formal trainer. Identity-warp, translated-plane, flat-texture, and ambiguity tests pass.

Neither sparse patch result has been adopted as a replacement forest. Counts do not establish quality. The search prior still centers on the potentially mis-scaled monocular estimate.

## Larger confirmed observation-model problem: one scale is not enough

Production binds every MoGe depth to one scene-wide conversion (.8277335). The local MoGe v3 source confirms metric_scale is predicted per image, then multiplied into depth. Its supplied fov_x is explicitly degrees; this inspection did **not** find a degrees/radians mistake.

`audit_v130_rigid_anchored_canopy_depth` fits a single additional log-depth ratio on known visible, eroded rigid pixels against the frozen independent opaque rigid median depth. It does not alter the rigid scene. Example ratios relative to the existing global conversion:

- 657: 1.74425, 89,414 rigid pixels, robust log scatter .01542.
- 660: 1.86929, 59,555 pixels, scatter .01944.
- 690: .72579, 7,950 pixels, scatter .11488.
- 707: 1.34472, 32,328 pixels, scatter .04535.
- 713: .87834, 89,847 pixels, scatter .00831.

Within-view coherent errors this large are inconsistent with treating all monocular outputs as already calibrated by the same scalar. Nearest-view canopy thin-depth agreement increases from 19.34% to 33.85%; local 3x3 agreement from 24.53% to 39.79%. Accepted reprojected sample counts differ (155,917 vs159,909) because geometry moves, so these are diagnostic rates, not paired ground-truth errors. Occlusion and different visible leaves still matter.

The initial diagnostic rejected ratios outside [.5,2]. This arbitrary bound incorrectly rejected consistent ratios such as 2.085 / 2.628 / 2.736. v2 removes that bound while requiring finite positive scale, enough real rigid samples, and consistent anchor ratios. Existing v1 captures remain immutable controls. Missing/inconsistent anchor views currently retain the global conversion; that fallback remains a limitation to evaluate, not calibrated certainty.

New `--rigid-anchor-scale` arms use the frozen rigid geometry **only to calibrate canopy observations**. No rigid geometry, opacity, or appearance is trained. Per-camera coefficients and their evidence audit are stored in the fresh foliage capture and reused exactly by subsequent geometry verification and optical optimization. They are not silently refitted under newer code. Native validation does not use these coefficients or ground-truth masks for rendering. Its auxiliary thin-alpha report still uses the original global-scale MoGe interval and must not be interpreted as calibrated thin coverage.

## v131 active artifacts

- `diagnostic_v131_rigid_anchored_dense2048`: initial bounded-ratio version; 400-step raw-geometry opacity experiment `diagnostic_v131_anchor2048_raw_opacity400` uses its immutable coefficients. At step100: 14.20422 / 16.58005 / 14.62853. Views657/660 canopy 11.2528 /12.2881, still substantial visible background.
- `diagnostic_v131_rigid_anchored_dense8192`: same initial bounded-ratio version, 1,179,496 leaves /839,322 real second-camera witnesses; not adopted.
- `diagnostic_v131_rigid_anchored_dense2048_v2`: no arbitrary ratio cutoff, 353,206 leaves /237,540 real second-camera witnesses.
- `diagnostic_v131_anchor2048_consensus_v2`: global one-hit selection using that v2 geometry and its stored calibrated intervals; in progress.
- `diagnostic_v131_rigid_anchored_dense8192_v2`: denser matched v2 builder; in progress.

v131 launch source archives are created **before** launching their processes, as sibling `*_launch_sources.tar.gz` files. Some earlier diagnostic directory `executed_sources.tar.gz` files were collected after execution: specifically v130 r1 was snapshotted after generalizing the patch peak-separation API (the 17-bin behavior remains identical). They are post-run source snapshots, not exact startup-hash evidence. Formal supervisor archives remain their actual launch sources.

Current project tests: 1120 passed, 15 warnings. Root-wide optional upstream tetra extension remains outside that command. More code/test changes must be rechecked. None of these test counts establishes reconstruction acceptance.

## Follow-up at 03:05 local

Calibrated raw2048 v1 opacity400 completed: **14.52683 /16.64957 /14.68471**. No rigid parameters changed. It is promising compared with the uncalibrated path, but its difficult views still show visible background and insufficient detail.

v2 global selection completed:

- 2048: 353,206 proposals →176,829 selected /176,037 actual reverified; pure-opacity400 is running.
- 8192: 1,180,169 proposals →695,477 selected /693,058 actual reverified.

Two matched all304 1200-step local-refinement experiments are running: `diagnostic_v132_anchor8192_local_geometry1200` (raw v2 geometry), and `diagnostic_v132_anchor8192_consensus_geometry1200` (selected v2 geometry). Opacity LR .05, position LR .005 with bounded source-relative displacement, DC color .0025, bounded log-scale .003, known-sky negative .1, frozen original rigid reference / boundary preservation 3/3. All rigid geometry, opacity, and appearance remain frozen. Full learned foliage snapshots are required; opacity-only exports are not valid replays of joint refinement. Moved geometry will require fresh verification before production adoption.

`diagnostic_v132_anchored8192_patch_stereo` checks image-patch depth correspondence again with the corrected source depth scale (17 bins within +/- .30). Earlier uncalibrated patch candidates often searched around the wrong depth. This is still a diagnostic, not an accepted source of new production authority.
