# v125: matched 2k outcome and canonical depth consistency

## Actual matched outcome

`eval_v124_2k_native_prespecified51` versus `eval_v125_2k_native_prespecified51`, same immutable 1k prefix, opacity LR 0.002, normalizer-only treatment:

- Canonical48 mean-view tree improvement: 0.00292076 dB.
- Interior: +0.00299897 dB; boundary: +0.00241624 dB.
- Rigid: -0.00021124 dB; hard pixels: +0.00021462 dB.
- Surface xyz/scale/rotation/opacity fingerprints are identical.
- View 690 still has very obvious church visibility through the dense crown.

This is **not an effective reconstruction fix**. The 360x correction in the scalar derivative is real, but does not repair geometric overlap or accepted material capacity.

## Independent observation consistency

`scripts/audit_canopy_depth_consistency.py` uses exact per-view intrinsics and camera-z unprojection/reprojection. Identity and translated-camera round-trip tests pass. It does not load a trained foliage model or change any evidence. Source views are the preselected canonical48; targets are four nearest camera positions in the same seq2 with available MoGe observations. Both source and target pixels must be valid tree pixels, including the distortion mask. Samples are deterministic, at most 1024 per source. The depth scale is the production 0.8277335147998328, resolution 640.

`audit_v125_canonical_depth_consistency/depth_consistency.json`:

- 187 nonempty directed pairs, 155,917 accepted reprojections.
- 19.342% fall in the target's production thin interval.
- 24.525% fall in the local 3x3 depth range expanded by its thin half-width.
- 41.518% are in front of the interval; 39.139% behind it.
- View 690 -> 689: 1/887 in the thin interval, median signed residual +1.9889 world units, baseline 0.6586 world units.
- View 707 -> 709: 0/877 in the interval, median signed residual -3.4745 world units.

These are **not ground-truth depth errors**: camera motion changes visible leaves and occlusion, and canopy masks are not correspondences. However, coherent signed offsets across nearly whole neighboring crowns warrant checking the assumption that local refinement stability supplies a calibrated multiview depth uncertainty. In particular, refining a single image stably does not prove its metric depth agrees with its neighbor. Do not blame canonical-cross-sequence motion for this within-sequence measurement, and do not claim all differences are bugs.

## Controlled diagnostic, not a production relaxation

Two opacity-only experiments start at the same corrected v125 2k snapshot and use the same canonical camera order, 400 steps, LR .05, RGB weight 1, rigid-reference preservation weight 3, optical weight .02. No envelope, skeleton, surface geometry, surface opacity, leaf geometry, or appearance parameters update. One retains thin depth membership; the other uses the existing epistemic scale envelope, clipped before opaque independently rendered rigid geometry. Native deployment rendering is unchanged and uses no masks/depth oracles.

- `diagnostic_v125_detail_thin_400`
- `diagnostic_v125_detail_uncertainty_400`

The wide interval is a causal diagnostic only, **not authorization to fill uncertainty with matter**, nor an accepted production correction. Evaluate actual native canopy and rigid images before deciding on geometry/observation-model changes.

## Checkpoint I/O

Fine-grid pending candidates grew to millions, making ordinary Torch serialization spend minutes pickling individual three-float tensor storages. The 2k original snapshots took about 349/358 seconds to load, transform and write compact copies; original size ~3.145 GB, compact ~1.575 GB. These real end-to-end timings must not be confused with the 20k-row save-only microbenchmark (3.15 -> .23 seconds).

`outdoor/checkpoint_encoding.py` stores only candidate three-vectors as plain floats; existing restore accepts them exactly. Three tests check resume/drain equality, float32 extremes, and atomic new-output creation with unchanged input. The live trainers still execute their archived pre-I/O-change code: their saving stalls have **not yet been repaired in place**. Inspection also confirms the final teacher export contains `static_ray_birth_state`; it is not guaranteed to avoid the large accumulator. Use an explicit compact immutable copy if necessary.
