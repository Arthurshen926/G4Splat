# v123: intrinsic interval queries and effective optical training

## Reproduced renderer defect

Depth-query transmittance products were accumulated inside the ordinary RGB traversal. Its early exit at residual transmittance below 0.0001 could hide an entire later query interval. This is wrong for an intrinsic interval-opacity observable: opaque foreground must affect RGB visibility, but must not remove the queried volume's own optical density or its derivative.

Forward and backward now traverse queried volume contributors independently of the RGB early exit. Surface compositing and ordinary RGB traversal are unchanged. CUDA tests reproduce both opaque surface and opaque volume prefixes. They assert surviving hit opacity, finite-difference-consistent gradients, zero gradients to out-of-interval contributors, and identical RGB with/without the query. Both tests failed with the old binary; the new native suite passes all 17 tests.

An additional local-extension import fix prevents silently loading an editable binary from the other worktree. The branch-local binary is checked at rendering. No binary in the other worktree was overwritten. Test collection now selects the same local extension. Complete suite after these changes: 1068 passed.

Historical checkpoint evaluation requires explicit exact-hash causal migration. This is not an exact implementation match or a blanket compatibility bypass.

## Real counterfactual evidence, not reconstruction success

On v122 3k, correcting the query does not materially change the very low native-opacity thin coverage. Thus this renderer defect is real, but does not explain most of the present low-opacity canopy failure.

With diagnostic-only 100x volume-opacity scaling, all-static thin alpha changes as follows:

| View | Old query | Corrected query |
|---|---:|---:|
| 632 | 0.6128 | 0.6932 |
| 690 | 0.2656 | 0.6865 |
| 707 | 0.2179 | 0.2262 |
| 768 | 0.2613 | 0.6371 |

This confirms independent query semantics. Scaling is neither a saved model nor a proposed production fix; it cannot establish visual improvement or rigid-pixel safety. View 707 also remains geometrically weak even in this permissive counterfactual.

## Next causal experiment

`scripts/calibrate_canopy_optical_ablation.py` freezes all geometry, rigid surfaces, sky and appearance. It separately optimizes foliage opacity with thin-hit positive evidence and rigid free-space negative evidence. Sixteen difficult seq2 cameras are excluded from calibration, and evaluated with the normal deployment renderer, without per-pixel oracle masks. These cameras were not excluded from the original 3k training, so this is a calibration holdout, not an unseen-scene test.

Diagnostic snapshots are explicitly opacity-only and must not be treated as complete production checkpoints. Compare all-view sampling against canopy-bearing-view sampling and assess effective positive updates separately from nominal steps. Remaining candidates include owner eligibility, low per-row optical update frequency, zero-footprint rows, and conflicting cross-view geometry. No claim that background leakage has been solved is warranted yet.

## Initialization validity mismatch

MoGe seed selection and rigid scene-scale support intersected keep channels 0, 1 and tree membership, but omitted channel 2 (distortion validity). They now share a tested four-channel semantic-support helper consistent with training. The initialization suite passes 52 tests. Existing diagnostic runs retain the original initialization; this correction is not silently mixed into their comparison.

Across the 352 seq2 mask records, 235,033 of 13,583,158 previously semantic-eligible tree pixels and 1,374,379 previously semantic-eligible rigid pixels are distortion-invalid. These are potential mask support counts, NOT counts of accepted MoGe depths or actual seeds: additional depth and selection filters still apply.

## Completed opacity-only ablation (not accepted)

At 640-pixel resolution, 1,200 opacity-only calibration steps on 219 canopy-bearing seq2 cameras, excluding the 16 evaluation cameras, raise mean tree PSNR from 11.75647 to 12.40208. Non-tree rigid PSNR falls from 16.49011 to 16.27239. Some boundaries deteriorate significantly; this is not an acceptable production change.

The first preliminary all-view run used the default source resolution, not 640. Its before/after gain of 0.29394 dB is internally comparable but must not be compared directly to the 640-pixel absolute PSNR.

Broader resolved-envelope/RGB experiments and thin-growth/rigid-preservation variants remain causal diagnostics. They deliberately test the representation's limitations, do not grant production ownership to new rows, and do not overwrite input checkpoints. Their final results must be inspected before selecting a production change.
