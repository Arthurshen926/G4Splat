# StMarysChurch first-principles reconstruction audit

Date: 2026-07-15

This audit is experimental and does not replace the retained MAtCha results.
It covers 512 retained-v2 training renders, 64 trajectory-heldout renders, all
20 retained charts, all 24 target-k-center charts, and the targeted repair run.

## Pipeline under test

1. Stage posed Cambridge train/heldout COLMAP data and semantic keep masks.
2. Audit training frames, retain useful marginal views, and remove hard failures.
3. Select clean charts by camera position, direction, sequence, and baseline.
4. Run posed MASt3R-SfM on explicit chart indices.
5. Build Depth Anything priors and strongly align chart pointmaps with matching.
6. Extract planes and refine chart depth.
7. Initialize 2D surfel Gaussians and refine with chart RGB/geometry plus all
   retained real views for dense RGB and monocular depth-order supervision.
8. In G4Splat, optionally propose trajectory-aligned views, preserve visible GS
   pixels, inpaint disocclusions with See3D, update planes, and repeat refinement.
9. Render heldout views and diagnose RGB residual, depth, alpha, semantic masks,
   connected artifact components, and multi-view support.
10. For targeted repair, classify independently triangulated component patches,
    edit unsupported foreground geometry, reseed clean geometry, and distill the
    unaffected region during a short refinement.

## Earliest confirmed failures

### Camera representation

- The 1,423-view train split has 13 quaternion norm errors above `1e-4` and four
  above `1e-3`; all are recoverable by canonical normalization.
- The old n24 dense set retained all four severe poses and selected
  `seq4/frame00190` as a chart.
- Heldout `seq1/frame00109` has norm `1.00747`. Raw conversion produces a
  non-orthogonal matrix; normalization restores the annotated camera center.
- At identical 1600x900 resolution, retained-v2 heldout PSNR changes from
  `15.825` to `15.921`. The affected frame changes from `11.24` to `17.42 dB`.

### Alignment mask propagation

- n24 alignment retained 50.94% of chart pixels, but initialization retained
  71.53% because the alignment mask was not written back to confidence.
- 25.59% of all chart pixels, or 36.44% of initialized pixels, were therefore
  rejected during alignment and then re-admitted during Gaussian initialization.
- Saved confidence had a zero fraction of 0.0 across all n24 charts.

### Catastrophic aligned chart

- `seq12/frame00114` has aligned/prior relative depth median `3.004`, P90
  `4.292`, and 99.9% of valid pixels above 25% relative error.
- It still entered initialization with positive confidence. This is now guarded
  as a whole-chart alignment failure rather than treated as ordinary depth noise.

### Supervision policy

- The old n24 command used RGB masks `[0, 2]` and geometry masks `[0, 1, 2, 3]`.
  It therefore kept sky in RGB while deleting trees from geometry, opposite to
  the intended `[0, 1, 2]` policy with trees retained.
- `dense_regul=strong` kept monocular depth weight 1.0 for all 30k iterations.
  With 1,370 views, each view was sampled about 22 times on average, versus about
  59 times for retained-v2's 512 views. Broad smooth depth sheets can therefore
  remain underconstrained by RGB.

## Output-scale evidence

- Retained-v2 train artifact fraction: median 3.4%, P90 18.7%; 113/512 views
  exceed 10% and 26/512 exceed 25%.
- The worst views form broad, smooth depth surfaces around tree/occlusion
  boundaries. They are not well described as isolated floaters.
- n24 is worse on 269 of 512 common training views and better on 168. Its heldout
  PSNR is 14.368 before camera normalization, versus retained-v2 15.825.
- Adding 261 reseed points changes effective alpha by only `0.000135` and depth by
  `0.000753`: the existing wrong foreground surface is nearly opaque and occludes
  the new geometry. Reseeding without erase/gating cannot repair this failure.

## Independent fixes now implemented

- Canonical quaternion normalization in QC output, chart selection, MAtCha camera
  loading, and the 2DGS renderer; invalid poses bypass reject quotas.
- Versioned audit/QC reuse checks so stale artifacts cannot silently mask a broken
  runner command or input-policy change.
- Alignment masks persist as zero confidence in `charts_data.npz`.
- Globally collapsed aligned charts are disabled and recorded.
- RGB and geometry both use masks `[0, 1, 2]`; trees remain supervised.
- Dense depth uses `strong_decay`: weight 1 through 7k, then 0.1, 0.01, 0.001,
  and 0.0001 in successive late intervals.

## Corrected preprocessing gate

- The versioned audit rejected 39 of 1,423 source views and retained 1,384.
- All retained COLMAP quaternions were canonicalized; 13 required material
  normalization and the maximum input norm error was `0.042866`.
- The corrected target-k-center selector produced 24 strict-clean charts. Every
  selected chart has zero `mask_0` area, passes masked image-quality thresholds,
  and all eight position/direction clusters have at least two support charts.
- The malformed old chart `seq4/frame00190` is no longer selected.
- For the representative failed region `seq2/frame00109`, the two closest charts
  are now `seq10/frame00078` (2.53 m, 31.0 degrees) and
  `seq10/frame00075` (4.48 m, 17.2 degrees). This improves local support over the
  old n24 set, but still needs a fresh alignment and 7k render gate to verify the
  actual pointmap geometry.

## Next controlled experiment

Run one new StMarysChurch 7k screen from the normalized QC dataset and regenerate
MASt3R/alignment because the old charts are contaminated. Gate it before 30k on:

- no selected soft/rejected pose or frame;
- masked chart confidence zeros matching the combined alignment mask;
- no catastrophic aligned chart left active;
- target-region chart coverage from at least two baseline-separated clean views;
- train and heldout RGB/depth contact sheets without new broad smooth sheets.

Only after that gate passes should the same frontend continue to a 30k
`strong_decay` run. Local component repair remains downstream and must not be
used to hide a failed geometric frontend.
