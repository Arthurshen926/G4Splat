# Cambridge adaptation (experimental)

This path ports the quality-control and dense-supervision changes used by the
MAtCha Cambridge experiments into G4Splat. It is isolated behind
`scripts/run_cambridge_g4splat.py`; the original G4Splat commands remain
unchanged.

## Policy

- Start from `datasets_full/<scene>/train`.  The reconstruction frontend is
  deliberately transductive: every training image participates in QC, target
  coverage, candidate/union-Chart auditing, and dense reconstruction.  For
  StMarysChurch this is all 1,487 train images; there is no `train_opt` or
  trajectory-heldout split in the active policy.
- Audit blur, exposure, COLMAP track support, reprojection validity, and intrinsics.
  QC is an annotation, not a reconstruction split.  A hard-reject image remains
  a dense RGB/depth input and a coverage target; it is merely denied initial
  Chart eligibility until a later joint-selection audit can justify it.
- Audit raw COLMAP quaternion norms before any pose clustering. The QC writer
  canonicalizes every recoverable quaternion, records material corrections, and
  hard-rejects only non-finite or zero-norm poses. Camera loaders normalize again
  defensively so heldout evaluation cannot use a scaled/sheared rotation matrix.
- Select up to 24 audit-clean charts with target k-center coverage over camera
  position, viewing direction, and sequence. Enforce two baseline-separated views
  per view cluster, reject near-duplicate poses, and require an empty `mask_0` for
  every chart candidate.
- Compute chart quality on pixels retained by `mask_0 & mask_1 & mask_2`.
- Apply `mask_0 & mask_1 & mask_2` to chart alignment, chart initialization,
  RGB loss, dense depth prior, normal consistency, and distortion loss. The current
  `processed/masks.pkl` files have three masks; no tree mask is applied.
- Use strong chart alignment with matching loss and chunked projections.
- Persist the combined MASt3R/semantic alignment mask as zero confidence in
  `charts_data.npz`. A conservative whole-chart gate also disables an aligned
  chart when its depth has collapsed globally relative to its input prior.
- Use all QC-audited posed views for dense supervision. A screen run uses 7k
  iterations; a full run reuses its frontend and uses the 30k `long` final pass.
- Keep dense monocular depth strong through 7k, then decay it by 10x steps through
  30k. This preserves early geometric guidance without forcing a wrong
  Depth-Anything ordering to dominate late RGB refinement.
- Keep the selected real charts at full RGB weight even when dense supervision is
  active. `--downweight-input-view-color-loss` remains an opt-in ablation for
  unusually severe illumination changes; it is not a Cambridge default.
- Initialize warp-filtered surfels on a 2x2 pixel grid. This retains roughly four
  times as many anchors as the earlier 4x4 screening setting and halves their
  initial footprint while remaining substantially lighter than the full pixel grid.
- Sky (`mask_1`) is excluded from RGB and geometry losses. A low-weight opacity
  penalty (`0.01`) discourages edge surfels from expanding into the now
  unsupervised sky, and rendering uses a white fallback background. Dynamic masks
  are not opacity-pruned, so transient occluders do not create static-scene holes.
- Learn a small regularized affine RGB transform per real chart/dense training
  image to absorb exposure differences. See3D pseudo-views are not corrected,
  and train-fit rendering uses the unmodified static Gaussian appearance. Disable
  this ablation with `--no-color-correction`.
- Generate See3D cameras on a pose-neighbor graph and interpolate rotations with
  Slerp. This keeps pseudo-views on the observed trajectory and derives image
  vertical from the calibrated chart poses instead of assuming world Z is up.
  The Cambridge runner enables this with `--scene-aligned-see3d-cameras`; use
  `--no-scene-aligned-see3d-cameras` only for a legacy G4Splat ablation.
- Preserve the observed GS-rendered region of each See3D proposal and save its
  visibility mask. By default, generated pseudo-view pixels do not initialize
  surfels and do not enter RGB/depth/normal losses. The pseudo-views still take
  part in G4Splat's upstream global plane fusion, so this default is the protected
  `plane-only` variant rather than plain MAtCha.
- Use real input cameras only for mesh extraction (`--no_interpolated_views`) and
  skip TSDF image sampling safely when a view/chunk projects no valid points.
- Merge cumulative See3D plane pointmaps with a two-pass chunked projection. It
  computes the per-pixel front depth first, then retains plane points within the
  original `0.01` depth tolerance without allocating an
  `H x W x points-per-pixel` tensor. This is required once the cumulative point
  set reaches tens of millions of points.
- Evaluate all train images as an explicitly labelled in-sample train-fit audit.
  Reports include raw RGB, `mask_0 & mask_2` (`dynamic_valid`), and
  `mask_0 & mask_1 & mask_2` (`static_valid`) metrics.  This is not a
  generalization evaluation and is intentionally allowed to overlap the
  reconstruction input.

## Commands

Prepare all scenes without training:

```bash
conda run -n g4splat python scripts/run_cambridge_g4splat.py \
  --phase prepare \
  --scenes GreatCourt KingsCollege OldHospital ShopFacade StMarysChurch \
  --gpu 0
```

Run the StMarysChurch 7k screen and in-sample train-fit audit on one GPU:

```bash
conda run -n g4splat python scripts/run_cambridge_g4splat.py \
  --phase screen --scenes StMarysChurch --gpu 0
```

Promote the completed screen artifacts and run the full G4Splat stages plus the
30k dense final refinement:

```bash
conda run -n g4splat python scripts/run_cambridge_g4splat.py \
  --phase full --scenes StMarysChurch --gpu 0
```

If a downstream command fails after an expensive See3D generation and plane
stage has already completed, resume before that stage's Gaussian refinement:

```bash
conda run -n g4splat python scripts/run_cambridge_g4splat.py \
  --phase full --scenes StMarysChurch --gpu 0 \
  --resume-after-see3d-plane-stage 2
```

Re-run only the in-sample train-fit audit:

```bash
conda run -n g4splat python scripts/run_cambridge_g4splat.py \
  --phase evaluate-screen --scenes StMarysChurch --gpu 0
```

The default output root is `output_cambridge_qc_n24_30k`. A run manifest records
the exact chart indices, mask policy, dense dataset, and command. Screen/full
directory names contain a short configuration digest, so protected runs cannot
silently resume legacy all-pseudo artifacts from an older experiment.

For StMarysChurch, this three-mask policy rejects 39 of 1,423 frames and keeps
1,384 dense views. The older MAtCha audit that rejected 53 frames used
`masks_with_tree.pkl`; reproducing 1,370 views would also reintroduce tree-mask
decisions, contrary to the current no-tree-mask ablation.

## Historical heldout results (obsolete for the active all-train policy)

The figures below belong to the retired `train_opt`/trajectory-heldout protocol.
They are retained only for provenance and must not be used to judge the current
1,487-image all-train reconstruction or post-reconstruction causal repair.

| Variant | Iterations | Raw PSNR | Raw SSIM | Static-valid PSNR |
| --- | ---: | ---: | ---: | ---: |
| MAtCha-like G4 initialization | 7k | 12.604 | - | 15.255 |
| Legacy G4 Stage 1 | 7k | 9.593 | - | 10.021 |
| Scene-aligned G4, all pseudo pixels | 7k | 9.544 | 0.456 | 9.906 |
| Protected G4 plane-only | 7k | 12.892 | 0.691 | 15.429 |
| Protected G4 inpaint-only init/geometry pseudo | 7k | 12.755 | 0.679 | 15.250 |
| Protected G4 plane-only | 30k | 12.980 | 0.711 | 14.950 |
| Latest MAtCha n24/30k | 30k | 14.368 | 0.767 | 15.864 |
| MAtCha retained_v2 | 30k | 15.825 | 0.796 | 17.878 |
| Latest MAtCha n24/30k, normalized eval cameras | 30k | 14.450 | 0.774 | 15.964 |
| MAtCha retained_v2, normalized eval cameras | 30k | 15.921 | 0.804 | 18.010 |

The normalized-camera rows do not retrain either model. One StMarysChurch
heldout pose (`seq1/frame00109`) had quaternion norm `1.00747`; canonicalizing
that pose raises its retained_v2 PSNR from `11.24` to `17.42` and explains almost
all of the `+0.095 dB` mean change. The n24 training set also contains four
poses with norm error above `1e-3`, including one selected chart, so the next
training result must use the normalized QC dataset before judging chart policy.

The original G4 pseudo-view path failed because it injected entire generated
frames as geometry and supervision, growing the model from about 130k to 372k
surfels while introducing scene-scale blur. Scene-aligned cameras fixed the
rotated and duplicated proposals but did not fix this contamination. Limiting
pseudo initialization and geometry supervision to disoccluded pixels, while
using the preserved visible region as low-weight RGB self-distillation, removed
the catastrophic failure but still regressed the plane-only 7k result by 0.136 dB
raw PSNR.

The protected plane-only 30k result remains 1.388 dB below the latest MAtCha n24
run and wins on only 14 of 64 views. G4Splat is therefore not retained as a
positive Cambridge NVS improvement, and Stages 2/3 were not run after the quality
gate failed. It may still be useful as a mesh/geometry experiment.

To reproduce the safer inpaint-only ablation explicitly, add:

```bash
--pseudo-initialization-mode inpaint_only \
--pseudo-geometry-mask-mode inpaint_only \
--pseudo-rgb-weight 0.01 \
--pseudo-geometry-weight 0.05 \
--pseudo-geometry-final-weight 0.005
```
