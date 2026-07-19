# Cambridge Stage-1 failure audit

This note audits the experimental StMarysChurch G4Splat Stage-1 result. It does
not change the MAtCha main result or any training behavior.

## Verdict

The failure is not a single semantic-mask boundary problem. The dominant chain
is:

1. Some proposed pseudo cameras render a large, smooth, incorrect splat as fully
   observed.
2. See3D and Depth Anything operate on that poor proposal without a pseudo-view
   quality or semantic gate.
3. The plane extractor accepts giant low-detail regions as planes.
4. Global plane fusion merges local planes using point-index overlap only. It has
   no normal-angle or plane-offset compatibility test, so unrelated surfaces can
   be joined and propagated back to real chart depths.
5. The legacy path then initializes and supervises Gaussians from all pseudo
   pixels, turning local geometric mistakes into scene-scale smearing and repeated
   contours.

The protected Cambridge path prevents step 5, but pseudo planes still participate
in steps 1-4. It is therefore a `plane-only` G4Splat ablation, not a clean MAtCha
baseline.

## Heldout metrics

All rows use the same 64 StMarysChurch trajectory-heldout cameras.

| Variant | Raw PSNR | Raw SSIM | Dynamic-valid PSNR | Static-valid PSNR |
| --- | ---: | ---: | ---: | ---: |
| Legacy initial 7k | 12.6040 | 0.6511 | 13.4370 | 15.2549 |
| Legacy Stage 1 7k | 9.5932 | 0.3976 | 10.1475 | 10.0213 |
| Legacy Stage 2 7k | 9.6143 | 0.4113 | 10.1379 | 10.0293 |
| Legacy Stage 3 7k | 12.3574 | 0.6533 | 13.1424 | 14.7700 |
| Legacy final 30k | 12.7261 | 0.6879 | 13.6309 | 15.1633 |
| Scene-aligned, all pseudo 7k | 9.5436 | 0.4558 | 10.1763 | 9.9062 |
| Scene-aligned, protected plane-only 7k | 12.8915 | 0.6913 | 13.8046 | 15.4290 |
| Scene-aligned, protected inpaint-only 7k | 12.7552 | 0.6791 | 13.6577 | 15.2501 |
| Scene-aligned, protected plane-only 30k | 12.9796 | 0.7115 | 13.9689 | 14.9501 |

The approximately 3 dB Stage-1 collapse remains after fixing camera orientation.
It is caused by direct pseudo initialization/supervision, not primarily by the old
world-axis camera generator. Protected pseudo modes remove the collapse but do not
beat MAtCha.

## Pseudo-view evidence

Stage 1 selected 13 pseudo views. Two especially diagnostic views are:

- Local view 10 / global frame 34: visibility ratio 1.000, one plane covers
  0.981 of the image, plane confidence is 0.977, but the RGB is an almost uniform
  dark blur.
- Local view 11 / global frame 35: visibility ratio 1.000 and one plane covers
  0.930 of the image, again from a low-detail blurred render.

The legacy See3D output changes already-visible RGB pixels by 0.036-0.092 mean
absolute RGB error across the 13 views. The protected merge reduces that value to
zero by restoring the GS render in visible pixels. This fixes visible-region RGB
contamination, but it does not reject bad cameras or stop their planes entering
global fusion.

There is also a mask mismatch in depth generation. Novel-view RGB visibility is
the visibility-grid result intersected with `alpha > 0.99`, while depth alignment
uses only `alpha > 0.9`. For local view 0 the RGB support is 0.631 but the depth
fit uses 0.995 of pixels. Unsupported rendered depth therefore anchors the affine
alignment of Depth Anything. The alignment is ordinary least squares with no
outlier rejection or degenerate-fit guard.

The selector requests 20 views. If too few views satisfy the intended 5%-60%
unobserved interval, it first admits views below 5% and then arbitrary remaining
views. That is why fully visible views 10 and 11 are sent through an inpainting
and plane-generation stage despite having no coverage hole.

## Real-chart geometry evidence

For the audited chart subset, the pure `mask_0 & mask_1 & mask_2` keep ratio is
usually 0.85-0.97. After intersecting with the MASt3R chart-validity mask, it often
falls to 0.41-0.61. Thus the jagged red holes in the chart visualization are
mostly missing or rejected pointmap support, not a direct Mask2Former boundary.

The plane-refined depth has large local changes in several real charts:

| Chart | Image | Plane delta p95 | Pixels changed >25% |
| ---: | --- | ---: | ---: |
| 5 | `seq12__frame00142.png` | 0.398 | 0.093 |
| 9 | `seq14__frame00009.png` | 0.279 | 0.107 |
| 14 | `seq1__frame00118.png` | 0.326 | 0.166 |
| 17 | `seq2__frame00173.png` | 0.369 | 0.189 |
| 20 | `seq4__frame00016.png` | 0.225 | 0.034 |

Charts 14 and 17 agree closely with their aligned DAV2 priors before plane
refinement, yet change strongly after global plane fusion. Both are members of
global plane 62 together with pseudo frames 32 and 35. Frame 35 contributes a
single plane covering 0.930 of its image.

## Global-plane merge failure

The current Stage-1 map contains 162 global planes:

- 79 contain at least one pseudo local plane.
- 62 are pseudo-only.
- 17 mix real and pseudo local planes.
- 13 of the 17 mixed groups have a maximum undirected local-normal mismatch over
  15 degrees, 11 exceed 30 degrees, 5 exceed 45 degrees, and 2 exceed 60 degrees.
- The mixed-group median maximum mismatch is 38.7 degrees; the maximum is 88.3
  degrees.

Global plane 24 mixes 10 real and 5 pseudo local planes with a maximum normal
mismatch of 88.3 degrees. Global plane 0 reaches 87.4 degrees. These are not one
physical plane.

The implementation uses
`max(intersection / size_a, intersection / size_b) > 0.5`. A small local patch
can therefore merge into a much larger unrelated plane, after which greedy union
makes later transitive merges easier. There is no normal, plane offset, robust
depth, minimum mutual-overlap, or real-view support constraint.

## Semantic-mask attribution

- Real chart alignment, chart initialization, RGB loss, and geometry losses do
  use `mask_0 & mask_1 & mask_2`.
- The current mask pickle has three entries, so trees are not explicitly removed.
- The saved file named `semantic_keep_frame*.png` is the intersection of the
  semantic mask and chart pointmap validity. Its name is misleading if interpreted
  as a pure semantic visualization.
- Pseudo views have no Cambridge source image and receive no semantic mask before
  Depth Anything, SAM plane extraction, or global plane fusion. Generated sky,
  transient objects, and uncertain boundaries can therefore become planes.
- Nearest-neighbor hard masks and a white background can contribute thin halos at
  sky/building boundaries. They do not explain the large duplicated facade
  contours or scene-scale smearing.

## Experiment provenance issue

`StMarysChurch_g4_qc_n24_strictclean_full30k_v2` is not a clean controlled run.
Its initial 7k model was promoted from a screen using grid-4 initialization,
downweighted chart RGB, no white background, and no color correction. Later
resumes used grid-2 initialization flags, white background, sky alpha suppression,
and color correction while reusing those earlier artifacts. Its manifest records
the later command, not the complete artifact history.

The current Cambridge runner uses a configuration digest in new output names and
safe pseudo defaults, which prevents silent reuse across new runner configurations.
The old `_v2` directory should still be treated as diagnostic evidence rather
than a publishable ablation.

## Reusable diagnostics

Run:

```bash
conda run -n g4splat python scripts/analyze_g4_stage_artifacts.py \
  --mast3r-root <stage1>/mast3r_sfm \
  --guarded-mast3r-root <guarded>/mast3r_sfm \
  --mask-pickle <scene>/processed/masks.pkl \
  --mask-dataset-path <prepared-scene>/train_opt \
  --output-dir <diagnostic-output>
```

It writes per-pseudo-view and per-real-chart CSV files, a JSON report, a pseudo
RGB/depth/plane contact sheet, and a real chart semantic/pointmap/depth/plane
contact sheet.

## Fix order

1. Gate global plane merges by mutual overlap, undirected normal angle, plane
   offset or cross-view depth consistency, and clean real-view support.
2. Reject pseudo views before See3D/plane extraction using minimum hole ratio,
   supported-pixel ratio, sharpness, exposure, depth-fit conditioning, and giant
   plane ratio. Never backfill to the requested count with zero-hole views.
3. Use the same support mask for RGB protection and DAV2 depth alignment; make the
   depth fit robust and reject degenerate affine fits.
4. Segment or project semantics onto accepted pseudo views, and prevent sky or
   transient regions from becoming global planes.
5. Save source camera pair, interpolation fraction, quality scores, and acceptance
   reason in the pseudo camera archive.
6. Only after the geometry gates pass, test inpaint-only pseudo initialization at
   low weight. Full-frame pseudo initialization should remain disabled for
   Cambridge.
