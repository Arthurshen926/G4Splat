# Controlled follow-up after v306–v310

User requested continuing promising directions, not additional external baselines.
Physical GPU0 remains excluded. No new rendering loss or rigid edit is enabled.

## Completed evidence

All four preceding runs exited successfully with frozen-parameter audits true.
v306/v3071600 tree improvements versus the same source are +0.386430 and
+0.355799 dB. Broader depth stratification did not beat fixed layers. Their
738 building-side tree-boundary PSNR changes are -0.777257/-0.675732 dB;
neither is approved for production replacement.

v309/v310800 source-optics-only controls improve mean tree PSNR +0.300199/
+0.198913 dB. Reducing directional SH actual Adam steps to .05 improves the
711 regression from -0.493156 to -0.098577 dB, but loses average quality and
does not fix738 building-boundary regression (-0.311117/-0.315256 dB).
This is a tradeoff, not a completed fix.

## Bounded next experiment

- v311: diagnostic_v311_joint_slowsh_fixed1600, physical GPU1.
- v312: diagnostic_v312_joint_slowsh_spatial1600, physical GPU2.
- Same original render-only002000 source,304 training views,48 excluded
  diagnostic views,1024 rays per seed view, fixed ratios [.65,.8,1].
- Candidate gain1, initial opacity.001; source persistent leaf optics trainable
  with directional SH step multiplier.05; source geometry and all building
  parameters frozen. Background target remains immutable source RGB, guard3.
- ONLY pair difference: candidate position radius0 versus8 initial sigmas.
-1600 iterations, evaluations/checkpoints every400. RGB feasibility auxiliary0.
- Archive: v311_v312_joint_conservative_spatial_sources.tar.gz, under the
  cambridge_moge3_static_canopy_v1 run root. Diagnostic artifacts only.

Hypothesis: spatially adjustable additions can recover detail while source
directional appearance updates remain conservative. This has not been proven;
do not compare to historical runs as a single-variable causal experiment.
Normal static renders must be inspected on dense660, sparse branches,711,
and building boundaries738/909. Mean tree PSNR alone is insufficient. Existing
48 views are development views, not an independent historical holdout.

No additional experiment family should be launched before reviewing this
pair's common checkpoint. Do not promote candidate rows to verified geometry
or replace the production map based solely on their RGB contribution.

## Completion and next controlled correction

v311/v312 both completed with returncode0 and frozen audits true. v312 tree
+0.593539 dB versus source (+0.079172 versus v311),47/48 improving views;
interior +0.670801, boundary +0.401254. Rigid -0.014975 and building-side
boundary -0.011713 mean.738 boundary -0.760069;909 boundary -0.232584.
711 tree -0.105709.660 full tree +0.561853, but normal rendered lower crown
still visibly leaks background. No production adoption.

User requested continuous follow-through, not ending the task at launch.
v313/v314 compare the v312 joint spatial recipe over800 steps with one
difference: independent training rigid-side boundary RGB preservation weight
0/1. Both retain global rigid guard3, immutable source RGB target, source SH
step .05, fixed depth candidates and all source geometry frozen. GPU1/2 only.
The new helper normalizes by training boundary pixel count, rather than the
whole rigid region. This tests local-error dilution; it is not yet a proven
cause or effective fix. No tree opacity target, inference gate, or depth label.
RGB-feasibility auxiliary stays0. Archive v313_v314_boundary_rgb_sources.tar.gz.

v315 read-only native building-radiance floor audit of v3121600 runs onGPU1,
separating persistent background leakage from appearance improvement. Archive
v315_building_floor_sources.tar.gz. No new baseline runs or source-model edits.
CPU suite after boundary helper:1331 passed,22 CUDA-dependent skips,15 warnings.

v315 completed:660 lower ROI native PSNR12.204679 to12.991930; structural
contribution .424097 to.383354, volume .553912 to.601659. Thus some actual
occlusion improves, not only color. Holding geometry/opacity fixed, surface-only
radiance still imposes73.1615% of current RGB error as a nonnegative-color floor;
64.4033% pixels have at least one channel below that floor. This is a local
fixed-state limit, NOT an impossibility claim for alternative reconstructions.
738 away-from-tree-interface building PSNR18.987404 to18.844830, so losses are
not exclusively within the5px boundary. Retain all-building guard and inspect
away-region metrics as well as the boundary. Native CUDA tests:22 passed,
including positive extinction-correction gradient for foreground occluders,
zero for hidden leaves, and a finite-difference check of the boundary loss.
