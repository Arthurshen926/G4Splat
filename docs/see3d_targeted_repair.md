# Targeted See3D repair (experimental)

This branch keeps the Cambridge tree-aware n24 chart policy as the default and
adds an isolated, auditable repair path around G4Splat's See3D stage. Generated
views are not part of the retained MAtCha result unless they pass explicit
quality gates.

## Data flow

1. Render full RGB, depth, alpha, and visibility for candidate novel views.
2. Reject candidates with no edit area or an almost flat/blank known region.
3. Select at most six real chart images by target-camera position and direction.
4. Optionally augment the visibility edit mask with bounded components from
   `/root/STDLoc/valid_support_mask`.
5. Run See3D on full images and full-size known masks.
6. Restore every known pixel exactly from the conditioning render.
7. Export references, masks, raw predictions, composites, and reports together.

For clean-support artifact components, `render_projective_artifact_repair.py`
also creates `select-gs-projective-condition`. Pixels reprojected from real
training images become known See3D conditions. Only residual coverage holes are
left for generation.

## StMarysChurch findings

The retained-v2 wrong-geometry window component has a 1.835 percent repair ROI.
Clean train-view reprojection covers 99.43 percent of that ROI, leaving only
0.0104 percent of the full image unresolved. See3D is therefore unnecessary for
that component; projective real RGB is the stronger source.

On six tree-aware G4Splat novel views, the prefilter rejected one uniform
gray-green render whose visibility mask incorrectly marked the entire image as
known. Pose-aware references produced more coherent coverage-hole completions,
but generated windows and roof details remain unsupported. Raw See3D predictions
changed known pixels by 0.038-0.066 mean absolute RGB; protected compositing
reduced known-region change to exactly zero.

The current policy is consequently:

- keep the tree-aware chart selection in the main Cambridge runner;
- keep See3D repair experimental;
- use real projective RGB whenever clean geometric support exists;
- use See3D only for residual coverage holes;
- do not feed See3D depth back as trusted geometry without independent checks;
- do not run long refinement from these pseudo-views until a 7k ablation improves
  heldout metrics and worst-view visual quality.

Audit bundles are generated with `scripts/export_see3d_repair_bundle.py`.
