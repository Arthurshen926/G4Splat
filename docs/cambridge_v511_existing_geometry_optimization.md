# Existing geometry optimization

User explicitly authorized moving beyond permanently frozen geometry. Worktree
`/root/G4Splat-v114`, branch `codex/moge3-static-canopy-v119`; GPU1/2 only.

## Implementation

`SurfaceGeometryDelta` exposes existing2D surface centers, tangent scales and
orientation as trainable parameters. All1,006,817surface rows are eligible;
no permanent building freeze is imposed in this experiment. Original tensors
remain an immutable reference for A/B and checkpoint reconstruction. Their
unchanged hash is NOT evidence that rendered building geometry stayed fixed.
Saved position/scale/rotation codes define the actual optimized geometry.

First trial: displacement norm bounded by0.5scene units, scales within0.5--2x,
orientation offsets bounded. Source appearance/opacity and foliage remain fixed
for this comparison, isolating surface geometry effects. The appended845
conditional proposals have trainable DC/opacity in both arms.

v511/5128steps: geometry enabled vsfixed, same native RGB and camera order.
Both completed.987,624surface rows moved; mean displacement0.0124022,max0.0289993.
Four-view means: tree+.0251062dB,rigid-.1912606,hard-.0089612 versus control.
Not adopted. Original reference tensors unchanged in both arms.

## Correcting objective scope before extending

The inherited high-weight local proposal loss originally acted on845new
surfaces only. With existing geometry enabled, letting that same tiny-region
loss drive all original surfaces reweights whole-building geometry incorrectly.
`backward_detail_objective` now routes local loss only to proposal DC/opacity;
existing geometry receives the full-frame valid RGB objective. Object/distance
invalid pixels are excluded. This is an expansion-of-scope correction in the
new experimental trainer, not an established historical production root cause.

Reduced initial geometry code learning rate0.01->0.001 (scale/rotation half).
This and loss scope change form a new recipe, not an isolated causal ablation.
v513/5148step smokes started onGPU1/2 with both arms using the corrected
objective. No formal extension until CUDA completion and movement checks.

Tests: zero-delta identity, gradient reaches original geometry through suffix,
source reference receives no gradient, displacement bound, and local objective
cannot drive scene geometry.3 targeted tests passed.

v513/v514 completed normally. Four-view geometry-control: tree+.00273061dB,
rigid+.00404811,hard+.00348830. Too short/small for a quality conclusion, but
the first trial's large rigid regression is absent. Do not isolate the effect
of loss routing from learning-rate changes; both changed between recipes.

Started v515 geometry/v516 fixed304steps onGPU1/2, supervised and detached.
Midpoint checkpoints and four-view A/B at76,152,228; final48-view native audit.
All304 training cameras visited once;48regression cameras excluded from updates.
Only checkpoint/evaluation publication changed after the successful smokes;
launcher checks the geometry/suffix/loss math helper hashes against bothsmokes.
Model state is atomically published before corresponding evaluation metrics.
An explicit restore entry for this new experimental protocol is not yet
implemented; supervisor safe-restart count is0, so no invalid automatic resume.

## Intermediate paired results

At228steps (four regression views only), geometry minus fixed control:

| View | Tree PSNR | Rigid PSNR | Hard boundary PSNR |
| --- | ---: | ---: | ---: |
|660|+0.077407|+0.179733|+0.120943|
|738|+0.091125|+0.190472|+0.091295|
|674|+0.039305|+0.000355|-0.049960|
|657|+0.092256|+0.338915|+0.114017|

660 railing rectangle regresses0.045357dB. These are photometric regression
metrics, not verified geometric accuracy or proof of resolved canopy leakage.
At304steps, movement audit confirms all1,006,817original surface rows moved:
mean0.0117320scene units, maximum0.0789243;1,006,793rows changed scale/orientation.
Final48-view evaluation pending when this section was written. No promotion.
Expanded targeted CPU checks:10passed (geometry, loss routing, suffix,
depth layers, observation matching). This is not a full-system test run.

## Completed304-step /48-view native regression

Both supervisors report completed/exit0. Geometry minus fixed control:
tree+0.131394843dB (47/48 improve), rigid+0.268549979 (43/48),
hard boundary+0.150229394 (41/48). Geometry minus original source is
respectively+0.131397466,+0.268557250,+0.150236169: the845proposal-only
control remains effectively unchanged. These results use the older source
checkpoint specified in the manifest, NOT a comparison against the latest
best canopy-optimized model. Regression views are not blind holdouts.

660: tree+0.0769434,rigid+0.1818504,hard+0.1352196,railing rectangle-0.0382290.
674: rigid-0.0545406,hard-0.0633907. Worst rigid regression682:-0.1093464;
worst hard-boundary regression704:-0.1027575. No production promotion.

Viewed native660 canopy and railing A/B crops: geometry changes are visible,
but severe background contamination and thin-structure blur remain. The
photometric gain must not be described as resolved canopy occlusion or
verified geometric accuracy. Next useful comparison should combine existing
surface refinement with the stronger canopy state, maintaining a matched
control and separately auditing deteriorating thin structures; extending
this old-source result alone does not establish a new overall best model.

Artifacts under run root:
`compare_v515_v516_geometry304/comparison.json`,
`compare_v515_v516_geometry304/660_canopy_ab.png`,
`compare_v515_v516_geometry304/660_railings_ab.png`.
