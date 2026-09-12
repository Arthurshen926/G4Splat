# Geometry-aware detail reconstruction: association to native rendering

## Completed association comparisons

v525 quarter-pixel bilinear refinement (same epipolar gate and search radius):
45triangulated tracks vs54integer, only2tracks with extra observations vs3.
Not selected for the next experiment. Synthetic fractional translation test
passes, but this does not establish real-scene superiority.

v526 changes the support search scope: nearest12training cameras, without
requiring their depth to have passed the previous MoGe gate. Same845 source
candidates, integer matching, epipolar1.5pixel, roundtrip1pixel, same
triangulation1.5pixel threshold.4436forward,1560roundtrip,408three-view,
292triangulated. This supports a scope limitation in the earlier association
pipeline, not a new claimed renderer/production bug.

102tracks have>=4observations;435leave-one-ray-out checks,269pass1.5pixels,
17tracks pass all their leave-one-out checks. Average held-out reprojection
1.391414pixels vs1.938824 for original proposal positions on the same selected
observations. Matching selection used all observations: not blind validation.

`v526_source_details.png` shows17source RGB neighborhoods, not reconstruction.
They include visible railing structures, window/cornice boundaries, and some
other detail. They do not form a dense canopy reconstruction set.

## Actual native surface prefixes

v527 uses physicalGPU1, pure original surface rendering and exact contribution
queries at74training rays from the17tracks. Summed contribution RGB is checked
against original native renderer at3e-5tolerance.9rays have>0.9surface
contribution weight in front of the proposed geometry;6have foreground RGB
greater than observed RGB+.03. These are conditional pure-surface diagnostics,
not certification that the foreground surfaces are erroneous. No deletion.

## New geometry-aware suffix

Optional `refine_proposals` now exposes new-point XYZ offsets and independent
two-axis scale updates (bounded0.5--2x). Uses full-frame native RGB plus measured
source/support reprojection Huber loss, weight.001, and existing optical update.
Source surfaces/foliage remain fixed only for this isolated pair. Position step
LR.001, scale-code LR.001. The stronger canopy state is NOT loaded here: this
is the original common source plus292detail proposals. Normal orientations
remain prior measurements and fixed, no structural verification is invented.

Fixed fractional-coordinate indexing in the new training evidence-mask path:
explicit rounded integer mask indices; geometric reprojection retains floating
observations. State now saves proposal position and scale offsets. This is
experimental path compatibility, not a historical canopy root cause.

v528/v5298stepCUDA smokes completed normally onGPU1/2.288new rows moved,
max displacement.008295scene units; four-view metrics essentially unchanged.
9targeted CPU tests pass (matching, surface suffix, surface geometry, loss scope).

v530 geometry/v531 fixed304steps launched detached/supervised onGPU1/2.
Launcher binds successful smoke hashes; no hot editing of trainer/math helpers.
At76steps four-view tree/rigid/hard differences exactly0at reported precision.
Not a quality claim. Final48view evaluation pending at this entry.

Prepared full-frame local9x9RGB evaluation around actual matched observations
in8training views chosen by observation count. This is deliberately a training
fit diagnostic, separate from48regression views, not a new holdout metric.
Core canopy regrouping and topology repair remain incomplete.

## Completed paired training and local/native rendering

v530/v531 both completed304steps and48view evaluation, supervisor exit0.
Geometry-minus-fixed means: tree+0.0000007153dB,rigid+0.0000004172,
hard+0.0000002186. No effective gain. Geometry-minus-source similarly negligible.
Inspected660railings A/B: unchanged blur and ghosting, difference nearly black.
No adoption, no longer extension of this recipe.

v532 local full-frame-rendered9x9observation regions,8training views:
geometry-minus-fixed mean-0.0004320dB; geometry-minus-source+0.0054547.
Thus the lack of gain is not explained solely by dilution in full-scene metrics.
These are training-fit diagnostics and not blind geometric validation.

This closes a full association/position-scale refinement/rendering test, but
does not establish a successful structural repair loop. The292sparse surfels
still use fixed monocular normals and bounded scales, no connected rod/leaf-cluster
geometry, and no source-surface replacement. Global RGB drives proposal geometry;
the priority local RGB term remains optical-only in this controlled recipe.
That scope may limit local geometry fitting and should not be misreported as
full local joint optimization. It is not proven the unique cause of failure.

Also, source845proposals were themselves preselected by the older conditional
depth/RGB check. Expanding support cameras removes only the later view-search
restriction, NOT the earlier source-proposal filter. Future measured structure
construction must address source coverage, not claim that v526 removed all
dependence on old-depth acceptance. No claim of solved canopy or railing.
