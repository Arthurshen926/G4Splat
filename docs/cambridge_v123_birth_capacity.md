# Fine-scale birth capacity instead of permanent coarse-cell exclusion

## Concrete failure

The visual-hull birth grid used 0.30 scene units while newborn isotropic sigma is 0.04. One accepted birth permanently tombstoned the entire 0.30 cell. Two independently supported points at (0.06, 0.06, 1.06) and (0.24, 0.24, 1.06) are about 0.255 apart, but occupy the same coarse cell. The second point receives zero births even with two new calibrated ray witnesses. Its screen-plane separation from the first is over six newborn sigmas for a frontal camera; increasing the first point's scalar opacity cannot cover the second point at the renderer's alpha threshold.

This is a capacity/spacing mismatch, not proof that every real uncovered pixel has this cause. Lowering the grid spacing alone also produces near-identical births on opposite sides of fine voxel boundaries, so it requires a geometric deduplication repair.

## Implementation

- Trainer defaults: midpoint grid 0.04, interval grid 0.08 (twice newborn sigma).
- Explicit minimum emitted-centre separation 0.04, checked through neighbouring spatial hash cells across both source grids and successive drain calls.
- Boundary duplicates are tombstoned without allocating extra optical mass. Other independently supported fine cells remain eligible.
- Separation and emitted centres survive capture/restore. Changing spacing/separation on resume is rejected; this experiment requires a clean accumulator.
- Legacy accumulator construction retains separation=0 for compatibility. The trainer explicitly opts into the new rule.
- The trainer now forwards the configured MoGe proposal budget to the accumulator. Previously its implicit 256 cap silently truncated proposals even when the caller requested a larger budget. Default proposal budget remains unchanged.

The synthetic regression reproduces the coarse-grid failure, permits the two distinct fine-grid points, rejects near-duplicate boundary points, preserves rejection across resume, and still permits a third distinct point. Targeted foliage/trainer tests after the spacing repair: 355 passed.

Longitudinal slab agreement and independent camera requirements remain intact. This still does not replace transverse pixel-cone validation, post-birth visibility verification, or adequate per-row optical training. No production model has been declared improved from birth counts alone.

## Read-only real-scene replay

The calibration script's `--birth-audit` mode feeds identical uncovered MoGe proposals to coarse and fine accumulators, excludes the 16 calibration holdout views, and never modifies teacher parameters. Its output centres are untrained candidates, not verified map elements. Use this replay to check whether finer spacing actually increases useful capacity before allocating a full training run.

## Superseding first-hit correction (v124)

The initial replay's coarse 14,005 versus fine 17,749 candidates is **not evidence of useful density improvement**. A subsequent regression showed that two first-hit rays with uncertain depth could fund eight separate fine-grid births along the same interval. Depth uncertainty had been incorrectly converted into material thickness.

MoGe and posterior first-hit proposal producers now explicitly identify single-first-hit evidence. Each calibrated camera/ray witness can fund only one initial material hypothesis; alternative depth cells cannot reuse spent support. Candidate priority favours consensus posterior depth, geometric duplicates associate with the existing point, and distinct ray observations remain eligible. The v6 accumulator persists witness consumption, per-camera sequence association, and geometric deduplication. This is an allocation constraint, not permission to increase opacity or bypass real mixed-render visibility verification.

Both v123 prefixes were deliberately stopped via their supervisors and saved at iterations 577 and 509 before their active foliage phase. They did not crash and are not completed 3k results. Clean v124 prefixes replace them. Full regression suite after the first-hit correction: 1,074 passed, 15 warnings.

Remaining limitations: no full transverse pixel-cone triangulation; consumed allocations are not yet reconciled against later pruning or motion. Tests and candidate counts do not establish that canopy background leakage has been solved.
