# v122: preserve depth constraints through ray-birth consensus

## Reproduced defect

The accumulator treated two camera rays visiting the same 0.30 m voxel as sufficient geometric consensus and emitted their weighted representative mean. Disjoint intervals [3.01, 3.03] and [3.22, 3.24] on the same axis produced z=3.125 with two support cameras. That point is inside neither interval. This is a concrete implementation defect, not a conclusion about how often it occurs in the real scene.

## Repair

Each visual-hull cell now retains each independent camera's original normalized-ray depth slab. Before draining a cell, a batched projection seeks a centre inside all retained slabs and the original voxel. A final independent feasibility check is mandatory. Inconsistent or unconverged cells remain pending and receive no consumed-cell tombstone. Duplicate camera visits cannot inflate evidence weights. Constraints survive capture/restore; old pending cells without constraints cannot be promoted until their support cameras provide the missing witnesses again. Existing checkpoints and model geometry are not rewritten.

This is longitudinal depth consistency, **not full pixel-cone triangulation**. Transverse reprojection consistency, semantic correctness and independent post-birth verification remain necessary. A finite projection budget may conservatively defer feasible cells. This repair does not establish that canopy background leakage is solved.

## Validation

- Disjoint same-voxel intervals emit zero births, including after checkpoint round-trip.
- Overlapping intervals whose weighted mean is outside the overlap produce a centre inside the actual overlap.
- Existing cross-view birth, independent camera, consumed-cell and lifecycle tests remain passing.
- Complete suite: **1066 passed**, including native CUDA tests.
- Synthetic CPU feasibility solve: 2,000 two-camera cells in approximately 0.101 seconds, all returned centres at z=3.19 inside [3.19, 3.20]. This is not a full training performance benchmark.

## Completed v121 fixed19 evaluation

Mean float-image PSNR over 16 authoritative seq2 views (three seq1 views are diagnostic only):

| Region | v120 3k | v121 3k |
|---|---:|---:|
| Tree | 11.750984 | 11.758085 |
| Inside boundary | 11.950010 | 11.951616 |
| Hard rigid interface | 14.527031 | 14.524378 |
| Non-tree rigid | 16.491261 | 16.489842 |

The canopy gain is only 0.007102 dB. Funding correction alone has no substantial multi-view reconstruction benefit at 3k.

## Clean controlled pilot

`/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1/pilot_v122_interval_consistent_birth_fixed_rigid_3k`

Same command, seed, clean initialization, 30k schedule horizon and fixed-rigid policy as v121; stop at 3k, GPU 1, independent detached supervisor. Only the interval-consensus implementation changes. New per-event `depth_consensus_rejected_cells` records missing/inconsistent/unconverged depth evidence together; it must not be interpreted as a count of proven physically impossible cells.

Acceptance requires correct-depth coverage and fixed19 renders, not birth count alone. If the constraint rejects most candidates, inspect geometric calibration/interval compatibility and transverse ray alignment rather than indiscriminately widening the intervals. The post-18k shared-opacity state machine is outside this pilot's scope.
