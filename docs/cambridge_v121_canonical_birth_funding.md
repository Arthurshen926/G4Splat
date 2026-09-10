# Canonical birth funding repair and controlled pilot

## Confirmed defect

The scene-canonical accumulator drains proposals after two independent cameras in one canonical sequence. `_initialize_static_ray_birth_mass_handoff` nevertheless required support from two sequences for its bounded additive existence budget. Legitimate canonical proposals with inadequate donors were therefore starved at birth.

The v120 3k topology trace contains 13 funding events and 4,209 candidates, with zero additive-funded children and zero additive mass. This establishes an inactive fallback; it does not prove that every newborn was transparent (donor transfers and later training can provide mass).

## Changes

- The funding function now receives the selected canonical sequence explicitly. It accepts at least two distinct support cameras only when every retained nonnegative camera belongs to that sequence, and the sequence count is one. Unknown and cross-sequence cameras fail closed.
- Legacy noncanonical operation retains the two-sequence requirement. Repeated camera IDs no longer count as independent support.
- The original event mass cap, donor conservation, unresolved lifecycle status, and rigid-opacity freeze are unchanged. Funding is not global rendering verification or MoGe positive-gradient permission.
- Same-row positive optical demand is no longer discarded merely because an ownerless birth has no replacement group. Such genuine dual-evidence rows still need localization; this patch does not claim a universally reachable repair exit.

Regression cases cover valid same-sequence support, duplicate cameras, unknown cameras, mixed sequences, legacy behavior, bounded total mass, and ownerless same-row demand.

## Completed v120 fixed19 comparison

Mean per-view float protocol PSNR over the 16 authoritative seq2 views; three seq1 views are diagnostic only.

| Region | v119 3k | v120 3k |
|---|---:|---:|
| Tree | 11.754581 | 11.750984 |
| Inside boundary | 11.950594 | 11.950010 |
| Hard rigid interface | 14.527778 | 14.527031 |
| Non-tree rigid | 16.490771 | 16.491261 |

There is no material reconstruction improvement from the v120 birth allocation changes alone. Metrics are reconstruction-fit diagnostics, not held-out localization accuracy.

## New clean pilot

Directory: `/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1/pilot_v121_canonical_birth_funding_fixed_rigid_3k`.

The detached supervisor launches a clean 3k run on GPU 1 using the v120 command, same seed and initialization, 30k schedule horizon, zero surface retirement budget, and no resume. The changed ownership rule only operates later than this pilot, so the early comparison targets canonical birth funding. No existing checkpoints were overwritten.

Acceptance: additive-funded children and realized mass must become nonzero on eligible proposals without violating the event cap; then compare thin-hit zero coverage, target attainment and fixed19 canopy/rigid metrics. More funded points alone is not success. Later shared-owner cleanup remains outside a 3k validation window.
