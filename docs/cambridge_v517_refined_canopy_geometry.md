# Surface refinement on the stronger canopy state

## Scope

Worktree `/root/G4Splat-v114`, branch `codex/moge3-static-canopy-v119`.
Physical GPU1/2 only. No production replacement or commit/push.
v480/800 is a stronger experimental canopy state, not an accepted final map.
The experiment tests whether v515's surface-geometry gains transfer to this
state; it does not optimize foliage further or claim joint convergence yet.

## Implementation

`scripts/load_refined_canopy.py` restores the exact oriented/selective spatial
candidate state, original persistent foliage optics, and static source position
offsets. Source checkpoint, masks, helper implementations, shape cohort and
camera splits are checked; unsupported kernel/parameterization recipes fail.
No candidate verification metadata is fabricated. Candidates remain jointly
depth-sorted by the native renderer with the saved source visibility gate.

`calibrate_native_detail_suffix.py --refined-canopy` uses that same canopy for
reference and updated surface renders. Source surface position/scale/rotation
are trainable only in the geometry arm. The fixed arm and geometry arm both
retain the845 normal-supported proposal optics, identical native RGB objective
and camera order. Local priority loss remains proposal-optics-only.
Trace now separates global RGB loss from local proposal loss.

## Verification completed

v517/v5188-step CUDA smokes completed. Geometry minus fixed four-view means:
tree+0.001876116dB,rigid+0.003433943,hard+0.001260996. Too small/short for a
quality claim. Reference source surface audit passed in both arms.

Initial reference metrics for660,738,674,657 (tree,rigid,hard) match
`audit_v495_native_v480_800/audit.json` updated metrics exactly, all12differences
zero. This validates reconstruction of the stronger reference via an existing
independent audit path, beyond merely loading tensors successfully.

10 targeted CPU tests passed for surface geometry, loss scope, proposal suffix,
depth layers and observation matching. These do not constitute a new full
suite or a dedicated unit-test suite for the new loader.

## Full paired experiment launched

v519:`diagnostic_v519_refined_geometry_304`,GPU1.
v520:`diagnostic_v520_refined_fixed_304`,GPU2.
304training cameras, same order;48regression cameras excluded from updates.
Native1920RGB throughout; checkpoints/four-view renders76/152/228,
final48-view evaluation304. These are development regressions, not blind tests.
Detached supervisor, max-safe-restarts0; explicit resume not implemented for
this small protocol. Launcher requires completed smokes and identical trainer,
loader and math-helper hashes. No hot editing while the pair is running.

Final metrics and visual results pending. The earlier+0.131dB tree result from
v515 must NOT be presented as this experiment's outcome. Outstanding issues:
canopy leakage, fine-structure blur, some rigid-boundary regressions; further
topology/geometry supervision ideas remain unimplemented, not silently done.
