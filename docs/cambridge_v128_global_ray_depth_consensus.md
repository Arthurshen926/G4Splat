# v128 diagnostic: one-hit depth selection with complete canonical negatives

## Why another geometry experiment

Fresh dense leaves with independent witnesses improve canonical48 canopy by ~3 dB but still trade away rigid / hard-pixel quality. Opacity-only optimization plateaus, and bounded position Adam initially gives only modest additional gains. This does not establish a renderer failure or a 3DGS capacity limit.

`refine_canopy_ray_depth_consensus.py` tests an explicit alternative: resolve each source ray's uncertain depth using all eligible canonical observations **before** learning opacity. It generates nine alternative locations along the original exact-K ray within +/- .15 log depth, then selects at most one. It never creates nine physical layers.

Evidence comes from 304 canonical cameras excluding the prespecified 48 evaluation views, including cameras with no substantial canopy area:

- Positive candidate agreement: known valid canopy, unchanged thin MoGe interval, and in front of any independent opaque rigid surface.
- Negative geometry: projected center in a 7x7-eroded observed rigid or sky region, in front of any independent opaque rigid occluder. A point hidden behind a wall is unknown, not negative.
- Eligible choice: at least two independent depth-agreement cameras, at most one conservative negative view. Prefer zero negatives first, then more positive views, then smaller depth movement. This is a diagnostic robustness setting, not an accepted production rule.
- All old verification counts are cleared after relocation. Only two actual candidate-camera native contribution checks, with foreground ordering, can restore persistent rendering authority. No projected agreement is directly inserted as a render witness.

## Actual selection result

Run `diagnostic_v128_global_ray_depth_consensus_r2`:

- Proposed source rays: 361,082.
- Retained rays: 140,935.
- Changed depth among retained rays: 106,005.
- Zero conservative negative views: 109,733; one negative view: 31,202.
- Passed new real native two-camera verification: 140,346.
- Depth-bin counts (-.15 to +.15, nine bins): 11,203 / 10,648 / 11,450 / 16,568 / 34,930 / 16,058 / 12,072 / 13,553 / 14,453.

Original full checkpoints and the input diagnostic capture are unchanged. Output is explicitly diagnostic / non-resumable and contains no modified rigid parameters. Early attempts were stopped before results: initial script needed an explicit lifecycle birth iteration after clearing verification; the next launch had a mistyped evidence path. These diagnostic setup mistakes are not spontaneous interruptions of the formal training runs.

## Active isolated evaluations

- `diagnostic_v128_consensus_canopy187_400`: same 187 canopy-bearing optimization cameras and 400 steps as the preceding dense optical control; changes the geometry / newly collected witnesses.
- `diagnostic_v128_consensus_all304_1200`: also includes the remaining canonical cameras as legitimate negative / RGB observations. It does not require every training camera to have a tree. First 400 steps and subsequent snapshots must be reported separately because the camera schedule differs.

Both keep geometry / appearance fixed during opacity calibration, use the corrected independent rigid query bounds and input-model rigid / boundary preservation weights 3 / 3. Native canonical48 is excluded from these new optimization streams and receives no oracle rendering gates. Source archives and exact arguments accompany outputs.

No acceptance is inferred from the high witness count. Actual multi-view image quality, residual church transmission, and rigid / boundary deltas remain decisive.

## Measured updates (2026-09-09, 02:10 local)

Mean per-view PSNR on the same native canonical48, tree / rigid / adjacent rigid:

- Original v125 2k: 10.93692 / 16.65432 / 14.80528.
- Global consensus 2048, initial: 12.81467 / 16.62609 / 14.61208.
- Consensus 2048, canopy187, completed 400: 13.82797 / 16.64269 / 14.63000.
- Consensus 2048, all304, interim 800: 13.85059 / 16.64279 / 14.63278.
- Earlier dense2048 bounded position diagnostic, completed 1200: 14.30437 / 16.52693 / 14.42877.

Inspected native GT/render pairs 690 and 713 at consensus all304 step500: much less large-scale church visibility, but blurred foliage / incorrect silhouettes and remaining boundary damage. This is **not** successful detailed reconstruction; geometry/opacity fingerprint preservation alone does not establish unchanged rigid image quality.

Denser source geometry is now built and independently reverified using the identical consensus rule: 1,194,918 proposed rays, 553,777 retained, 416,368 relocated; 433,997 have zero conservative negative views and 119,780 have one. At most one point is retained per source ray. Two matched all304 experiments start from this same capture:

- `diagnostic_v128_consensus8192_opacity1200`: only opacity.
- `diagnostic_v129_consensus8192_local_refinement1200`: additionally DC leaf color LR .0025 and bounded leaf log-scale LR .003; positions, rotations, all rigid geometry/opacity/appearance remain frozen. Tangent scales stay within [.25,1.5] of their initial values, and thin depth scale cannot increase. Higher-order view-dependent SH stay bitwise unchanged. Full foliage snapshots, not opacity-only exports, are required to reproduce this arm.

These are isolated non-resumable diagnostics, not a claim that the production seed path has already been replaced. Canonical sequence selection in the diagnostic now comes from the checkpoint contract instead of hardcoded `seq2`; this is identical on this scene and fixes portability, not the observed canopy error.
