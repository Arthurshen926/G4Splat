# v127: intrinsic optical queries need independent opaque-depth bounds

## Confirmed implementation error

The production MoGe pre-hit / hit passes intentionally set `surface_gate=0` to measure intrinsic volume alpha. Previously their interval bounds depended only on MoGe depth, not the independently rendered rigid scene. Consequently:

- If a monocular depth lies behind a known opaque wall, an occluded leaf between wall and monocular free-space bound receives a **negative opacity derivative**. Native visibility does not justify deleting it.
- A monocular positive hit behind that wall can receive positive optical funding because the intrinsic pass removed the wall.
- A canopy-labeled pixel whose MoGe depth contradicts a nearer opaque wall can incorrectly authorize foreground free-space deletion.

The v126 foreground check protected **new witness eligibility**, not these existing optical-loss calls. Those are different paths. Fixing one did not fix the other.

## Repair

`clip_canopy_depth_queries_before_rigid` uses an independent surface-only native median-depth / alpha render. Only alpha >= .95, finite positive rigid depth is authoritative. Clearance is max(.03,.01*z).

- Pre-hit query stops at the nearer conservative bound / rigid-front depth.
- Hit upper bound stops before the same wall; hit centers behind the wall cannot fund positive optical mass.
- A dominant canopy observation with contradictory depth is marked unknown for both optical sources, not interpreted as empty foreground.
- Unknown / nonopaque rigid depth leaves valid original queries unchanged.
- The same bounded hit / validity is used for new birth proposals and independent camera candidates.
- Surface parameters receive no gradient through the independent observation. Production native RGB rendering is unchanged. The detached depth maps are reused by the subsequent witness pass.

CPU tests cover hidden-leaf free-space, canopy conflict, hit truncation and unknown-wall identity. A CUDA regression renders a leaf at z=3 behind an independent wall z=2.5 while the monocular free bound is z=3.5: the old negative opacity gradient is positive, the repaired hidden-leaf gradient is exactly zero, a real foreground leaf retains its derivative, and native RGB is bitwise unchanged. Native/query targeted tests: 21 passed. Current project `pytest tests`: 1107 passed (the optional upstream tetra extension issue remains separately documented).

## Controlled validation

`pilot_v127_rigid_bounded_optical_common2k_to3k`, supervisor 13783 / trainer 13784, GPU1, starts from the same immutable v125 corrected 2k compact checkpoint as v126. Stops 3k, retained 2.5k and 3k, same local detail LR multiplier 10 / independent witnesses / frozen rigid geometry and opacity / original initialization. It does not inherit a dense experimental initializer. Startup and actual loss execution must be confirmed; source archive is preserved.

Two additional opacity-only diagnostics use the identical fresh 2048-point-per-camera seed, same 187 optimization views excluding canonical48, same initial parameters and 400 steps:

- `diagnostic_v127_dense2048_ordered400`: only independent query bounding added to the v126 dense diagnostic.
- `diagnostic_v127_dense2048_ordered_boundary400`: additionally gives the rigid pixels adjacent to canopy their own input-model preservation mean (weight 3), so their small area is not diluted by the rest of the rigid image. This is a separate factor, not silently bundled into the query-only control or production trainer.

Actual diagnostic iteration 75 shows 1,500 pre-hit bounds clipped and 1,034 canopy depth conflicts made unknown in that view; these are observations of the repaired path, not quality acceptance.

The v126 2048 / 384 seed opacity diagnostics both finished 400 steps. Canonical48 tree / rigid / hard: 2048 = 14.27073 / 16.45263 / 14.23043; 384 = 13.92353 / 16.49431 / 14.32983. Input v125 2k = 10.93692 / 16.65432 / 14.80528. Both improve tree substantially but retain rigid / boundary regression and visible church transmission, so neither is accepted. Continued independent validation is required.

## Completed comparisons

Dense2048 query-bound-only step400: tree / rigid / adjacent rigid = 14.284 / 16.449 / 14.223. Adding separate boundary preservation: 14.002 / 16.525 / 14.485. Query correction is causally valid but on its own does not deliver a large image-quality gain.

Original-init v126 immutable 3k native canonical48: 11.20238 / 16.65863 / 14.72754. Its supervisor was intentionally requested to stop after retaining 3k; status `stopped_checkpointed`, validated step3001, zero automatic restarts. This was not a spontaneous crash. The matched v127 continuation reached2500 and saved its immutable checkpoint; 3k comparison remains pending.
