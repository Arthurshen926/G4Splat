# Independent projected optical-depth kernel control

User explicitly approved this isolated experiment on 2026-09-10. No production
renderer, building geometry, or installed extension is replaced. Physical GPU0
is excluded. Source lives in `experimental/canopy_optical_depth_rasterizer` and
builds the distinct `canopy_optical_depth_rasterization._C` package in place.

The volume-only pixel alpha becomes `-expm1(-tau*G)`, with
`tau=-log1p(-clamp(peak,0,1-1e-6))`. The existing 0.99 pixel cap, minimum alpha,
projected support, ordering and surface kernels remain unchanged. This is a
projected optical-depth model, not a full three-dimensional ray integral.
Both RGB backward and intrinsic volume interval-query backward are changed.
The volume derivatives are zero outside peak clamps and at the alpha cap;
surface derivative behavior is intentionally unchanged.

Motivation: same-position, same-covariance optical-depth splitting preserves
the centre alpha but not off-centre alpha under the old `peak*G` kernel. For
peak=0.9 and G=exp(-0.5), original alpha is 0.545878, two half-tau children
compose to 0.657458, while the proposed kernel gives 0.752560. This proves an
invariant mismatch with tau-based factorization, not the cause of canopy
failure. The current fixed-topology diagnostic does not split during fitting.

Native tests cover splitting, opacity and geometric finite differences,
intrinsic interval ownership, and unchanged surface output. Scene read-only
48-view A/B is required before considering training. A darker canopy alone
does not demonstrate correct reconstruction; building preservation and
leakage-region RGB error must be inspected together.

## Concurrent loss control

v363 absolute versus v364 relative radiance, 800 training images, fixed paired
seeds and bounded persistent source positions: relative minus absolute mean
tree PSNR -0.013568 dB, interior -0.011883, boundary -0.018724, rigid -0.001796,
hard-building interface -0.006641. Absolute versus original source is +0.667346
tree but -0.013089 rigid and -0.006623 hard. Relative is not an improvement and
is not promoted. These are the existing 48 diagnostic views, not blind heldout
data; neither result demonstrates that background see-through is solved.

## Completed isolated kernel validation

Independent build succeeded. CUDA tests: 3 passed, including reused native
position/geometry, interval-query and foreground-only floor finite differences.
Pure-surface RGB is bitwise equal; surface opacity gradients agree within the
stated floating-point tolerance. CPU project suite: 1375 passed, 29 skipped.

v365 (original) and v368 (experimental) audited the identical v363 step-800
checkpoint over all 48 views. v366 stopped on the original cross-worktree
library guard; v367 stopped on Python 3.9 annotation evaluation in the isolated
adapter. Neither failure was a training interruption or OOM. The final adapter
clones only render_hybrid into private globals, explicitly imports the new
package, and retains a path guard against that package's actual directory.
Original Python globals, CUDA binary and source files remain unchanged.
Both successful audits returned zero; experimental_backend.json records both
binary hashes and the adapter-source hash. Do not compare their source-relative
deltas: the source is also rendered by each respective kernel. Compare matched
updated metrics directly.

New minus old mean canopy PSNR: +0.013086 dB (30/48 positive, worst view 694
-0.129785). Mean rigid -0.006710, interface -0.024760, away-from-interface
-0.003376 dB. Leakage ROI 660: 13.269372 -> 13.443674 dB, surface contribution
0.371245 -> 0.361194, irreducible nonnegative-color MSE 0.033696 -> 0.032773.
ROI 657: +0.017969 dB; ROI 690: +0.191696 dB, but its old dense-interior ROI
regresses about 0.097084 dB. This is a small, mixed read-only effect, not a cure.
Building parameters staying frozen does not prevent changed foliage from
occluding building pixels. No production adoption; paired training approval
has been requested separately.
