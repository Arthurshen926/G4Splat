# v147: production color-target consistency and spatial gradient ownership

## Confirmed production target error

The static volume-isolated branch unpremultiplied its volume-only image and
then fitted that intrinsic leaf color to the observed RGB. These are different
physical quantities: C=(1-A)B+A*F. For F=.2, B=.8, A=.5, the correct image is
C=.5. The old branch still demanded that F increase toward .5, making the
native image too bright. A coarse tree-object mask also includes real gaps,
so this is not restricted to a negligible boundary case.

The production branch now keeps all deployment-visible static occluders and
the frozen surface in its forward render. It compares native composite RGB
to observed RGB. Existing detail geometry/appearance permissions remain;
opacity, surface and sky gradients remain disabled for this branch. Detached
native volume contribution defines appearance support. Hidden leaf colors
cannot learn through an occluder omitted from this auxiliary render.

The training contract string and color-quantity metadata change explicitly.
This is not an exact same-implementation resume into historical checkpoints.
The old unpremultiplication helper remains only for legacy diagnostic controls;
the production branch no longer calls it. Default branch cadence/weight are
every4 / .75, so the objective is not nominally a zero-weight dead path.

## Real-scene self-consistency test (not a reconstruction-quality metric)

`audit_v147_stable_native_color_self_consistency` uses the current native image
as an exactly self-consistent synthetic target in5 real views. It restores all
temporary updates and fingerprints geometry, shape, color and opacity unchanged.

The old color objective has losses .11037, .11253, .06169, .06674, .14662 on
657,660,690,713,768 respectively despite exact native image agreement. A virtual
DC Adam update worsens native agreement. The corrected objective has exact
zero loss AND exact zero feature gradient in every view. This proves the target
coupling error; it does not claim the original photographs are reconstructed
perfectly or quantify its eventual training-quality gain.

## Two further implementation corrections

1. Covariance-form SSIM left ~1e-10 numerical feature gradients at exact RGB
   equality. Adam eps1e-15 can normalize tiny cancellation noise into a real
   update. The equivalent difference form uses squared mean difference and
   variance of the difference, whose gradients are exactly zero at equality.
   A regression test includes an actual Adam step; another checks equivalence
   to the original SSIM objective in float64 away from equality.
2. Oriented multiscale Sobel loss masked only stencil centers, and average-
   pooled masks at scale2. This allowed a canopy loss to write pixel gradients
   into neighboring rigid/unknown pixels. It now requires the full downsampling
   and Sobel footprint to be owned. Empty/flat support returns differentiable
   zero. Tests verify exact zero gradients outside the authorized mask.

## Validation in progress

`diagnostic_v147_native_color_target400` and
`diagnostic_v147_legacy_intrinsic_color_target400` start from identical
free-XYZ1200 foliage. Only leaf DC color updates; geometry, opacity, rigid state
and sky stay frozen. Both use the same production photo objective form and
the same304 canonical training cameras, excluding the prespecified48 evaluation
views. Only the supervised quantity/occluders differ. This is a component causal
comparison, not a full production schedule rerun.

Earlier v144 extinction-only controls finish15.22237 versus15.22672 tree PSNR
(darkening adds only .00435dB); this extra conditional bound is not accepted as
a meaningful quality improvement. Source-wall sweep and foreground-contrast
initialization also have not demonstrated a breakthrough. v145 annealing and
v146 bounded orientation controls are still pending.

No claim that canopy background transmission is fully solved is justified yet.

## Completed v147 paired result

Both400-step color-only arms completed without OOM. Native-composite supervision
finishes15.047131 tree /16.601053 rigid, versus legacy intrinsic15.023090 /
16.606169, from identical15.009652 /16.608036. The +.024041dB difference is small:
the target error is causal and worth fixing, but this is not a canopy solution.

The expanded CUDA finite-difference regression covers all XYZ, scale, quaternion,
RGB and opacity coordinates, not only screen-x. It passes, including depth and
orientation. No missing global XYZ derivative was established by this check.

## v148 convergence and v149 clean initialization integration

v148 compares2400 additional native-image optimization steps with DC-only and
degree3 SH, identical geometry/opacity updates and LR annealing. The latter uses
the standard .05 rest/DC actual-Adam-step ratio. Rigid/sky remain frozen. Both
start from the same v1321200 capture; these are diagnostic continuations, not
an exact production resume. No final result yet.

A diagnostic calibration-order bug is fixed: applying the per-view scale after
constructing depth bounds also multiplied the absolute thin-layer thickness cap.
Depth is now calibrated BEFORE constructing the metric-capped bounds. The new
canopy field is separate from the global rigid/Chart depth target. Old running
controls retain their launch code; their existing results are not relabeled.

The new typed fresh canonical initializer preserves measured-single/native
multiview witness metadata instead of entering legacy dynamic-observation fusion.
It rejects trained/proposal captures and camera authority from excluded views.
Production camera streams exclude the same48 cameras without renumbering them;
raw historical training was still not blind to these cameras. Production loads
the seed-bound per-image canopy calibration, leaving rigid/Chart gauge untouched.
These interface changes are implemented, but a full production train has NOT
yet validated this new initialization.

`initialization_v149_calibrated_canonical8192` was stopped by its own preflight:
raw checkpoint geometry is bitwise equal to the native rigid PLY, but the baked
deployment atlas exp/log round-trip changes765 log-scale rows by at most
2.3841858e-7. XYZ/quaternion/opacity are identical. This is not real geometry
motion or an OOM. The separate `_v2` rebuild verifies raw geometry bitwise,
permits only few-ULP log-scale export differences, then explicitly restores the
exact PLY log scales BEFORE seeding. Historical files are untouched. Real atlas
position/rotation/scale changes still reject the handoff.

Tests after initial v149 interface integration:1181 passed /15 warnings. The
subsequent export-alignment adjustment needs its separate checks. Native image
acceptance and a completed full production run remain outstanding.

## v149 integration result and v151 lifecycle correction

The `_v2` initializer completes with1180169 measured leaves,826005 native
multiview-verified leaves. It exports a typed `production_initialization` with
source-bound per-view depth calibration and excluded camera IDs.

The first formal v149 launch inherited the envelope-first legacy lifecycle:
the MAIN RGB branch hid local leaves until1500 even though this initializer
contains no envelopes. Its native auxiliary color branch could train leaves
earlier, so this was not complete absence of all foliage learning. This
training/lifecycle mismatch is corrected with the explicit new
`static_measured_handoff_quality` profile: static_foliage fromiteration1,
same13500/19500/25500/30000 later endpoints, validated fresh measured leaves
only, and rigid geometry must use frozen/appearance_only ownership.

The old v149 run was intentionally SIGTERM-stopped at861 after an atomic
checkpoint; it was not an OOM or spontaneous interruption. The500 checkpoint
confirms bitwise unchanged rigid XYZ/scale/rotation/opacity and zero excluded
indices in ALL ten30000-entry camera schedules. The separate clean
`formal_v151_fresh_measured_lifecycle_3k` is running, with a fixed30k horizon
and500/1500/3000 retained checkpoints. It is not a resume of v149.

v150 tests a further diagnosis, not an adopted permission change: formal
defaults restrict appearance to registered support/witness camera IDs, whereas
the earlier diagnostic allows persistent leaves to learn color from other
canonical native-visible views. Both new800-step arms use native RGB, fixed
geometry/opacity and identical v145800 starting foliage. Atstep125 exact scope
permits2546 rows versus849896 for persistent scope; these are owner-mask counts,
NOT actual nonzero gradient counts or proof of a quality gain. Geometry and
opacity authority are not expanded.

Latest full tests after the new lifecycle profile:1182 passed /15 warnings.

## Dense-crown decomposition (posthoc diagnostic regions)

`docs/canopy_dense_regions_v149.json` defines five GT-inspected dense interiors,
not training masks or prespecified blind acceptance regions. Native v145800
leaf alpha means for657/660/690/713/768 are .96786/.92514/.99860/.96552/.97586.
Actual rigid alpha means are .03057/.07419/.000930/.03348/.02400. Thus some dense
interiors already have negligible building contribution, while others retain
local leakage. Their blurred/gray appearance cannot all be attributed to
background transmission. For690, dense-region PSNR improves23.284→25.603 with
annealing, but gradient strength remains only .385 ofGT; global canonical48
PSNR improvement is only .192dB. None of these facts establishes full success.
