# Cambridge v119 renderer and canopy ownership audit

## Scope

This audit isolates four possible causes of the dense-canopy/background-
building failure: the native mixed renderer, the retained rigid surface,
foliage coverage/optical thickness, and cross-role ownership/lifecycle.  The
rigid handoff is treated as a fixed input while canopy changes are evaluated.

## Renderer findings

- The production static path uses one native mixed CUDA pass with replacement
  policy `disabled`.  Python two/three-layer compositors are diagnostic only.
- Native 2D surfels enter the per-pixel event stream at their ray-intersection
  depth.  The event merge remains exact when more than one 256-item surface
  batch is interleaved with volume events; RGB, alpha, depth and gradients are
  checked against an explicit front-to-back reference.
- Depth-clipped volume queries use the same event stream and have forward and
  finite-difference backward tests.  A volume event behind the query bound
  contributes zero pre-hit optical depth.
- No implementation sorting/compositing defect has been reproduced in the
  current CUDA path.  A 3D Gaussian is nevertheless a point-depth splat in
  that event stream: its entire projected footprint uses its centre depth.
  This is a representation approximation, not evidence that a wide Gaussian
  physically lies on one side of a facade at every pixel.  Production avoids
  depending on that approximation by transferring broad envelope ownership
  to spatially local detail rather than promoting a Python layer-mean
  compositor.

## Rigid branch isolation

- The MoGe3 static production profile now uses an `appearance_only` mature
  handoff and creates zero rigid-completion seeds.  Surface xyz, tangent
  scale, rotation, opacity, Chart inverse depth and topology are immutable.
- Canopy RGB is backpropagated only through foliage parameters.  Structural
  RGB sees rigid/sky ownership, so a tree residual cannot move or thicken the
  facade.
- Static mixed contribution alpha is no longer used as a rigid/free volume
  penalty.  A foreground surface cannot hide an incorrect volume and thereby
  reduce its negative evidence.  Exact surface-zero, depth-clipped role
  queries are the only rigid-front optical authority.

## Ownership and lifecycle defects fixed

1. Smooth EWA tails were interpreted as physical conflict because any
   non-zero gradient created permanent debt.  Event-relative material source
   filtering now prevents remote numerical tails from reaching either Adam or
   persistent ownership state.
2. One global opacity was forced to choose between canopy-positive and
   rigid-negative pixels.  The state is now triaged into positive-only,
   negative-only, shared and unknown.  Shared scalar gradients and momentum
   are frozen until a local representation exists.
3. Shared rows were frozen again after the verified local mass handoff, so a
   successful detail split could not retire the broad envelope.  Post-handoff
   enforcement now preserves one-sided local retirement while rejecting any
   restoration/growth under live rigid debt.
4. Split children retained parent-wide positive/debt ledgers after independent
   verification.  Successful families now atomically clear those ledgers and
   reacquire evidence for each local footprint; failed families still restore
   the exact parent snapshot and conservative ledgers.
5. A steady localization cadence starved verification maintenance.  Existing
   transactions now settle/rollback before a colliding iteration may open a
   new one.
6. Debt-bearing detail was excluded from localization, making the exact row
   that needed more spatial bandwidth unsplittable.  Durable detail may now be
   camera-plane split, but only while its measured projected radius remains
   above the ordinary 2 px bandwidth target.
7. Persistent conflict debt began while ordinary topology was still changing,
   allowing a broad parent label to be inherited by unrelated children.  The
   current exact negative source remains active, but monotone debt begins only
   after the 18k topology boundary.
8. The material-evidence threshold was a trainer default rather than a runner
   identity.  It is now explicit in the resolved command, checkpoint contract
   and completed-result validation.

## Existing-result counterfactual

Removing the v118 envelope at deployment is not yet a valid fix.  Across the
16 authoritative fixed views, detail-only rendering loses about 0.93 dB tree
PSNR while improving rigid PSNR by only about 0.04 dB.  Therefore the envelope
must first transfer its optical responsibility to verified local detail; it
cannot simply be hidden.

## Validation contract

The next clean run is evaluated at 3k, 6k, 12k, 18k, 21k, 24k, 27k and 30k.
Structural invariants are strict (no off-owner gradient, behind-query alpha
zero, no shared-owner scalar change, family settlement, exact rigid handoff).
Image metrics are descriptive rather than brittle pass/fail gates.  The fixed
suite reports canopy core/interior/boundary, exact rigid-front optical alpha,
hard building pixels and surface-only rigid quality.  A detail-only deployment
counterfactual is reported but cannot become production until it preserves
canopy quality and reduces rigid-front leakage.
