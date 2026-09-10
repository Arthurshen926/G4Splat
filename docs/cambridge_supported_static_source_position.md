# Authorized bounded static source-leaf position experiment

User explicitly approved per-leaf static position refinement on2026-09-10,
with supported persistent leaves only and protected buildings. This differs
from earlier shared deformation fields; no temporal state or depth truth.

Isolated diagnostic only, not production implementation or export. Default
source-position radius0 preserves the existing path. Positive radius requires
joint persistent optics and immutable-source background targets. Eligible rows
are exactly persistent_static_evidence_mask AND static_leaf_mask. Original
source XYZ parameters remain frozen; separate offsets are used in native
static rendering. Existing source scales/quaternions, all building geometry,
appearance and opacity, sky/reference state remain frozen. The existing
candidate XYZ/size/appearance recipe is unchanged in both arms.

For each supported leaf, delta = radius * initial_max_sigma * code /
sqrt(1+||code||^2). Thus the Euclidean displacement is strictly bounded by the
INITIAL largest principal sigma times radius; optimizing appearance or any
candidate size cannot expand this bound. CLI radius0..4, initial test radius2,
codeLR.01. No leaf IDs or geometry evidence are fabricated. Radius is a
controlled local search limit, not metric depth supervision.

Checkpoint stores separate indices, initial extents and codes. Read-only
auditor reconstructs eligibility/bounds from the immutable source, verifies
them exactly and rejects unauthorized state changes. Native source-position
wrapper preserves appearance, scale, rotation, role and visibility fields.
Optical-only gradient/component audit modes reject this new exception rather
than silently omitting its positions. Other native RGB/floor/capacity audits
use the same wrapper. Paired control must use the new exact same script/helper
hashes, identical seeds and cameras; only radius differs.

CPU identity, bound, ineligible-row, checkpoint-authority and static-view tests
pass. Native zero-position identity and finite-difference checks are running.
Need full scene8-image smoke and checkpoint replay before800-image paired runs.
No quality claim from passing invariants.48 evaluation cameras are excluded
from new fitting but historically trained source makes them NOT blind holdout.

Native24 tests passed, including exact zero-displacement raster identity and
finite-difference code gradient; project1366 passed,25 skipped. v350 full-scale
8-image smoke launchedGPU2, radius2, priority seeds, auxiliary0, source codeLR.01.
During seeding, BEFORE the position helper is imported after the seed loop,
checkpoint dtype verification was tightened (reject fractional/float IDs rather
than casting them). Corrected source archive:
v350_static_source_position_dtype_checked_sources.tar.gz. This supersedes the
earlier v350 smoke archive for replay; no training forward math changed.3 source
position CPU tests pass after this change. No production CUDA/model edits.

v350 completed returncode0, frozen checks pass.881782 supported rows,592522
actually displaced, max displacement.00884510 scene units, max bound fraction
.07704084 after8 images. All deltas finite; source-position gradient nonzero.
Source and candidate initial PSNR fields across all48 views match v344 exactly
(native unit test separately checks raw zero-offset RGB identity). This is a
causal smoke, not an efficacy result. v351 checkpoint replay/native floor audit
launchedGPU2, using verified saved indices/extents/code and unchanged rigid state.

v351 completed normally. Restored source-position statistics exactly match the
smoke.48 native rigid PSNR fields reproduce training evaluation within
3.8147e-6 dB (not a claim of full-image bitwise equality). v352 radius0 control
launchedGPU2,800 images/eval400, priority seeds and auxiliary0. v353 radius2
will follow onGPU1 after v347 completes; only radius and output may differ.
Archive:v352_v353_static_source_position800_sources.tar.gz. Both use fresh
source initialization, unchanged candidate budget and protected original
building RGB. No production checkpoint promotion.

Before v352 manifest publication, the source-position wrapper additionally
rejects non-null gradient-authority overrides (matching the enclosing native
candidate adapter's existing restriction). Otherwise adding a new offset after
a base-only gradient gate could silently bypass that gate in another caller.
Current training passes no such overrides, so its math is unchanged. This is a
new-interface guard, not evidence of a historical training defect. Corrected
pair archive:v352_v353_static_source_position_authority_checked_sources.tar.gz.
No further helper edits during paired runs; compare manifest hashes explicitly.

Authority predicate inspected directly: resolved non-dynamic VERIFIED rows,
proposal_kind NONE, verified_camera_count>=2, verified_sequence_count>=1, then
static_leaf_mask intersection. This grants experiment eligibility only; it does
NOT revalidate geometry after a displacement. No production export may silently
reuse old support metadata as proof of the moved geometry. v352 manifest helper
and main hashes explicitly match the authority-checked archived implementation.

After v347 completed returncode0, v353 radius2/800 launchedGPU1; v352 radius0
continuesGPU2. Source-position pair auxiliary0 in BOTH arms.400/800 checkpoints
must be compared under the same-code contract and followed by normal native
surface-floor/rigid-region audits. New displaced-source geometry is not exported.

Both manifests now explicitly compared: only output and source-position radius
differ. Main/helper SHA, candidate_count, seed_audit, ray_sampling_audit,
training_views and excluded_views match exactly. v353 has nonzero source-position
gradients at75 images. v352400 reproduces prior v344 mean tree within.0000145 dB;
this sanity check is not a replacement for the fresh same-code paired control.

At400 source motion adds tree+.053422 dB, interior+.060034, boundary+.035728
versus no-motion; rigid-.000105, hard+.012237. Absolute tree+.446200 and
hard+.013432 versus original.738 hard improves+.026628 versus control but still
-.575787 versus original;674 hard worsens another-.046089. No mean-only adoption.
Native660 visualization still visibly leaks. Report:compare_v352_v353_0400.json.
v356 early400 source-position native audit launchedGPU2 while v353 continues.

v356 completed.400 displacement bound fractions p10/50/90/99/max are
.0992/.2230/.3743/.4904/.7095; no row above.9. Displacement p50.01226 and
max.10307 scene units. No evidence that merely expanding the source-position
bound addresses current saturation.660 lower ROI PSNR12.909582, surface.390709,
volume.592461; v357 same400 no-motion audit launchedGPU2 to avoid comparing this
early point against an800-image control. v353 continues800, no adoption.

v353 completed800 returncode0, all frozen checks pass. Relative to v352:
tree+.069351 dB, interior+.075893, boundary+.050270, rigid-.000480,
hard+.005332. Absolute tree+.614484, rigid-.009155, hard+.001528.47/48 tree
views improve versus original, NOT blind heldout. Max source displacement
.102580 scene units, max bound fraction.777128, all881782 eligible rows moved.
Report:compare_v352_v353_0800.json. No range expansion justified by saturation.

v357/v356400 native paired660: PSNR12.790399->12.909582, surface.395124->.390709.
v355/v358800: PSNR13.027876->13.174141, surface.383090->.377049, volume.600803->
.607706; source-motion floor-MSE.034899715, still72.48% of current ROI MSE.
Both native audits completed0. Native660,674,632 images inspected: no solved
canopy claim. This is real small occlusion gain, not a finished production fix.
