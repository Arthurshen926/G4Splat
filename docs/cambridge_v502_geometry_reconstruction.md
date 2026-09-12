# Native detail and selective geometry reconstruction

Baseline: `85817d8719b4b06e2000b544772586ea47fce425`, worktree
`/root/G4Splat-v114`, branch `codex/moge3-static-canopy-v119`.
User attachment: `77ec8b5f-9fef-4f25-a60f-f3de924f2f6b/pasted-text.txt`,
570 lines, read completely. Experiments may use physical GPU1/2 only.

## Closed prior experiment

v501 final native400 comparison (fine minus coarse): tree -0.0188796520 dB,
tree interior -0.0189776421, tree boundary -0.0300128261,
rigid -0.0034580231, rigid/tree interface -0.0159854492.
Both v498/v499 and their native audits completed; no model promotion.
Denser/smaller coupled recipe is rejected, not a universal representation limit.

## Attachment disposition

- A1 native-to-cache loss and normal gate: implementing measured native sample
  retention, separate depth/normal validity, per-training-view audit.
- A2/A3 local thin structure: next conditional association of preserved native
  layer samples; only independently supported geometry may enter a local suffix.
  No global building unfreeze. Evidence packets alone are not reconstruction.
- B1-B4 canopy grouping/selectivity: requires actual training-view associations
  and surface-prefix responsibility. Existing LP, shape and rotation experiments
  already exist and are not proposed as new work. No fresh opacity/scale sweep.
- C alternating geometry/appearance and revalidation: pending evidence-backed
  local geometry. Source support IDs alone cannot certify moved primitives.
- Crop renderer: known contract mismatch, continue native full-frame rendering.
- Final cross-sequence/full database validation: pending candidate quality.

## Implementation

`outdoor/moge3_depth_layers.py`: area-bin reduction (including noninteger ratios), original native
sample depth and pixel coordinates, per-layer coverage, mixed/unresolved flags.
Mixed cells have no continuous base authority; >2 modes have no two-layer
proposal. A sample is never reassigned to the low-resolution cell-center ray.
This is prediction preservation, not geometry verification.

`build_moge3_chart_base.py --depth-evidence-policy layered` explicitly selects
new archive construction. Legacy remains available. Layered mode prevents mixed
cell MoGe replacement and keeps proposal metadata in raw MoGe camera-z units.
Normal absence no longer deletes otherwise valid continuous-cell depth under
this policy. Existing source archives and deployed geometry are not rewritten.

Fixed masked NaN propagation in scalar/normal area reduction: invalid values
are replaced before interpolation; multiplying NaN by zero was insufficient.
Actual scene prevalence still needs audit, so no root-cause claim.

Tests: 17 passed (`test_moge3_depth_layers.py`, `test_moge3_evidence.py`).
Tests cover native ray identity, separate surfaces, unresolved modes, empty
cells, identity reduction, NaN isolation and depth/normal decoupling.

v502: training-only CPU audit, all304 training cameras; 48 regression cameras
excluded. Exports bounded mixed-cell native evidence packets. Does not claim
Chart-to-initialization-to-handoff transfer or output rendering improvement yet.

`verify_moge_layer_proposals.py`: prepared association step. Selects source
views using training-only layer statistics, checks neighboring training native
depth/RGB and baseline, retains supporting pixel IDs; normals do not gate depth.
This remains conditional evidence, not independently triangulated geometry.
No renderer or scene tensors have been changed by these audits.

## Completed first evidence loop

v502 all304 train views completed. Rigid valid depth cells48,806,742,
mixed524,417 (1.0745%), exactly two separated modes65,970,
normal-only rejection38,628 (0.0791%). Tree valid8,737,061,
mixed120,470 (1.3788%), exactly two modes6,681,
normal-only rejection27,942 (0.3198%). Mixed uses within-cell >10% depth
span; two modes require adjacent sorted samples separated >10%, so mixed
and two-mode counts are not interchangeable. These are prediction statistics,
not actual thin-object error rates. Normal gating is not a large global
explanation in this sample; local importance remains possible.

v503 sampled12,288 near-layer observations from24 train views selected using
rigid two-mode counts.845 pass two other training-view depth3%, RGB L1<.12,
baseline>=1 degree checks; all845 have usable direct normals. These checks
are conditional and do not establish independent triangulation.

v504/v505: native-depth versus area-depth positions for the same845 proposals,
native full-frame mixed renderer, initialopacity.01, fixed position/size/normal,
only appended2D DC/opacity optimized,8steps. Both completed, frozen original
surface arrays unchanged. Four-view regression changes essentially zero
(roughly1e-5dB). This is executable-path verification, not quality improvement.
No longer training launched from this result. Investigating whether these
proposals mostly occupy existing surfaces or lie behind source contributors.

v506: native source-surface median-depth audit against all24 proposal cameras,
physicalGPU1, read-only. Pending at launch. Median depth is not full prefix
visibility and cannot certify a source surfel is wrong.

Visualization: `visual_v502_660_depth_transfer.png` in run root; supplemental
660 evidence visualization only, never used to construct training proposals.

v506 completed:845 samples,595 with original surfacealpha>.9. Among these,
166 original median depths are >10% nearer than proposals,223 >10% farther,
206 within10%. Median source/proposal depth ratio1.05974. Neither blanket
surface correctness nor blanket MoGe correctness follows from these data.

v507(native)/v508(area) launched supervised onGPU1/2; both completed304 updates,
final48view native evaluation pending at this entry. Same845 proposals and
fixed size/normal, only positions differ between arms. Initial/final optics
are trained identically; no geometry is certified by this comparison. Added
fixed660 railing rectangle metric and final native crop visualization script.
Original surface frozen audit runs on completion. No production adoption.

v509 real56Chart archive build completed at288x512, using the same four source
input hashes as original chart-base archive. Added noninteger area-bin handling
before this run; keeps original pixel IDs even when pooling bins overlap.
Archive `evidence_v509_layered_chart_base.npz`, SHA256
`fb010c327fdfe094066cbf371970a433ed8714fa27464262de71192f60639c79`.
Of8,257,536cells:162,586mixed; old MoGe admission on42,977mixed cells, new0.
New depth-without-normal562. Mixed two-mode proposals27,904 across all charts.
Scale changes0.827733515->0.832358810 after excluding mixed calibration cells.
This archive is not used by v507/v508 and has the original56Chart camera scope,
not the304canonical-only proposal scope. It has not replaced a production
archive or been used to claim a training-only generalization result.

## Final v507/v508 result and observation reassociation

Both304-step jobs and48-view native evaluation completed normally; original
surface arrays unchanged. Native minus area means: tree +0.0000007749dB,
rigid -0.0000033379, hard +0.0000002782. Native minus original source:
tree +0.0000032584, rigid +0.0000073910, hard +0.0000074307.
No effective image improvement, no adoption. Native crop A/B saved under
`compare_v507_v508_native_details304/660_railings_ab.png` and
`660_canopy_ab.png`; both show essentially unchanged reconstruction.

v510 RGB observation reassociation: local native5x5patch matching around
depth-projected proposals, distinct-minimum check, reverse match <=1pixel,
triangulation of >=3 fixed-camera rays, max native reprojection <=1.5pixels.
845 proposals ->1142 accepted forward pairs ->273 roundtrip pairs ->40
three-view proposals ->4 reprojection-consistent triangulations. This limited
local matcher does NOT prove all other proposals incorrect, and does not
establish a global data impossibility. It does show that the previous
depth/RGB agreement set does not automatically provide measured point matches.

v510 initial export kept prior support arrays with a warning and is NOT to be
used for geometry or supervision. v510b rebinds support view/pixel arrays and
source camera-z to the triangulated geometry; prior fields are separate.
No triangulated proposals have been inserted into a model or trained.

Remaining: improve measured correspondence coverage, associate observations
into stable local structures, targeted canopy geometry replacement/splitting,
surface-prefix visibility verification, and full database/sequence validation.
The attachment's complete A/B/C work packages are not yet finished. There is
no newly accepted reconstruction checkpoint from this phase.
