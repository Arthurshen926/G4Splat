# Candidate optimization direction audit and controlled follow-up

v328 completed normally, read-only, on v324800. All48 rigid PSNR values match
the original checkpoint evaluation within1.91e-6 dB after CPU-first payload
loading. Candidate opacity remained bitwise unchanged during the304-view
gradient audit. This is partial native-render parity, not a new efficacy result.

Contribution-weighted net gradients request opacity increases for96.09% of
candidate weight in660 lower ROI,96.77% in657,91.39% in690. Saved Adam first
moment opposes that full-training gradient for24.48%,45.68%,27.07% respectively.
These percentages weight current candidate pixel contribution, not row counts,
and the saved moment is neither the next optimizer step nor proof of a bug.
Candidate geometry, colors and refined source optics are held fixed in this
partial-gradient diagnostic. Original immutable source RGB remains the target
outside tree pixels, exactly as in training.

660 lower ROI:PSNR12.204679->12.825957; surface contribution.424097->.394647,
volume.553912->.588321. Fixed-geometry/opacity surface-radiance floor accounts
for72.64% of current MSE. Weighted candidate size1.9256 times initialization,
91.74% of its contribution comes from size>1.8. Thus neither color-only fitting
nor indiscriminate size growth is established as a solution.

## Controlled gradient accumulation

Optional gradient_accumulation1..8, default1. `steps` continues to mean rendered
training images, not Adam updates. Four sequential views each backpropagate
loss/4 before one optimizer update; no four-view graph retention. Partial final
windows use their actual count. Evaluation boundaries must be full windows;
checkpoints record image count, optimizer count, and no pending accumulation.
Learning-rate horizon remains the image budget, and the same camera order,
losses, parameter bounds and immutable building target are retained.

This changes update frequency, so any result must acknowledge that equal image
budgets have fewer Adam steps. It does not manufacture an opacity target, use
evaluation ROI as training evidence, or enable the pending feasibility loss.
Four CPU tests confirm mean-gradient/Adam equivalence, batch1 behavior, partial
windows and invalid schedules. v329 eight-image native smoke launchedGPU1,
full557514 candidates, only two Adam updates. v326/v327 continue unchanged.
Archive:v329_gradient_accumulation_smoke_sources.tar.gz.

v329 completed returncode0, frozen audits all passed. Actual position, size,
candidate opacity and eligible source-optical updates occurred. Peak allocated
VRAM4.23GiB. Project suite1352 passed,23 skipped.

v330(single view/update,GPU1) and v331(four views/update,GPU2) now run a matched
1600-training-image budget with800/1600 evaluation. Optimizer updates are
1600 versus400 by design, with the same image-indexed learning-rate horizon.
Both use source-opacity LR.02, photometric depth seeds, bounded XYZ/size,
slow directional source SH and immutable building/sky RGB protection. No extra
auxiliary loss. Archive:v330_v331_gradient_accumulation1600_sources.tar.gz.
They share the GPUs with the ongoing v326/v327 long controls; no GPU0 usage.

Additional CPU-only v324800 state inspection: only0.12968% of all candidate
rows have any XYZ tanh coordinate above90% of its bound;0.000897% exceed99%.
Max-axis bound-fraction quantiles p10/.5/.9/.99 are.1973/.4009/.6411/.8144.
These are global row counts, NOT leakage-ROI contribution-weighted numbers.
They do not support a claim of widespread hard position-bound saturation and
do not justify simply increasing the position radius.

## v330/v331 first matched800-image evaluation

Four-view accumulation tree+.344628 versus single-view+.558239, a loss of
.213611 dB at equal image budget. Hard boundary changes+.011210 versus-.016913;
rigid-.003737 versus-.012567. This is not a demonstrated improvement: better
background preservation accompanies less canopy fitting and fourfold fewer
Adam updates. Both continue1600 images. Report:compare_v330_v331_0800.json.
Default single-view v330800 differs from old v326800 by+.000260 mean tree dB;
close but NOT bitwise identity. No claim of exact native training parity.

## Next isolated risk: color gradient ownership

Current background RGB preservation updates foliage colors as well as opacity
and geometry. This permits color camouflage of a wrong occluder in principle;
it is a design risk, not yet a proven explanation of the observed artifacts.
Added optional canopy_only_color_gradients (default false): one unchanged
native forward, canopy loss backpropagates to all trainable parameters, other
RGB losses backpropagate only to opacity/geometry. Frozen scene parameters are
excluded. No inference gating, new loss target, or changed forward image.
CPU tests confirm unchanged total opacity/geometry gradients, canopy-only color
gradient and compatibility with accumulation. Not enabled in v326/v327/v330/v331.

Native mixed-rasterizer gradient parity test passed, including reused forward
graph and frozen building. Entire native suite23 passed; CPU project suite
1355 passed,24 skipped. This verifies gradient routing, NOT image quality.
No full-scene color-ownership run launched yet: wait for an existing GPU slot
instead of stacking a third full-scene process on one24GB GPU.

Added optional CPU immutable-reference RGB LRU (reference_cache_size default0).
It stores only original frozen scene RGB, not updated source optics or candidate
renders, and preserves floating dtype/bits without compression. Keys are fixed
camera ID and resolution; this diagnostic does not optimize cameras. Stored
CPU tensors are never exposed for caller mutation. Capacity<=512, stats recorded
in final frozen audit. First enabled training reference is fetched twice and
checked bitwise. Tests cover disabled mode, eviction, no aliasing, no gradients
and native GPU roundtrip parity. Intended to avoid repeatedly rasterizing the
same immutable target; not enabled in the four ongoing runs.

Updated project tests1357 passed,24 skipped; native23 passed. After v327 exited
normally, v332 full557514-candidate8-image scene smoke launched on GPU2 with
canopy-only color gradients and reference cache384. Source-opacity LR.02,
accumulation1, all other candidate settings unchanged. Archive:
v332_color_ownership_cache_smoke_sources.tar.gz. No production replacement.

v332 completed returncode0, frozen audit passed; actual opacity/XYZ/size/source
optics updated. Reference cache8 entries,1 hit,8 misses,22,118,400 bytes. First
hit preserved RGB bits. v333 all-RGB-color control800 launchedGPU2 with cache384;
v334 canopy-only-color arm will use GPU1 once v330 exits. Both800 images,
eval400, accumulation1, same source-opacity.02/slowSH.05/photometric/XYZ+size
recipe. Only color-gradient ownership differs. Archive:
v333_v334_color_ownership800_sources.tar.gz.

v330/v331 both completed1600 images, returncode0, frozen checks passed. Final
tree+.731021/+.546517 (single/four-view); four-view loses.184504. Rigid
-.022690/-.009160, hard-.012648/-.000786.738 hard remains-.881289/-.791136;
674 rigid-.311564/-.136786.660 four-view image still clearly leaks the building.
Do not promote accumulation4: lower background damage is accompanied by weaker
canopy fitting and does not resolve the difficult region. Report:
compare_v330_v331_1600.json. Default remains1.

After v330 exited normally, v334 canopy-only-color800 launchedGPU1. v333 runs
GPU2. Both use reference cache384; only color-gradient ownership differs.

v333 completed800 returncode0, frozen audit passed. CPU reference cache304
entries,497 hits(including one validation lookup),304 misses,840,499,200 bytes.
At400, mean metrics differ from old un-cached v324 by~1e-5 dB, not exact training
bitwise parity. v333/v334 main hashes match; only non-output argument difference
is canopy_only_color_gradients false/true. v335 read-only48-view radiance-floor
and contribution audit launchedGPU2 on v333800. v334 continuesGPU1. Audit archive:
v335_v336_color_ownership_audit_sources.tar.gz.

v334 completed800 returncode0; paired final tree increment+.001962, interior
+.000080, boundary+.004151, rigid+.001222, hard+.005297 versus v333. Too small
to identify color camouflage as the primary bottleneck or claim a repair.
Report:compare_v333_v334_0800.json. v336 native floor/contribution audit launched
GPU1 on v334800, matched to completed v335 on v333800. Default ownership remains
unchanged for the independently authorized feasibility controls.

v335/v336 audits completed normally.660 lower ROI(all-RGB/canopy-only color):
PSNR12.825936/12.817992; surface contribution.394646/.394835; volume
.588323/.587910; floor-MSE.037895825/.037894364. Current MSE.052168261/.052263774.
Thus color ownership did not materially reduce the actual building leakage;
the tiny full-cohort metric gain is not a canopy-occlusion breakthrough.
