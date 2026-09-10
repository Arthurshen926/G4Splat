# v126: independent measured-leaf witnesses and local opacity updates

## Findings, not acceptance

At matched 2k, scalar normalization alone improved canonical48 canopy only 0.00292 dB. Raising **all** volume opacity LR to .02 instead of .002 improved canopy by 3.374 dB, but lost .785 dB rigid and 1.245 dB hard-pixel quality. View 690 acquired broad blurry foliage while still showing the church. Surface geometry and opacity fingerprints were unchanged, demonstrating that fixed geometry alone does not guarantee unchanged rigid rendering. The global-high-LR arm is rejected; supervisor 8869 received an intentional checkpointed stop request after iteration 2500. Its stop completion must be checked, not assumed.

The thin-versus-epistemic-envelope opacity-only diagnostics both completed 400 steps. Fixed16 tree changed 11.6705 -> 12.0135 (thin) or 12.0402 (wider); rigid 16.4409 -> 16.4203/16.4187. The incremental difference is small, so widening positive depth intervals is **not** adopted. Canonical observation inconsistency remains a measured limitation, not an excuse to fill a thick volume.

## Missing lifecycle edge

Initial measured-single leaves can only render/verify in their fixed support table. A single-entry table therefore cannot obtain a second independent witness, even where another canonical image has real matching depth. The earlier verification pass visits cameras but does not create that missing candidate edge.

The repair permits a **temporary candidate**, not direct persistence:

1. Existing measured-single static detail, no unresolved proposal; one real canonical witness; new camera and free candidate/witness slots.
2. Exact-K projection of its existing center into a different canonical camera, on known tree pixels with valid distortion/sky/transient masks and inside the unchanged thin MoGe interval.
3. Native mixed rendering exposes only those additional candidates in this trial view. Existing real-contribution verification must still see canopy contribution, not predominantly rigid content.
4. Retain the new support only if this camera was actually recorded as a witness. Unwitnessed candidates and failed render trials roll back their temporary support; global visibility still requires two real cameras.

`audit_v125_measured_independent_support_native48` found 22,487/29,785 measured leaves can obtain such witnesses. Native canonical48 tree improves only .02854 dB from merely making them eligible; this is not a quality breakthrough. This first audit also performed ordinary verification of other pending rows in visited views. A second `audit_v125_measured_only_support_capture` masks their responsibility out and saves a clearly diagnostic foliage capture (not a trainer checkpoint) for isolated follow-up optimization.

`diagnostic_v125_detail_control_1200` and `diagnostic_v125_detail_verified_1200` compare opacity-only optimization at the same camera order and step budget, without changing envelope/skeleton/rigid or leaf geometry/appearance parameters. Pending outcomes must be evaluated, not inferred from witness counts.

## Row-local learning rate

`outdoor/rowwise_optimizer_step.py` rescales static-detail Adam **displacement**, not gradients. Adam largely normalizes gradient scale, so gradient multiplication is not an equivalent LR change. The operation leaves moments unchanged and precedes optical-mass compensation, geometric projections and strict ownership vetoes. A float64, 20-step regression matches two independent Adam optimizers at LR .002/.02 exactly to 1e-12; default multiplier 1 is a bitwise no-op. The new control is explicit and checkpointed.

## Validation run

`pilot_v126_independent_witness_detail_lr02_common2k_to6k` starts from the immutable corrected v125 2k compact snapshot, not from the rejected global-high-LR model. It uses the original .002 envelope/volume base LR, multiplier 10 on static detail, independent measured-leaf verification, unchanged rigid appearance-only policy, zero rigid retirement/completion, and unchanged evidence/renderer/phase horizon. Requested stop is 6k with retained 3k/4k/5k/6k checkpoints. This is a causal continuation, **not clean reinitialization**; the old immutable initializer still predates the distortion-mask fix.

Supervisor at launch 27684, trainer 27685, GPU2. Startup/model updates must be checked separately from launch success. Code is archived in its `executed_python_sources.tar.gz`. Generic explicit trainer-repair resume allows only state-compatible Python modules; new helper hashes and training controls are persisted. No CUDA/renderer compatibility override is introduced.

## Scaling repairs

- The depth-slab solver constructs CPU float32 metadata contiguously and skips only **exactly feasible** fixed points. No relaxed tolerance, proposal rejection, quota reduction, or geometry change. 10,000-cell benchmark: 1.124 -> .312 seconds, centers and acceptance decisions bitwise equal. This does not bound total pending-cell growth.
- Both rolling and final checkpoints now compact small candidate vectors without numerical changes. Current v124/v125 controls execute their archived older code and do not acquire this repair in place.
- Real compact 2k data.pkl is 532,438,483 bytes, despite only 957 ZIP entries. The supervisor's fixed 256 MiB cap incorrectly rejected it. Uncompressed metadata whose declared bytes fit the physical checkpoint is now CRC-checked in 1 MiB chunks; compressed expansion remains capped. The actual 1.575 GB checkpoint passes validation. The legacy two-million ZIP-member guard remains; old millions-of-tensors snapshots may require an explicit compact copy before supervised recovery.

## Status

No production model has yet passed the requested dense-canopy/no-rigid-regression acceptance. Continue actual native multi-view validation and visual inspection; neither passing tests nor launching v126 establishes the target outcome.

## September 9 continuation: measured outcomes and superseded attempts

- Matched 3k native canonical48: v124 tree 11.0967857, v125 11.1048865 (+.0081008 dB); rigid 16.6825260 -> 16.6816185; hard 14.8145250 -> 14.8123836. The normalization repair remains mathematically necessary but is not a practical reconstruction breakthrough.
- The first v126 launch failed before training because a derived positive-camera population audit count was compared inside the immutable negative-cleanup camera schedule. Only `positive_exact_support_camera_count` is now excluded from that comparison; actual camera list SHA, count, canonical sequence, cadence, and weights remain strict. Five regressions cover the distinction.
- `...independent_witness_detail_lr02_common2k_to6k_r1` was intentionally stopped and successfully checkpointed at 2150, to add a missing foreground condition. Nonzero native contribution behind an almost opaque wall is not a valid foreground witness.
- `pilot_v126_front_verified_detail_lr02_common2k_to6k_r2`, supervisor 1059 / trainer 1064, restarts from the same immutable corrected v125 2k snapshot. Both ordinary canonical verification and auxiliary MoGe verification now require center depth before independently rendered opaque rigid depth, with clearance max(.03,.01*z). New measured support additionally requires the original unchanged thin MoGe interval. The run has performed real updates through 2250; final quality is pending.
- The rejected global-LR run was deliberately stopped at 2501. Its old supervisor rejected the saved ZIP's 9,139,512 members; this was not an OOM or a spontaneous training crash. Its validated compact 2k snapshot is preserved. New compact checkpoint encoding prevents millions of tiny tensor storage entries without changing numerical values.
- Opacity-only detail control at 1200: fixed16 tree 12.0650, rigid 16.4088. The independently verified diagnostic at 1200 reaches tree 12.7264, rigid 16.2838. That capture predates the new strict foreground check and still shows church transmission / dots; it is rejected as an acceptance result.
- Native 3k evaluation used explicitly marked **non-resumable** render-state extracts. Extraction defers training-only tiny storages and discards optimizer/candidate state; all 148 retained tensors were compared bitwise against normal `torch.load` of the real 2k source, including NaN representations. Original checkpoints are unchanged. Extraction is not a recovery/migration path for training.
- A separate fresh dense canonical seed diagnostic excludes all 48 prespecified evaluation cameras from seeding and second-camera verification. It uses exact-K source depth before opaque rigid geometry, small tangent footprints and .02 depth scale (not epistemic uncertainty as material). All leaves start measured-single with one source observation. Persistence still requires an independent thin-depth candidate followed by a real native foreground contribution. This is a capacity experiment, not a production checkpoint or evidence of success.
- Root-wide pytest collected an optional upstream tetra-triangulation test whose `tetranerf.cpp` extension is not built in this environment: 1 failed, 1106 passed. This is outside the active canopy/native renderer path and is recorded rather than hidden. The new extraction and dense-unprojection targeted tests both pass. Do not describe root-wide pytest as all passing.
