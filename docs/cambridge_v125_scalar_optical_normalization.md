# v125: optical scalar-map normalization defect

## Reproduction

`_weighted_mean` is a CxHxW reducer: its denominator includes value.shape[0]. MoGe's prehit optical-depth map and hit log-tau Huber map are HxW. Passing them without a channel dimension divides the intended weighted mean by image height again. At 640x360, both loss and gradient are 360 times smaller than the declared factor. Example v122 iteration 2800 reports hit loss 0.03763438, rather than approximately 13.54838. This changes the balance with other evidence/regularization; it is not merely a logging bug. Adam and gradient ownership can make the final parameter effect nonlinear, so do not claim 360x parameter growth.

The new regression repeats identical scalar content at heights 1, 7 and 36 and checks loss, both separated components and the gradient of shared logits. It fails on the old implementation with a 1/7 discrepancy. Explicit one-channel dimensions on the **two MoGe loss calls only** fix it. Other RGB, rigid, geometry and cleanup objectives are unchanged. Targeted trainer tests: 329 passed; full suite: 1,079 passed, 15 warnings.

## Controlled experiment

Keep GPU0's `pilot_v124_single_hit_canonical_stream_lr002_3k` running as the old-normalization control. Archive its executed Python sources and preserve `common_prefix_001000.pth`; its trainer hash was checked against that archive. MoGe optical supervision is inactive before the 1500-step detail phase, so the 1000-step state is a valid shared prefix before this treatment.

The high-LR v124 arm was intentionally stopped/checkpointed at iteration 998, without an OOM or spontaneous failure. GPU1 now runs `pilot_v125_scalar_optical_mean_common1k_to3k`, initialized from the immutable common prefix, opacity LR 0.002, same evidence, model, renderer, camera schedules, phase horizon, budgets and initial geometry. The existing explicit trainer-only repair-resume mechanism is used; no CUDA/model/data compatibility changes are waived. This is a common-prefix causal continuation, not a fresh initializer run or an exact unmodified resume.

Do not change the live source again during these two runs. GPU0 executes the archived old normalization; GPU1 executes the scoped corrected normalization. Comparison must use saved trainer hashes, not the current working-tree filename alone.

GPU2 subsequently became free (no external compute process remained). `pilot_v125_scalar_optical_mean_lr02_common998_to3k` continues the intentionally stopped high-LR v124 checkpoint at 998, using the same normalization repair and opacity LR 0.02. This additional arm tests update speed; unlike the GPU0/GPU1 pair, it has a high-LR prehistory and is not a strictly identical-state normalization-only comparison. Supervisor PID at launch 8869, trainer 8873. The other training processes are not stopped.

## Status

Awaiting actual post-Adam growth, coverage and native multi-view reconstruction validation. Fixing normalization does not establish correct geometry, absence of ownership conflicts, adequate representation capacity or elimination of background leakage.

Matched live step 1510, before significant trajectory divergence: both arms expose 346 positive-gradient rows; source growth magnitude changes from 0.000304218 to 0.10957025. Total post-Adam logit delta over the audited owner rows changes from 0.915529 to 2.208982. This confirms a real optimizer effect, not just a scalar log correction; that total delta is **not** an attribution of every update to positive MoGe evidence. Mean thin alpha at this early step remains about 0.00847 in both arms. Step 1506 has zero coverage and zero growth in both: normalization cannot create missing geometry.

At control step 1600, 21,826 accumulator cells are eligible but verification-debt backpressure limits births to 512 (configured 2048, capacity scale 0.25). All 512 receive initial mass funding. The pending map has 487,956 cells. Thus candidate generation, accepted birth throughput, actual two-camera verification and post-verification training time must be reported separately. Increased candidate counts alone are insufficient.
