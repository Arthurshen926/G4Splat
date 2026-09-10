# Pending birth verification latency

Read-only checkpoint-copy audit: `audit_v122_pending_real_verification/verification_audit.json`.

At v122 iteration 3000, 3,661 ray births remain unresolved. A sweep over their exact candidate cameras, using the same native mixed renderer, same task-field lookup, all actual occluders and the existing two-camera verification function, resolves 3,312. Zero-witness ray births fall from 3,396 to 40. The sweep executes 119 of 122 available candidate cameras. No evidence count is fabricated from a sequence label or an interval query; these are real rendered T-before-times-alpha witnesses.

This is a demonstrated attainable sweep result, not a geometry-independent upper bound: newly verified leaves acquire ordinary deployment visibility and can occlude later candidates. It does not make failed candidates valid. No input checkpoint is modified and no model is saved. The existing verification routine also refreshes DC colour from real RGB witnesses, so any before/after image comparison includes that normal colour initialization, **not just a visibility-gate change**. Geometry and opacity are not optimized.

The v124/v125 active canonical MoGe stream already adds more frequent real-render verification than v122. Measure the remaining delay in those runs before assuming all of the old backlog still exists. Also distinguish the inevitable final-step newborns of a 3k prefix from a fully settled 30k model.

Current capacity evidence: both the old v122 and new fine-grid control hit a 512-per-event allocation limit from global verification debt. Finer candidate generation alone does not increase actual birth count when this limit is saturated. Unrelated unresolved scaffold/split populations can therefore throttle independently depth-supported ray births. A follow-up should inspect per-proposal-kind/age debt and useful missing-camera verification visits, while retaining exact real-camera acceptance and rigid protection.

## Native image result: verification alone is insufficient

`audit_v122_pending_real_verification_native48/verification_audit.json` repeats the sweep with native before/after evaluation. Mean-view canonical48 tree PSNR changes only 11.01785 -> 11.06498 (+0.04713 dB); rigid 16.68667 -> 16.68360; hard 14.81961 -> 14.81848. Mean-view canopy surface alpha changes 0.72027 -> 0.71493 (this is not the pooled-pixel alpha in the other evaluator). View 690 still shows the church through the dense crown. Do **not** claim that fixing verification delay alone solves canopy leakage. Low optical mass, insufficient actual accepted density and geometry/coverage remain the important downstream tests.
