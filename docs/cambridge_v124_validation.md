# v124: single first-hit material allocation and canonical canopy training

## Controlled experiments

Clean 3,000-iteration prefixes with the unchanged 30,000-iteration phase horizon, seed 1701, resolution 640, original v113 initialization, and rigid v113 warm start. Surface policy is appearance_only, surface retirement budget zero, rigid completion zero. Geometry is protected; rendered rigid appearance must still be measured because foreground foliage can obscure it.

- `pilot_v124_single_hit_canonical_stream_lr002_3k`: GPU 0, volume opacity LR 0.002.
- `pilot_v124_single_hit_canonical_stream_lr02_3k`: GPU 1, volume opacity LR 0.02.

Update: the lr02 arm was intentionally stopped/checkpointed at 998 after the scalar optical normalization defect was reproduced. It is not a completed 3k experiment. The lr002 arm remains the control; GPU1 is reassigned to the v125 same-LR normalization experiment from its 1k common prefix. See `cambridge_v125_scalar_optical_normalization.md`.

Both use corrected intrinsic query CUDA, local extension binding, fine spatial capacity with 0.04 separation, single-first-hit witness consumption, seeded spatial proposals with budget 1024, and the independent canonical-canopy MoGe schedule. The MoGe initialization distortion-mask repair is in code but is **not** present in the old immutable initialization used for this controlled comparison.

The saved 500-step contract confirms 233 canonical canopy cameras in the production MoGe stream (different from the diagnostic calibration's 219 after excluding fixed16). The optional canonical analytic-ray supplement is disabled: every=0 and camera_count=0. Its old full-step schedule indexing remains a reproducible dormant bug, but cannot explain these runs. Do not conflate it with the repaired active MoGe schedule or silently change the running implementation mid-experiment.

## Preselected evaluation views

Original canonical 16: 558,632,633,634,663,671,677,682,690,707,736,743,756,768,776,909.

Additional 32, selected before v124 results by tree-mask coverage (at least 20%) and camera position/orientation diversity, not reconstruction scores: 729,713,655,697,748,717,760,680,733,694,710,763,751,656,685,777,668,704,703,731,705,692,766,698,772,715,711,674,660,657,738,747.

Cross-sequence diagnostic only: 408,409,410. Do not pool these with the canonical reconstruction score. These views are not an unseen training test set; the full-scene RGB training has access to them.

Compare native renderer RGB with no semantic oracle routing, canopy interior/boundary/rigid/hard-pixel metrics, thin-hit coverage and actual post-Adam growth, and fixed visual pairs. Increased opacity alone is not success: previous diagnostic optical calibration gained about 2 dB canopy PSNR while losing rigid quality and producing blobs. No such diagnostic has been accepted as a production repair.

## Acceptance status

Pending trained multi-view validation. Unit tests and launch status do not establish an effective reconstruction improvement, much less complete elimination of unwanted building visibility.

## Baseline and corrected real replay

`eval_v122_native_prespecified51/metrics.json` reproduces the prior fixed16 scores (tree 11.75647, rigid 16.49011 dB). Additional32 tree is 10.64855, rigid 16.78494 dB. All canonical48 mean-view tree is 11.01785, rigid 16.68667, hard-pixel 14.81961 dB. Mean-view and pooled-pixel PSNR are reported separately. The evaluator also fingerprints rigid xyz, scale, rotation and opacity for bitwise protection checks.

`diagnostic_v124_single_hit_birth_replay/birth_audit.json`: 223,742 identical input proposals at budget 1024 per qualifying view produce 18,927 coarse versus 28,903 fine candidates, both under single-first-hit consumption. This is +52.7% candidate capacity, not trained quality or verified density. Fine replay ends with 894,351 pending cells, a CPU/checkpoint-scaling risk to monitor. These counts cannot be compared causally against the earlier 256-budget replay.
