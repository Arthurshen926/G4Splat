# Dense seed capacity isolation, not production acceptance

The original matched v124/v125 3k comparison improved canonical48 canopy by only .00810 dB. A new diagnostic isolates dense canonical foreground geometry while retaining the original rigid model and the production native renderer.

`build_dense_canonical_canopy_diagnostic.py` excludes all fixed16 + prespecified additional32 views from seeding **and** second-camera verification. The opacity-only follow-up uses `--validation-cohort canonical48` and also excludes these 48 views from its own optimization. They remain reconstruction views used by the historical input model, not held-out original RGB training.

Both new seed arms use the same cameras, exact calibrated K/poses, production MoGe metric scale, validity masks, depth-before-independent-opaque-rigid rule, initial opacity .04, and .02 camera-depth thickness. Tangent scale is proportional to source sampling spacing, capped at .15 world units. Fine .015 voxel deduplication creates no extra view authority. One actual source measurement initializes `MEASURED_SINGLE`; a second thin-depth-consistent native foreground witness is required for persistence. No inference mask/depth oracle is introduced.

| Source samples per camera | New leaf rows | Second-camera verified rows |
| --- | ---: | ---: |
| 384 | 71,024 | 41,123 |
| 2048 | 361,082 | 213,775 |

Initial canonical48 native tree / rigid PSNR: 384 arm 11.95586 / 16.30578; 2048 arm 13.11694 / 16.09822. Input v125 2k was 10.93692 / 16.65432. Thus dense initialization raises coverage but **already harms rigid rendering**, despite unchanged rigid geometry. View 690 still exposes the church. Neither initializer is accepted.

Active follow-ups at creation of this record:

- `diagnostic_v126_fresh_dense_2048_optical400_r2`, GPU0.
- `diagnostic_v126_fresh_dense_384_optical400_inputref`, GPU1.
- Both: 400 steps, opacity-only Adam .05, original thin optical weight .02, RGB weight 1, negative weight 1, original-model rigid-reference weight 3, identical 187-camera order. All other learned parameters are read-only; no unused geometry gradients accumulate. Native canonical48 metrics / six difficult-view images are saved at 0/100/200/300/400.

Superseded diagnostic attempts are retained: first dense optimization failed before updates due to restoring a fresh model's dynamic-rank schema into the old model; replacement now constructs a model from the explicitly diagnostic capture's own schema and rejects dynamic leaves. The `_r1` arm was intentionally stopped before completion because rigid-reference preservation used the fresh initializer as its baseline, accidentally preserving its new contamination. `_r2` uses the **input checkpoint's** rigid-pixel rendering as the fixed reference. This diagnostic repair does not change the active v126 trainer.

## Candidate lifecycle audit, still unresolved

Read-only inspection of the immutable v125 2k candidate state found 1,750,889 pending visual-hull cells. 31,930 contain consumed first-hit camera constraints (53,197 such constraints), including 6,560 wholly spent cells, 9,488 with one remaining live camera, and 15,882 with at least two live cameras. This is not enough to explain the whole backlog. Existing support eligibility excludes spent witnesses, but stored center/color/depth constraints still include their history. A future lifecycle change must distinguish spatial evidence from positive optical funding and preserve valid unspent observations; indiscriminate clearing/re-crediting would violate the one-hit rule. No unsupported lifecycle migration has been applied to the running experiment.

No new initializer or optimization in this document has yet demonstrated the requested dense-canopy/no-rigid-regression result.
