# Cambridge native hybrid mainline audit (v57)

This audit is intentionally stricter than a code-presence checklist. A design
item is marked complete only when its producer, persisted contract, training
consumer, checkpoint/resume path, renderer and evaluation path agree.

## Answer to “were all prior recommendations implemented?”

No. The current branch closes most renderer, camera, evidence-lifetime and
foliage-posterior correctness defects, but four major method components remain
partial or absent:

1. the saved MASt3R graph is fixed-camera geometric pointmap agreement, not a
   database-wide reciprocal descriptor/cycle/union-find track graph;
2. `ChartSurfaceModel` is a persistent observation graph, not a continuous
   learnable MAtCha inverse-depth atlas with UV-domain quadtree refinement;
3. rigid background completion behind foliage is not yet a standalone
   G4-visibility birth/verification stage;
4. evaluation is still all-1,487-view reconstruction fit, not an isolated
   Cambridge database/query localization protocol.

Those gaps must not be described as enabled merely because the Evidence Store
contains MASt3R or Chart-named arrays.

## End-to-end implementation matrix

| Contract or method | Status | Evidence |
| --- | --- | --- |
| Exact fixed Cambridge K, including principal point | Closed and exercised | All 1,487 cameras are identity checked at training/evaluation; camera geometry and RGB content mappings are checkpoint hashes. |
| No COLMAP point/track geometry and no historical trained-Gaussian initialization | Closed on the active run | The surface handoff and Evidence Store both reject those provenance sources. Fixed Cambridge calibration remains allowed. |
| RGB source identity | Closed and exercised | Seed, training and evaluation compare content-addressed image mappings. A launch against `dataset_qc_tree_v5/train_all` was correctly rejected before step zero; the active run uses `cambridge_all1487_shared640_bilinear_v1`. |
| Plane/Chart/inverse-depth provenance separation | Closed | Plane is an external factor, not a hard depth overwrite; source-resolution Chart and pointmap factors retain their own provenance. |
| Independent RGB, geometry and topology epochs | Closed and exercised | Separate deterministic schedules and factor-consumption state are checkpointed. |
| Persistent observation-level track factors | Closed for the current graph | Real camera/UV/depth observations remain external to Gaussian topology and are consumed after split/prune. |
| Strict descriptor MASt3R graph | Not complete | Current graph is projection/pointmap agreement based and covers selected evidence cameras, not all database keyframes and reciprocal descriptor cycles. |
| Continuous MAtCha Chart atlas and UV refinement | Not complete | Native Chart observations constrain rendered surfaces, but the representation is still discrete and cannot split in chart UV. |
| Occlusion-aware probabilistic visual hull | Closed and exercised | Hit/free/unknown, rigid z-buffer occlusion, sequence/baseline/angle support, spatial covariance and per-ray intervals are persisted. |
| Unit-consistent ray/depth likelihood | Closed and exercised | Stored camera-z intervals are converted to unit-ray distance before analytic transmittance likelihood. |
| Native 2D surfel + 3D EWA same-tile renderer | Closed and tested | One CUDA tile list, center-depth radix order and front-to-back pixel loop; both backward paths have focused tests. |
| Three foliage roles | Closed in representation; evidence quality differs | Canonical crown and dynamic leaf are dense. v15 now produces measured static branch/trunk candidates rather than an empty or semantic-only branch. |
| Spatial uncertainty | Closed | Per-image spatial grids route uncertainty; geometry cannot escape through early uncertainty gradients. |
| Volume adaptive split/prune and optimizer migration | Closed and exercised | Role/instance/spatial allocation, optical-mass-safe split, prune and Adam-state migration are one topology transaction. |
| Canonical/dynamic ownership | Closed as explicit local mass replacement | Same-sequence visibility and temporal residuals are separated from exact base-state ownership; canonical attenuation is local optical mass, not a global gate. |
| Sequence-local foliage graph | Closed as a conservative transition graph in v15 | Reciprocal cross-camera nearest correspondences are limited to one sequence, one tree instance, an eight-frame span, 6 cm, and at most three observations. |
| Per-candidate surface replace-and-retire | Closed as a continuous posterior in v17 | It starts after surface topology stabilizes, accumulates cross-sequence depth/counterfactual evidence, and gradually attenuates optical depth instead of applying a 0.45 hard cliff. |
| G4 visibility rigid completion behind trees | Partial | Occlusion and ownership factors exist; no complete rigid birth/verification producer exists. |
| Standard universal 2DGS/3DGS export | Deferred | The current explicit decision makes the native mixed Teacher authoritative. Standard export would require a real bake/distillation stage and is not claimed. |
| Database/query localization evaluation | Not complete | Current all-view metrics are reconstruction diagnostics and historical-comparison metrics, not localization proof. |

## v15 foliage contract repair

Source:
`initialization_local_instance_dense_exactrgb_v12_supportedhit_compact`

Output:
`initialization_local_instance_dense_exactrgb_v15_measured_skeleton_sequence_graph`

- Input foliage rows: 1,072,236.
- Canonical candidates: 114,308.
- Measured static skeleton rows: 387.
- Sequence-local reciprocal groups: 29,549.
- Multi-camera rows participating in those groups: 64,708.
- Duplicate alpha owners removed: 35,159.
- Output foliage rows: 1,037,077.
- Dynamic rows with two or three real observations: 30,817.
- Same-camera rows merged: zero.
- Ray-bound canonical rows removed: zero.
- Persistent ray table: unchanged at 2,276,243 rows.

The skeleton classifier stores continuous confidence and measured anisotropic
line/cylinder frames. It requires cross-sequence canonical support and a
coherent local geometry. Wood colour, occupancy, depth NLL and free-space
evidence modulate confidence rather than forming an all-or-nothing chain.

## Validation protocol

The causal short run uses the same no-COLMAP rigid handoff, Evidence Store,
640x360 RGB content mapping and target indices as v52-v55. Checkpoints at
1k/2k/4k are retained. Every checkpoint must be evaluated twice:

1. indices 408-410 for the known tree-heavy failure mode;
2. all 1,487 views for raw/static/non-tree/tree metrics.

Selected-view improvement alone is not accepted as whole-scene improvement.
The historical comparison remains PSNR/SSIM/MAE
18.755777/0.830073/0.082067, but that model is never used for initialization.

## Remaining causal ceiling

Even a perfect foliage repair cannot exceed a weak rigid foundation. The
current allowed 24k no-COLMAP rigid handoff reaches approximately 15.91 PSNR
on the 27-view diagnostic and about 15.78 non-tree PSNR on all 1,487 views,
well below the historical full-scene model. Therefore the next architecture
work after v57 is not another foliage loss coefficient:

1. build and persist a true descriptor/cycle MASt3R rigid graph;
2. replace the discrete Chart graph with a learnable continuous Chart atlas;
3. add verified G4 visibility rigid completion behind foliage;
4. retrain the rigid handoff under those three producers before the final
   long mixed run.

## v57 results

Results are appended only after the checkpoint and evaluator artifacts exist.
