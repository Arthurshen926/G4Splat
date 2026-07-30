# Cambridge unified hybrid mainline audit (v39)

This note records the implementation status of the external design reviews
against `codex/hybrid-causal-repair-v2`. It deliberately distinguishes a
stored artifact, an exercised training factor, and a completed method.

## Evaluation contract

- Fixed Cambridge intrinsics/extrinsics: 1,487/1,487 cameras, exact identity.
- Training/evaluation RGB: the same content-addressed 640x360 image mapping.
- Primary metrics: `protocol_aggregate`, per-view metrics followed by a mean,
  historical global-scalar SSIM, and the same static/dynamic/tree masks.
- `aggregate.ssim` is a windowed optimization diagnostic and must not be
  compared with the historical scalar SSIM.
- Canonical rendering never receives ground-truth semantic masks.
- Conditioned deployment rendering derives ownership from rendered
  surface/volume/sky alpha, not from ground-truth semantics.
- No historical trained Gaussian, COLMAP point/track geometry, or historical
  all-real-RGB Gaussian initialization is used by v39. The handed-off surface
  is the current no-COLMAP rigid stage declared by
  `native-rigid-surface-handoff-v1`.

## Advice implementation matrix

| Design item | Status | Concrete implementation / remaining gap |
| --- | --- | --- |
| Fixed calibrated Cambridge cameras and exact-K pointmaps | Implemented | Source-depth retargeting preserves camera-z and unprojects with the fixed camera. Runtime camera hashes are immutable resume/evaluation contracts. |
| Unified, provenance-checked Evidence Store | Implemented | Evidence, initialization and RGB content hashes are checked independently. Training logs actual source/factor consumption. |
| Remove COLMAP point-cloud/track geometry | Implemented on the active mainline | Camera calibration remains; surface and foliage geometry come from MASt3R/MAtCha/DAV2 evidence. |
| Preserve SfM track and primitive-role metadata | Implemented | Track lineage, source, layer role, tree instance, support cameras/sequences, observation UV/depth and covariance survive initialization, split and checkpointing. |
| Observation-level MASt3R factors | Implemented as the current transition model | Track observations and native pointmap pixels remain external evidence and supervise rendered/conditioned geometry. |
| Strict reciprocal descriptor/union-find MASt3R track graph with full A-B-C cycles | Partial | Fixed-camera multiview tracks, exact-K reprojection, retriangulation and cross-sequence support exist. The graph is still primarily projection/pointmap based and does not yet implement the full descriptor-cycle system over all 1,487 views. |
| True continuous MAtCha Chart atlas with learnable inverse-depth field and UV quadtree | Not complete | Native-resolution Chart observation factors, retained Chart provenance and Chart-aware surfels exist. `ChartSurfaceModel` is still a discrete observation graph, not the proposed continuous atlas parameterization. |
| Geometry factors at evidence/source resolution | Implemented | Pointmap/Chart factors use source camera coordinates and native evidence resolution rather than pretending low-resolution Chart pixels are high-resolution targets. |
| Independent RGB / geometry / topology evidence epochs | Implemented | Deterministic independent schedules and consumption coverage are checkpointed and audited. |
| Occlusion-aware probabilistic visual hull | Implemented | Hit/free/unknown labels, rigid z-buffer occlusion, multi-view/sequence/baseline/angle requirements, posterior covariance and occupancy are retained. |
| True ray/depth posterior loss | Implemented | Persistent per-ray camera-z intervals are converted exactly to unit-ray distance and optimized with analytic Gaussian transmittance. Unknown/occluded evidence is not treated as free space. |
| Dense posterior has a renderer consumer | Implemented in v39 | Every valid dense hit now creates a low-opacity, exact-camera, sequence-local 3D Gaussian birth with a finite source-pixel footprint. |
| Multiple physical tree instances in one connected mask | Implemented in v39 | Instance ownership is propagated per pixel from the nearest calibrated local 3D observation inside each mask component. A component-wide majority vote no longer erases touching crowns. |
| Canonical and dynamic ownership | Implemented as explicit replacement, not deformation | Canonical crown is cross-sequence; dynamic leaves are exact/same-sequence gated. Replacement is local and opacity-progressive. A single deforming canonical primitive is not implemented. |
| Volume adaptive densification/split | Implemented | Per-role, per-instance and spatially balanced selection; optical-footprint-preserving split; optimizer-state migration; child priors; screen/world-scale caps. |
| Per-candidate replace-and-retire | Implemented as a bounded cleanup mechanism | Local replacement groups, evidence lineage and gradual mass handoff are active. It is not treated as a global success metric or hard gate. |
| Static trunk/branch branch | Representation implemented; evidence absent | No fake skeleton is inferred. The current StMarys evidence contains zero line/cylinder-supported trunk/branch seeds, so this branch is honestly empty. A measured line/cylinder producer remains missing. |
| Sequence/time dynamic leaf branch | Implemented | Exact-camera visibility, same-sequence temporal decay, low-rank deformation, direct UV/depth observation factor and dynamic-only topology are present. |
| Spatial uncertainty instead of two image scalars | Implemented | Per-view spatial uncertainty grids route residual confidence; early geometry gradients do not pass through the uncertainty escape path. |
| Native 2D surfel + 3D EWA same-tile rendering | Implemented | Both primitive families enter one CUDA tile list, one center-depth radix order and one front-to-back pixel loop. |
| G4 visibility-based rigid completion behind trees | Partial | Rigid occlusion evidence and ownership conflicts exist, but a complete cross-view background birth/verification producer is not yet a standalone closed stage. |
| Standard universal 2DGS/3DGS export | Deferred by the current teacher-only decision | The authoritative output is the native mixed Teacher. A standard student/baked export is not part of the active run and should not be claimed as complete. |

## Causal chain defects fixed after v34

1. Ray intervals were stored in camera-z but compared directly to Euclidean
   unit-ray distance. Off-axis rows were displaced by roughly 4–15%.
2. Ownerless single-sequence hits were incorrectly admitted to the canonical
   posterior without same-ray accepted geometry.
3. The code comment said rejected ownerless hits belonged to the dynamic
   factor, but no dynamic primitive was created for them.
4. A connected tree mask was incorrectly treated as one physical instance.
   In 00408, 926/928 exact points projected inside the tree mask, yet the
   component purity rule deleted every dense posterior row.
5. Dynamic leaves were frozen until iteration 10,500 despite a mature rigid
   handoff. They now start after the short 2,400-step canonical bootstrap.
6. Dynamic observation subsampling selected the same 2,048 rows on every
   visit. A rotating window now eventually covers every exact-view birth.
7. A 0.02 replacement normalization made a newly initialized 0.025–0.08
   dynamic seed immediately erase its canonical predecessor. Replacement is
   now gradual relative to 0.08 optical mass.
8. Posterior verification/coverage state was not included in resumable
   checkpoints. It now has an exact runtime-state contract.

## v31 initialization audit used by v39

- Surface: 153,440 evidence seeds; unchanged from v35.
- Canonical crown: 85,143 (previously 12,252).
- Sequence-local dynamic leaves: 293,886.
- Dense dynamic births: 236,269, exactly matching dense ownerless hit rows.
- Total foliage: 379,029.
- Ray table: 1,193,732 rows across 96 cameras.
- Canonical support: minimum two views/two sequences; median five views and
  three sequences.
- 00408 owner: 3,434 dense births plus 928 pointmap/DAV2 observations, with
  normalized UV support spanning almost the full frame.
- Static skeleton: zero; no inferred/fabricated trunk evidence.

## Results

The complete v39 checkpoint and all-view evaluation are appended here after
the running 30k validation. At iteration 1,000 (before the dynamic branch is
valid), the 00408–00410 diagnostic already changed as follows:

| Checkpoint | Raw PSNR | Tree-static PSNR | Non-tree-static PSNR |
| --- | ---: | ---: | ---: |
| v34 final 30k | 9.915 | 9.876 | 12.384 |
| v39 1k canonical | 11.684 | 12.241 | 11.631 |

This establishes that the evidence-to-primitive and instance-assignment
repairs restore missing crown coverage. It does not yet establish high
frequency quality: the 1k canonical render is intentionally low-frequency,
and the conditioned dynamic branch starts after iteration 2,400.

