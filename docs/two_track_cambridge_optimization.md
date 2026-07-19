# Cambridge two-track optimization

This implementation keeps two experiments separate.

## Track A: repair the native G4Splat geometry frontend

The Cambridge runner now enables three safety policies:

1. The aligned-chart gate rejects both severe depth conflicts and charts with
   too little surviving support. `charts_data.npz` records
   `alignment_gate_valid` and the JSON report records explicit reasons.
2. Plane refinement is a bounded residual correction. A chart with insufficient
   support or excessive depth change falls back to its aligned depth; isolated
   excessive pixel changes are also reverted. The original refined TIFF is kept
   as `*.pre_safety_gate.tiff`.
3. With See3D disabled, the final 30k pass loads the actual 7k PLY through
   `--init_ply`. It no longer moves the 7k point-cloud directory and recreates
   the scene from chart points.
4. Canonical tree pixels are hard-masked for Chart alignment, the aligned-depth
   conflict audit, and chart-to-plane construction. The later free-Gaussian
   stage retains its softer tree weighting, but trees cannot become geometric
   Chart anchors.
5. A globally fused plane must have support in at least two distinct Charts
   before it can rewrite aligned depth. A one-view SAM region is useful only as
   a local segmentation; treating it as a global surface creates an
   unconstrained depth prior. Each run writes
   `plane-refine-depths/global_plane_cross_view_support_audit.json`, including
   every rejected singleton plane and the per-Chart eligible-plane coverage.
   A Chart with no eligible planar patch simply keeps aligned depth; this is
   not silently converted into a failed or generated geometry observation. If
   pseudo views exist, only the original real Chart IDs count toward those two
   views. The default is
   `--min-global-plane-views 2`.

All 1,487 StMarysChurch training images remain dense reconstruction inputs and
coverage targets.  QC never creates a validation split.  Train-fit rendering
and metrics use that same QC dataset (including normalized poses), rather than
silently switching back to the raw pre-QC COLMAP cameras.  Initial Chart
eligibility, the hard alignment gate, and the quality-aware active subset are
separate, auditable decisions; inactive Charts are hard-zeroed before plane
construction rather than left available to leak invalid geometry downstream.

Run only the alignment/gate feedback round first:

```bash
conda run -n g4splat python scripts/run_cambridge_g4splat.py \
  --phase frontend --scenes StMarysChurch --run-tag chart-r1 --gpu 0
```

Use a new `--run-tag` after every replacement round. The tag participates in
the configuration digest, so pointmaps/alignment from an older Chart set cannot
be silently reused. Once a round has no rejected Chart, run a controlled screen
with the same accepted selection and an isolated tag:

```bash
conda run -n g4splat python scripts/run_cambridge_g4splat.py \
  --phase screen --scenes StMarysChurch --run-tag accepted-rN --gpu 0
```

After alignment, inspect:

```text
<run>/mast3r_sfm/aligned_chart_conflict_gate.json
<run>/mast3r_sfm/plane-refine-depths/plane_refinement_safety_gate.json
```

The 7k screen is only a geometry and renderer gate, not a fixed final budget.
Its densification schedule is explicit as
`--screen-free-gaussians-config`: `default` is the historical 7k schedule,
while `screen_long7k` keeps densification through the entire screen and delays
the normal/distortion terms.  This option participates in the screen identity,
so its effect can be compared without mixing Chart, Plane, or final-schedule
changes.  The full pass accepts `--final-iterations 20000|40000|80000`; its output name
records that budget. The screen identity deliberately excludes those final
schedule flags, and a full run clones rather than renames the passed screen,
so multiple final budgets start from bit-identical 7k geometry. For runs
longer than 30k, the default
`--final-non-position-lr-decay-from 30000`
`--final-non-position-lr-final-mult 0.1` decays appearance, opacity, scale,
and rotation learning rates while leaving the position schedule independent.
This makes a longer run a controlled schedule ablation rather than an implicit
change in all parameter groups.

The chart selector now stores its complete filtered reserve pool. If alignment
rejects charts, re-optimize the *entire* redundant set without pinning a prior
accepted Chart and keep target coverage over all 1,487 reconstruction cameras:

```bash
python scripts/select_joint_chart_set.py \
  --selection <prepared>/chart_selection_n48.json \
  --gate-report <run>/mast3r_sfm/aligned_chart_conflict_gate.json \
  --scene-path <prepared>/dataset_qc_tree_v5/train_all \
  --n-images 40 \
  --output <prepared>/chart_selection_joint40.json
```

Rejected charts are removed from the reserve pool and replacements are selected
by the same pose/direction, baseline, sequence-aware target-k-center objective.
Their temporal neighbours are recorded as a quarantine set, but are not made
automatically ineligible: a single failed Chart is not evidence that every
neighbour is geometrically invalid. A correlated-failure corridor is physically
blocked only after three direct failures establish a run (with the configured
small bridge gap and padding); it is recorded in the selection JSON. Any
remaining selected quarantine neighbour must pass the next joint alignment
gate.
After the second alignment gate, run `select_quality_aware_charts.py`, audit its
active coverage, and use `--active-chart-selection-json` so the selected subset
is actually consumed by plane/2DGS geometry. Repeat until there are zero active
depth-conflict Charts, no support-deficient active Chart, and coverage passes;
do not reuse pointmaps from a different chart set.

For geometry ablations, use the full 1,487-view QC dataset and vary one of the
joint Chart set, active quality subset, plane refinement, downsampling, or
supervision schedule at a time.

## Track B: frozen reconstructed 2DGS/MAtCha plus causal 3D-ROI repair

`scripts/causal_2dgs_repair.py` is the post-reconstruction repair entry point.
It keeps the source checkpoint immutable and follows this order:

```text
all 1,487 real-train RGB/structure prescan → selected real anomaly rays
→ masked-gradient primitive candidates → compact 3D/co-view clusters
→ opacity-zero counterfactual → causal class
→ aligned MAtCha Chart (original SfM/MASt3R corroboration or fallback; SIFT last) / plane / density-filtered footprint
→ G_remove, G_update, G_create → selected-row constrained edit → validation
```

The prescan consumes the frozen checkpoint's own all-train render cache and
the matching GT cache. It records all 1,487 names in
`full_train_anomaly_prescan.json`, then merely ranks where the costly
alpha/depth/distortion rerender should run. It has no geometric authority and
cannot itself create an ROI or approve an edit. The cache is required to be
under the frozen model path, preventing a stale metric directory from being
silently used to choose causal targets. A manual target list remains an
explicit experiment override, but does not bypass the complete prescan. Version
5+ reports fail the contract audit unless that all-train prescan completes.
Without a manual list, the default target hand-off is a deterministic greedy
joint objective over the high-residual prescan pool: residual priority,
marginal real-camera pose/direction coverage, and sequence novelty. This
prevents a handful of adjacent frames from monopolizing a causal batch while
remaining strictly a triage policy; the saved `target_selection` rows record
each marginal gain and have no geometric authority. Use
`--auto-target-selection priority` only for a controlled ranking-only
ablation.
The causal loader reads all 1,487 COLMAP pose/intrinsic records without
eagerly decoding every RGB image; it loads image tensors only for target,
control, and real-support views that are actually rendered. Thus full-train
coverage is retained without making a multi-cluster diagnosis pay the memory
cost of 1,487 full-resolution tensors.

For every opacity-zero removal, version 6 uses a real-camera causal envelope
instead of selecting only visually clean controls: 24 same-sequence temporal
neighbours (within 16 frames) and eight pose-near real cameras are proposed
before exact per-primitive renderer visibility filtering. Their frozen-model
quality may break ties but can never filter an observation out. At least 12
visible controls are required. This specifically protects against the common
failure where a deletion improves one bad target frame yet damages adjacent
low-quality frames that see the same primitive. The report records each
control's temporal offset, pose distance, and selection stage; the contract
audit rejects a version-6+ accepted removal without this envelope.
`legacy_metric_nearby` is retained only to reproduce historical diagnostics and
cannot be combined with `--apply-approved`.
When comparing several cluster budgets on the unchanged frozen checkpoint,
`--full-train-prescan-report <prior>/full_train_anomaly_prescan.json` can reuse
that completed report only after verifying the exact 1,487 camera names and
the frozen metric-cache SHA-256. An edited output must receive a fresh
all-train evaluation/prescan; it cannot inherit the baseline diagnosis.

An image-space anomaly mask is never treated as a 3D ROI. `ArtifactROI3D` is
created only after independently coordinate-audited real-image evidence
supports a plane/footprint. When Cambridge's original sparse model retains
tracks but omits per-image feature coordinates, its COLMAP pose frame is first
matched to the frozen 2DGS cameras, then target-visible track points still need
two static real support cameras; SIFT is only a fallback. The ROI stores world
points, plane/slab evidence, and attributable primitive IDs. Its mask is
projected separately into every camera. Virtual cameras are only depth-warp
stability diagnostics, never optimization targets.

The first pass diagnoses and writes counterfactual images, 3D evidence, strict
See3D eligibility masks, and causal sets without modifying a model:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n g4splat python scripts/causal_2dgs_repair.py \
  --source-path /root/MAtCha/output_cambridge/datasets_full/StMarysChurch/train \
  --model-path <frozen-2dgs-model> --iteration 100000 \
  --mask-pickle /mnt/pool/sqy/Cambridge_stdloc/StMarysChurch/processed/masks_with_tree.pkl \
  --metrics-json <frozen-2dgs-model>/evaluation/full_train_fit/ours_100000/rgb_metrics.json \
  --target-image-names-file configs/causal_repair/stmarys_diverse_residual_targets.txt \
  --output-dir /mnt/pool/sqy/<isolated-causal-audit> --replace-output
```

Only a counterfactual-verified `G_remove`, an attributed + real-surface-bounded
`G_update` whose temporary projection to that independent plane improves target
rays without hurting visible clean controls, or real-track-supported `G_create`
can be applied. Add
`--apply-approved` only after inspecting the diagnosis. The optimizer has state
only for selected rows; all other baseline rows are checked bit-identical. The
output model is written only if the diagnosed target rays show a positive local
improvement, fixed-affine real-camera envelope controls stay within tolerance,
and virtual depth/normal probes do not regress. Before every counterfactual, a connected
candidate chain is split into bounded 3-D interventions; shared appearance in
one bad image cannot authorize a facade-length deletion. A missing visible
real-camera envelope is an explicit no-op: it cannot be treated as evidence of zero
side effect. Expected-versus-median renderer depth disagreement is saved as a
weak ranking diagnostic only; it never supplies the independent
Chart/track/plane geometry evidence for an ROI.

Version 7 makes the final target gate match the causal intervention: it uses
only the union of target rays from clusters that actually passed opacity-zero
causality, never unrelated views admitted merely for all-train triage. This is
not a relaxation of cross-view safety—the full real-camera envelope remains in
the same fixed-affine post-edit gate and an empty causal target scope is a
no-op.

See3D is not a geometry branch: it is requested only for
`M_surface & ~M_real & ~M_model`, with geometry frozen and appearance/opacity
as the only allowable updates. An observed real training surface normally has
`M_real`, so the correct result is often no See3D request.

Before promoting any local edit, run the read-only contract audit. It checks
the 1,487-view input count, opacity-zero removals, direct plane-projection
geometry trials, and final write gate:

```bash
python scripts/audit_causal_repair.py \
  --report /mnt/pool/sqy/<causal-run>/causal_repair_report.json \
  --output /mnt/pool/sqy/<causal-run>/contract_audit.json
```

If the gate writes `edited_model`, promotion still requires a paired full
1,487-image train-fit render using the same RGB evaluator and semantic mask as
the frozen baseline.  Compare at sequence-bootstrap level; a local target
gain alone is not a global improvement claim:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n g4splat python 2d-gaussian-splatting/render.py \
  -s /root/MAtCha/output_cambridge/datasets_full/StMarysChurch/train \
  -m /mnt/pool/sqy/<causal-run>/edited_model --iteration 100000 \
  --resolution 2 --data_device cpu --rgb_only --skip_test --skip_mesh \
  --output_dir /mnt/pool/sqy/<causal-run>/evaluation/full_train_fit/ours_100000

CUDA_VISIBLE_DEVICES=0 conda run -n g4splat python scripts/evaluate_render_dir.py \
  /mnt/pool/sqy/<causal-run>/evaluation/full_train_fit/ours_100000 \
  --dataset-path /root/MAtCha/output_cambridge/datasets_full/StMarysChurch/train \
  --mask-pickle /mnt/pool/sqy/Cambridge_stdloc/StMarysChurch/processed/masks.pkl \
  --output /mnt/pool/sqy/<causal-run>/evaluation/full_train_fit/ours_100000/rgb_metrics.json

python scripts/compare_heldout_metrics.py \
  --baseline <frozen-model>/evaluation/full_train_fit/ours_100000/rgb_metrics.json \
  --candidate /mnt/pool/sqy/<causal-run>/evaluation/full_train_fit/ours_100000/rgb_metrics.json \
  --bootstrap-unit sequence \
  --output /mnt/pool/sqy/<causal-run>/evaluation/full_train_fit/comparison_vs_frozen_train_fit.json
```
