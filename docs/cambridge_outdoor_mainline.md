# Cambridge outdoor structural mainline

The new mainline is an explicit replacement for the legacy fixed-`n24`
G4Splat experiment identity.  It keeps all real database images in RGB and
topology supervision, but treats Charts only as high-confidence geometry
anchors.

Run the complete St Marys Church pipeline with:

```bash
conda run -n g4splat python scripts/run_cambridge_outdoor_mainline.py \
  --scene StMarysChurch \
  --gpu 0
```

The default output root is
`/mnt/pool/sqy/G4Splat_runs/cambridge_outdoor_mainline_v1`.

The runner is restartable by stage:

```bash
# audit/QC/camera and task-semantic contracts
python scripts/run_cambridge_outdoor_mainline.py --phase prepare

# fixed-camera MASt3R alignment and hard gate
python scripts/run_cambridge_outdoor_mainline.py --phase frontend

# static structure graph and dynamic Global/Local Chart selection
python scripts/run_cambridge_outdoor_mainline.py --phase select

# bounded plane refinement followed by inverse-depth fusion
python scripts/run_cambridge_outdoor_mainline.py --phase planes

# 7k structural screen, then 40k all-real refinement and evaluation
python scripts/run_cambridge_outdoor_mainline.py --phase mainline
```

The mainline records these artifacts under `prepared/<scene>` and the run's
`mast3r_sfm` directory:

- `scene_manifest.json`: immutable calibrated-camera/data contract;
- `task_semantics_outdoor_v1.json`: distinct RGB, matching, plane, depth and
  densification semantic roles;
- `static_structure_graph.npz`: static multi-view structure units, local
  blocks and structural view graph after the hard gate;
- `quality_aware_selection/structural_chart_selection.json`: final dynamic
  Chart set selected from structure coverage and true triangulation angles;
- `inverse_depth_fusion/`: fused depth, inverse-depth variance, source
  bitmask and supporting-source count for every Chart;
- `evaluation/trajectory_heldout/`: no-appearance-fit held-out trajectory
  rendering and static metrics.

The new defaults freeze calibrated poses and per-camera intrinsics, require
three real Chart views for global planes, use all-real `dense_only` topology
statistics with spatial-block-balanced real-camera sampling, leave
`max_plane_abs_depth` unset, disable See3D, and treat a missing tree support
map as explicit unknown/neutral evidence instead of zero support.  The
structural geometry loss consumes the fused inverse-depth variance and source
bitmask: plane/Chart evidence is primary and mono-only evidence is a weak
fallback.  Plane proposals are semantic-gated, retain multiple normal modes,
and use a `64`-pixel absolute lower bound in addition to their relative
structural-support threshold.

The prior `18.756 / .830 / .0821` number is retained as a train-fit diagnostic
baseline only.  The mainline writes a separate held-out trajectory result, so
it cannot be conflated with in-sample reconstruction quality.
