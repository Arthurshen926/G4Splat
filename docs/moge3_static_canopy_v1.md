# MoGe3 + static canopy v1

Branch: `codex/moge3-static-canopy-v1`

This branch keeps the existing MASt3R correspondence graph, MAtCha chart
atlas, bounded plane factors, native rigid 2DGS and static foliage 3DGS.  It
replaces production DAV2 monocular evidence with source-separated MoGe3
metric depth/normals.  Raw SfM points remain disabled by the default
`mast3r_only` geometry policy.

## Non-negotiable geometry contracts

- Cambridge image identity, pose and pixel intrinsics are authoritative.
- MoGe3 receives the true horizontal FOV computed from Cambridge `fx`.
- MoGe3's public normalized point-map is not calibrated geometry: v3 fixes
  its principal point to `(0.5, 0.5)`. G4Splat recomputes every camera-space
  point from metric camera-z depth and exact pixel `fx, fy, cx, cy`.
- One immutable view archive stores metric depth, exact-K point-map, direct
  normal, depth-derived normal, valid masks, every refinement depth, and
  refinement stability.
- The all-view index hashes every view, input image, camera record, model
  checkpoint and generator implementation. Missing/mixed views fail closed.
- Every view also records and validates its checkpoint digest, MoGe commit,
  refinement count, resolution level and FP16/FP32 mode. A cache cannot reuse
  a view produced under another inference contract.
- MoGe3 and legacy DAV2 cannot be registered in the same Evidence Store.
- MoGe3-to-Cambridge gauge uses one positive scene-wide scale. Per-camera
  affine depth fitting is forbidden.

## Static tree ownership

- A MoGe3 tree pixel may create a single-camera measured proposal only when
  its depth is in front of a known rigid hit. Behind-rigid evidence is
  `unknown`, never positive foliage mass.
- Where rigid depth is absent, the tree hit is retained as spatially
  uncertain single-camera evidence; it is not treated as free space.
- Single-camera rows remain exact-camera visible measured proposals.
  Persistent deployment requires the existing multi-camera voxel fusion and
  verification lifecycle.
- MoGe3 rows have a 1.10x scale ceiling, bounded opacity, and refinement-based
  depth covariance. They cannot grow into an unconstrained low-frequency fog.
- Sequence identity is evidence provenance only. It is never a deployment
  visibility switch for persistent static foliage.

## Current implementation

- `outdoor/moge3_evidence.py`: pure NumPy exact-K schema, math, validation,
  atomic view/index writers and content verification.
- `scripts/build_moge3_evidence.py`: official MoGe3 v3 offline generator,
  loaded lazily in a separate Python 3.10+ environment.
- `scripts/supervise_moge3_evidence.py`: detached, run-locked multi-GPU
  sharding with a frozen command contract, atomic heartbeat and final index
  pass. Completed per-view archives survive interruption and are validated
  before reuse.
- `scripts/build_hybrid_teacher_evidence.py`: registers `moge3_index` as a
  distinct metric source, registers the source-separated Chart base and
  compact runtime cache, and rejects a mixed MoGe3/DAV2 store.
- `scripts/build_moge3_chart_base.py`: fits one cross-view scene scale and
  emits immutable MAtCha, direct-MoGe3 and confidence-adaptive Chart bases.
- `scripts/build_moge3_runtime_cache.py`: converts the 149 GB full-resolution
  source cache once into content-addressed 640x360 mmap arrays. Training never
  repeatedly decompresses 100 MB source archives per step.
- `outdoor/training_evidence.py`: exposes named `moge3_*` depth/normal/
  refinement fields from the mmap cache; it never overwrites legacy
  `mono_depth`.
- `outdoor/role_aware_initialization.py`: global scale recovery, exact-K tree
  back-projection, rigid-front clipping and measured-to-persistent lifecycle.
- `scripts/run_cambridge_hybrid_teacher.py`: default
  `moge3_static_quality` profile; DAV2 seed and temporal witness paths are
  zero/disabled and no legacy cache is auto-discovered.
- Teacher protocol:
  `cambridge_native_hybrid_teacher_v113_moge3_thin_hit_interval`.
- Native mixed CUDA exposes intrinsic volume transmittance before and inside
  an arbitrary per-pixel target-depth interval. Surface rows and volume behind
  the upper bound cannot receive gradients from this query.
- Canonical MoGe3 supervision uses two owner-isolated volume-only loss passes
  plus one all-owner read-only coverage query: pre-hit volume receives
  negative-only opacity gradients, while hit-interval growth is exact-camera
  measured-detail owned. Forward and live owner gates are identical for each
  loss pass.
- Global scene-scale uncertainty only weakens reliability and moves the
  negative pre-hit bound conservatively. Positive hit/birth support uses
  local refinement uncertainty with a strict 0.40 m metric half-width; one
  monocular first hit can no longer authorize a multi-metre crown segment.
- A missing hit with no live primitive is not ignored. It emits a bounded
  exact world-ray interval to the existing static birth accumulator. A low-mass
  leaf proposal is appended only after two independent canonical cameras agree
  in one voxel; it then remains subject to the normal verification lifecycle.
- Rigid pretraining now consumes the confidence-adaptive MoGe3 Chart base,
  auxiliary depth and normal factors. Its one scale comes from the immutable
  MAtCha/MoGe3 Chart consensus. The later trained-rigid foliage rebuild must
  agree with that same scale within the fixed log-scale radius; a second,
  incompatible foliage gauge fails closed instead of silently mixing units.

## Cache generation

The generator environment is intentionally independent from the Python 3.9
training environment.

```bash
/mnt/pool/sqy/conda_envs/g4splat_moge3_v1/bin/python \
  scripts/build_moge3_evidence.py \
  --dataset /path/to/fixed_cambridge_dataset \
  --scene-contract /path/to/scene_contract.json \
  --output /path/to/moge3_exact_k_cache \
  --model Ruicheng/moge-3-vitl \
  --device cuda:0 \
  --refine-steps 3 \
  --resolution-level 9 \
  --use-fp16
```

Use `--start` and `--limit` for resumable multi-GPU shards. After all view
archives exist, run once without `--limit`; valid views are reused and the
complete content-addressed index is emitted.

For a formal run, prefer the detached supervisor so terminal loss cannot stop
all shards:

```bash
python scripts/supervise_moge3_evidence.py \
  --dataset /path/to/fixed_cambridge_dataset \
  --scene-contract /path/to/scene_contract.json \
  --output /path/to/moge3_exact_k_cache \
  --python /path/to/isolated_moge3/python \
  --gpus 0 1 2 \
  --model-revision IMMUTABLE_HF_COMMIT \
  --refine-steps 3 --resolution-level 9 --use-fp16
```

## Acceptance

Physical invariants are strict: exact-K reconstruction, all-view identity,
behind-rigid rejection, no DAV2 override, single-camera fail-closed
deployment, bounded proposal scale and source/gradient ownership must pass
100%.

Image quality is evaluated rather than over-gated. The fixed suite must cover
multiple tree-heavy and historically poor cameras and report canopy interior,
boundary, true holes, rigid/tree hard pixels, non-tree quality, edge energy,
opacity/transmittance and view-by-view regressions. A useful change should
improve the broad distribution, not only one showcase frame, but small metric
noise is not a reason to discard a physically correct repair.

## Current validation state and remaining work

- The full 1,487-view exact-K cache is complete. Its index hash is
  `8a2a1cdc9247f184e5c5bf4ba831d4291a779b5dd7005470215d1ab12b60dd5c`.
- The 56-camera Chart artifact accepted scale support from 31 views at one
  scale of `0.8277335148` with log-scale sigma `0.100990`; it provides about
  2.414 million valid MoGe3
  rigid Chart pixels. The earlier five-view `1.3699` foliage pilot is no
  longer treated as an independent authority: the trained-rigid fit must
  agree with the Chart gauge before foliage initialization can publish.
- The v9 Evidence contract is hash-bound to `mast3r_only`, has no DAV2 or
  COLMAP-track artifact, disables SfM coverage, and exports one unconditional
  static map.
- Native target-depth forward/backward tests cover pre-hit, hit and behind
  separation, finite-difference opacity gradients, empty queries and the
  ordinary mixed renderer path.
The remaining causal sequence is:

1. rebuild the rigid-depth-calibrated foliage initialization under the v112
   Chart-scale authority contract;
2. run a short real-training smoke and prove source gradient, post-policy
   gradient and post-Adam parameter delta agree by role;
3. run matched clean A/B checkpoints and the multi-view hard suite for canopy
   interior, boundary, true holes and building-hard pixels;
4. only then start the formal 30k training. The current implementation and
   tests establish mechanism correctness; they do not yet claim the final
   visual transparency problem is solved.
