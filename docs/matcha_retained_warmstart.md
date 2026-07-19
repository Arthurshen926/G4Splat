# MAtCha retained-model warm start (experimental)

This path uses a completed MAtCha model as the immutable static-scene baseline,
then lets G4Splat add a small number of See3D-supported Gaussians. It is an
isolated experiment and is not part of the Cambridge main result.

## Motivation

Running the original G4Splat stages from the chart initialization was weaker
than MAtCha retained_v2. Full-frame generated pseudo-views also contaminated the
global plane map and produced large blurred splats. The warm-start path instead:

1. Adapts the retained MAtCha scene and PLY without rerunning SfM or alignment.
2. Renders candidate pseudo cameras from the retained 30k model.
3. Preserves supported rendered RGB and asks See3D to synthesize holes only.
4. Rejects pseudo-views with excessive synthesized area, low visible-region
   sharpness, or an unreliable robust DAV2-to-rendered-depth fit.
5. Builds depth/confidence priors without G4Splat's global plane rewrite.
6. Seeds Gaussians only in accepted synthesized holes and freezes every retained
   baseline Gaussian.
7. Applies low pseudo-view weights and projects each new Gaussian back into
   conservative opacity, scale, position, diffuse-color, and SH bounds after
   every optimizer step.

The adapter is `scripts/prepare_matcha_warmstart.py`. See3D quality filtering is
implemented by `scripts/filter_see3d_views.py`; safe priors are built by
`scripts/build_warmstart_refine_depths.py`. Training options are exposed by
`scripts/refine_free_gaussians.py` and
`2d-gaussian-splatting/train_with_refine_depth.py`.

## StMarysChurch validation

The controlled experiment starts from the exact retained_v2 iteration-30000
PLY (460,299 Gaussians), uses the same 512 dense training views, and evaluates
the same 64 trajectory-heldout cameras. Of four Stage-1 proposals, the quality
gate accepts two. The native constrained 3k refinement adds 2,794 Gaussians
(0.61 percent) and keeps all baseline PLY fields byte-value equivalent.

| Variant | Raw PSNR | Raw SSIM | Dynamic-valid PSNR | Static-valid PSNR |
| --- | ---: | ---: | ---: | ---: |
| MAtCha retained_v2 | 15.8253 | 0.79595 | 17.9502 | 17.8778 |
| G4 warm start, native constrained | 15.8322 | 0.79577 | 17.9544 | 17.8845 |

The change is a non-regression, not a material NVS improvement. The gain is
concentrated in a few cameras covered by the two accepted pseudo-views (up to
0.31 dB per view). The tree-occluded heldout view 00014 is unchanged because it
has no reliable static multi-view support. Filling that view from its heldout RGB
would leak evaluation data; hallucinating the trees into the static field would
also be geometrically incorrect.

An unconstrained warm-start run produced a black rectangular splat in view
00063. The cause was a small set of added Gaussians whose opacity, footprint,
and color escaped their local support. The retained safety settings are:

```text
--freeze_init_ply
--warmstart_reseed_pixel_stride 4
--warmstart_reseed_max_scale 0.30
--warmstart_max_opacity 0.15
--warmstart_max_scale 0.30
--warmstart_max_position_delta 0.15
--warmstart_diffuse_only
--warmstart_clamp_dc
```

The no-reference valid/support audit is effectively unchanged (invalid fraction
about 2.02 percent), which confirms that this Stage-1 experiment does not repair
the existing large unsupported components. Further work should use
artifact-directed camera proposals with clean training-view support. Components
dominated by transient occlusion should be view-gated or modeled by a separate
transient field rather than reseeded into the static Gaussian model.
