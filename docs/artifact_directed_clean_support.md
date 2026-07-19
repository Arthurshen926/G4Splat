# Artifact-directed clean-support repair (experimental)

This experiment is isolated from the retained MAtCha result. It diagnoses a
render artifact, finds posed clean-support training views, edits only attributed
bad Gaussians, and optionally appends low-opacity Gaussians from verified local
geometry. It does not rerun SfM or chart alignment and does not modify the source
retained checkpoint.

## StMarysChurch case

The source is the retained_v2 iteration-30000 checkpoint with 460,299 Gaussians.
The selected training-view component is `seq2/frame00109`, component 1. Although
the detector component covers 34.55 percent of the image, most of it is a
near-field tree/occlusion mixture. Multi-view feature and depth checks support
only a 1.835 percent window patch as static wrong geometry.

The accepted patch has:

- 15 plane inliers from 18 triangulated points;
- 0.00907 normalized median plane residual;
- 99.43 percent depth-supported projective RGB coverage;
- 0.00170 median relative error against the clean-support plane depth.

The generic See3D output contained repeated window bars and blue stripe
artifacts, so it is retained only as a rejected comparison. The accepted pseudo
RGB reprojects real clean-support training images onto the verified local plane.

## Controlled results

All baseline Gaussians are frozen exactly. The conservative edit suppresses
three differentiably attributed large-footprint Gaussians. Projective reseeding
adds 261 diffuse-only Gaussians (0.057 percent), disables densification, caps
opacity at 0.15, scale at 0.30, and displacement at 0.15, then runs 3,000
iterations with 512 real dense views and low pseudo-view weights.

| Variant | Target component PSNR | Target full PSNR | Pseudo clean-depth mean error |
| --- | ---: | ---: | ---: |
| Retained v2 | 9.9722 | 13.1618 | 0.4879 |
| Conservative edit, k=3 | **10.2669** | **13.3685** | 0.3957 |
| Conservative edit + projective 3k | 10.2648 | 13.3666 | 0.3962 |
| Aggressive edit, k=20 | 10.1131 | 13.2575 | **0.3048** |
| Aggressive edit + projective 3k | 10.1074 | 13.2510 | 0.3063 |

On the same 64 heldout cameras, the k=3 edit is pixel-equivalent to retained_v2
except for one one-level pixel change. Projective 3k changes only 10 views and is
a strict non-regression, but the gain is immaterial:

| Variant | Raw PSNR | Static-valid PSNR | Invalid fraction |
| --- | ---: | ---: | ---: |
| Retained v2 | 15.825318 | 17.877829 | 0.0202223 |
| Conservative edit, k=3 | 15.825318 | 17.877829 | 0.0202223 |
| Conservative edit + projective 3k | 15.825464 | 17.878029 | 0.0202223 |

The aggressive edit improves depth while reducing RGB quality, and reseeding
does not recover it. This rejects the hypothesis that the conservative edit
merely left too much foreground opacity for the new points to contribute.

## Interpretation

The reliable positive result is the conservative opacity edit: it improves the
diagnosed component by 0.295 dB without measurable heldout regression. The
current projective reseed is not retained as a positive improvement. It cannot
repair the visually dominant region because that region is transient tree
occlusion without clean static multi-view support; filling it with a facade or
generated foliage would put unsupported content into the static field.

Use `scripts/evaluate_artifact_guided_model.py` to compare arbitrary Gaussian
checkpoints against the same component, clean-support views, and pseudo clean
depth. The detailed report and render triplets are under
`artifact_model_evaluation/` in the isolated experiment directory.
