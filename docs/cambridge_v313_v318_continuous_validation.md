# Continuous validation: boundary protection and candidate footprint

User explicitly requested continued work through results, not ending at launch.
Only physical GPU1/GPU2 used. No baseline runs, production export, or rigid edits.

## v313/v314800 completed

Both returncode0 and frozen audits true. Adding training-only, independently
normalized rigid-boundary RGB protection (weight1) versus weight0:

| Mean PSNR delta versus original source | Control | Boundary protection |
|---|---:|---:|
| Tree | +.340039 | +.274502 |
| Interior | +.394092 | +.327283 |
| Tree boundary | +.203759 | +.140877 |
| Rigid | -.007586 | -.006149 |
| Rigid-side tree boundary | -.001191 | -.001037 |

738 rigid-side boundary improves from -.393824 to -.289323 versus source,
but mean boundary advantage is only+.000153 dB and tree loses.065537 dB.
This loss reweighting is not adopted as default; geometry coverage remains
the central unresolved issue. Reports compare_v313_v314_0400/0800.json.

The helper affects only the rigid-side training mask and immutable reference;
it does not create an inference gate or opacity target. Native finite-difference
tests confirm correction acts on foreground occluders and not hidden leaves.
Old gradient-balance audit now rejects auxiliary-loss configurations it cannot
account for, rather than silently omitting their terms. Historical zero-auxiliary
audits remain valid. This is a diagnostic scope fix, not evidence of a new
production renderer defect.

## Independent spatial and size learning

New diagnostic SpatialFootprintCandidates adds bounded isotropic size learning
to spatial candidates. Sizes range half through twice initialization; positions
remain bounded in the ORIGINAL sigmas. The size variable cannot shift the center
or expand the permitted displacement. Original tree/building geometry unchanged.
No post-step optical-mass retraction or forced enlargement is used.

v3168-step CUDA smoke completed with frozen audit true and nonzero position,
size, opacity gradients and actual opacity updates. This is not an efficacy test.
Training and saved-model auditor reconstruct the same class; unit state roundtrip
and independent-Jacobian tests pass. CPU suite1336 passed,23 skips,15 warnings;
native rasterizer22 passed (including boundary gradient test).

v317/v318800 are matched controls onGPU1/2. Both use the v313 recipe:557514
fixed-depth candidates, joint persistent optics with source SH step.05,
position radius8, guard3, boundary extra0, feasibility auxiliary0. Only changed
argument is spatial_footprint_log_radius0 versus ln2. Evaluations every400.
Archive v317_v318_joint_spatial_footprint_sources.tar.gz. Await actual comparison
before adoption. Native660 leakage and738/909 building damage remain mandatory
diagnostics; no blanket opacity target and no forced new verified geometry.

At400, v318 versus matched v317: tree+.071529, interior+.085406,
tree-boundary+.053187, rigid+.000263, rigid-side boundary+.006720 dB.
Versus source, tree+.302596.738 boundary is nevertheless .072280 dB worse
than control (-.362848 versus source). No adoption based on average gain.
The saved candidate state has size-ratio quantiles [min,p10,p50,p90,max]
[.545919,.661425,.782237,1.181242,1.986854];102938 rows grow>1.01 and446414
shrink<.99. Most rows shrinking does not establish actual ROI coverage; added
read-only contribution-weighted size summaries for the final native audits.
This audit addition has a CPU unit test and does not change either live run.
