# Camera-constrained observation reassociation

This phase addresses the attachment's observation-association gap, not another
global opacity/scale sweep. CPU-only; GPU0/1/2 untouched. No model replacement.

Implemented deterministic source-anchored triplet consensus with at least three
observations, final native1.5pixel reprojection check, refit and support rebinding.
v521 same845conditional proposals:1142forward,273roundtrip,40three-view,
4triangulated. Identical accepted count to v510b; outlier poisoning of the joint
fit is not demonstrated as the dominant bottleneck. Do not claim quality gain.

Added known-camera epipolar constraint before local patch selection. Same5pixel
search radius, photometric thresholds, roundtrip1pixel, triangulation1.5pixel;
epipolar band1.5native pixels. This changes candidate selection and ambiguity
competition, so it is not proof of correctness merely because count increases.

v522:1281forward,467roundtrip,104three-view,54triangulated. Source observations
span13training views; all observations32training views, no regression-view use.
Maximum inter-ray angle min/median/max1.980/8.728/27.693degrees. Maximum accepted
reprojection1.484638pixels. Prior-to-triangulated displacement min/median/max
0.008276/0.121908/2.589415scene units. Large deviations need visibility checks,
not automatic acceptance or arbitrary clipping back to the monocular prior.

4CPU tests passed: measured-ray triangulation, textured/flat matching,
outlier rejection and rotated-camera epipolar geometry with pixel-center offsets.
After v522 completed, extracted the identical line formula into a tested helper;
no GPU training or model ingestion has been launched.

Limitations: integer tiny-patch matching still weak for repetitive railing and
viewpoint changes. Native normals/scales in packet remain prior measurements,
not newly validated shape. These54points have NOT been inserted into a model.
Visibility/actual surface-prefix support, locally grouped geometry construction,
canopy-specific correspondence and final rendering improvement remain pending.
Do not launch another optics-only54point run and call it structural repair.

Artifacts: `audit_v521_consensus_association/audit.json` and
`audit_v522_epipolar_association/audit.json` under the Cambridge MoGe run root.
