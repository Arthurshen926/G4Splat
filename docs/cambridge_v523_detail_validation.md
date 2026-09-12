# Reassociated detail scale consistency and held-out-ray audit

Implemented source-angular-footprint rebinding after triangulation:
sigma_new = sigma_old * z_new / z_old. Previous exports updated positions and
source depth but retained the old world sigma. This is an inconsistency in
the experimental association export, NOT an established production canopy bug:
those exports had never been inserted into the model. Prior sigma retained
for provenance; normals still monocular priors, not verified local surface shape.

v523 rerun with identical camera-constrained matching reproduces54tracks.
5matching/triangulation/scale CPU tests passed. No GPU use or training launched.

v524 leave-one-ray-out audit:51/54tracks have exactly3observations and cannot
support a three-ray fit plus held-out observation;3tracks have4observations.
Of12held-out projections,8meet1.5native pixels. Only row637passes all4;
row585passes1/4,row594passes3/4. Correspondence selection still used all
observations, so this is not independent matching confirmation or a blind test.
Do not interpret untestable51tracks as wrong; the evidence is insufficient.

Conclusion: constrained patch matching improved association count, but a robust
stable-detail reconstruction path remains incomplete. No visible reconstruction
gain established. Do not train a tiny optical-only suffix from these points and
claim the attachment's structural repair loop is complete. Next association work
needs more measured support/structure-aware correspondence, not threshold loosening.

Artifacts under Cambridge MoGe run root:
`audit_v523_rebound_association/` and `audit_v524_detail_heldout.json`.
