# Controlled relative-radiance feasibility objective

Within the user's approved RGB-only auxiliary scope: only positive excess of
native transmitted building RGB above observed training RGB inside canopy.
No fixed opacity, depth target, inference mask or rigid edits. This is an
objective experiment, not a newly discovered historical renderer defect.

Compare squared positive absolute excess with squared positive log ratio:
relu(log(B+epsilon)-log(I+epsilon))^2, epsilon=1/255. B is black-foliage native
surface radiance at ACTUAL geometry/opacity; I is detached observed RGB.
Native extinction/order stays identical. Relative units emphasize large ratios
in dark regions; whether this improves reconstruction is unproven.

To avoid mistaking a global strength increase for better gradient allocation,
compute ONE FIXED scalar per training view from the immutable original source.
It matches initial total absolute gradient w.r.t. surface RGB between the two
objectives (sum over canopy pixels/channels), then remains frozen throughout
training. This does NOT guarantee identical Gaussian-parameter gradient norms,
nor equal gradient strength after the initial source. For initially feasible
views, use the positive local squared-radiance scale so later violations still
receive gradients. Never derive normalization from evaluation ROIs/cameras.

Proposal selection remains the same existing absolute-RGB priority in BOTH
arms. Normalization is separately recorded in the manifest, checked between
paired arms along with hashes. Default domain absolute, default auxiliary0.
For the planned controlled pair keep supported source-position radius2 in both
arms, unchanged source dimensions/buildings/candidate budget; only domain
absolute vs relative differs, common weight1. Neither is production adoption.

CPU4 objective tests pass: initial surface-RGB gradient totals match, relative
distribution differs, targets detached, zero noncanopy/below-observation
gradients, initially feasible views do not disable later constraints. Native
foreground/behind-surface gradient and finite-difference tests are running.
Require scene8 smoke with final checkpoint replay before800-image paired runs.

Native25 tests passed, including both absolute and relative floor finite
differences, negative foreground-logit gradient and exactly zero hidden-leaf
gradient. v359 relative/GPU2 and v360 absolute/GPU1 scene8 smoke launched with
identical priority seeds, source-position radius2, auxiliary weight1 and all
existing building/source-dimension safeguards. Only domain differs. Archive:
v359_v360_relative_radiance_smoke_sources.tar.gz. Full project suite running.

Project1370 passed,26 skipped. Both scene8 smokes completed returncode0 and
frozen checks pass; actual candidate/source optics and bounded source positions
update. Peak allocated4719.3MiB.187 per-training-view normalization coefficients
match exactly, min/median/max.0017073/.0376769/.1722315. Coefficients are fixed,
not recomputed from refined geometry. v361 relative/GPU2 and v362 absolute/GPU1
native checkpoint replays launched before longer comparison. No adoption.

CPU saved-state check confirms exact equality of cloud.xyz, cloud.scales and
cloud.quaternions across smoke arms, plus ray_sampling_audit and photometric
seed audit and fixed normalization coefficients. v361/v362 completed normally;
all48 rigid PSNR replay errors<=1.90735e-6 dB. v363 absolute/GPU1 and v364
relative/GPU2 launched800 images/eval400 after both passed. Common weight1,
source-position radius2. Archive:v363_v364_relative_radiance800_sources.tar.gz.
