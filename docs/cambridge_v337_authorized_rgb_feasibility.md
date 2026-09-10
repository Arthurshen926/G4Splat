# Explicitly authorized RGB-feasibility control

User explicitly approved the controlled RGB occlusion feasibility auxiliary
on2026-09-10 in reply to the asynchronous question. This supersedes the earlier
pending approval; it does not authorize a fixed opacity target, depth truth,
building edits, inference masks, or unvalidated production replacement.

Existing helper was read and checked: black foliage radiance while retaining
the actual geometry, opacity, ordering and visibility; no sky term. Penalize
only positive channel-wise excess of transmitted rigid-surface radiance over
observed RGB, squared and normalized within training canopy pixels. Once below
the observation, gradient is zero. It is a necessary radiance condition, not a
proof that source geometry or appearance is correct. Frozen rigid appearance
errors may also cause excess; building and difficult-view validation remains
essential. Ordinary full RGB/background protection stays enabled.

v337 scene smoke8 launchedGPU2 with weight1, full557514 candidates, standard
all-RGB-color gradients, reference cache384, source opacity LR.02, existing
XYZ/size/photometric recipe. GPU1 continues v334 color-ownership experiment.
Archive:v337_authorized_rgb_feasibility_smoke_sources.tar.gz. Synthetic tests
already establish zero noncanopy/hidden-leaf positive authority and preserved
native extinction gradients. Scene smoke must still pass before full pairing.

Use the same color-ownership policy for both next arms; only auxiliary weight0
versus1 may differ. Do not confuse that comparison with the independent
color-gradient routing experiment.

v337 completed returncode0; frozen checks pass, opacity/XYZ/size updated,
step8 feasibility loss.000146067, peak allocated VRAM4.63GiB. At the time of
launching the follow-up, v334400 had only+.005631 mean tree dB over v333, not a
material breakthrough. Keep the established all-RGB-color policy for this first
auxiliary test and reuse the exact same-code/config completed v333 zero-weight
control. Main SHA was explicitly checked against the current file.

v338 weight1/800 images/eval400 launchedGPU2. Only output and
surface_rgb_feasibility_weight differ from v333. Other training, candidate,
camera and source contracts must be checked before matched reports. Archive:
v338_authorized_rgb_feasibility800_sources.tar.gz. No production adoption.

After the v336 read-only audit releases GPU1, also test weight4 (the preexisting
bounded maximum) with the same v333 weight0 control and v338 weight1 arm. This
is a single-variable strength comparison, not a new opacity/depth target. Keep
the all-RGB-color policy: completed v334 only added.001962 mean tree dB. No
claim that stronger auxiliary weight is better before multiview RGB/occlusion
and protected-building results are available.

v339 weight4/800 images/eval400 launchedGPU1 after v336 completed normally.
Main SHA matches the v333 zero-weight control; same source archive as v338.
v338 weight1 continuesGPU2. Neither uses canopy-only-color gradients, inference
masking, fixed opacity target or geometry pseudo-label. Both must finish their
matched validation before any recommendation to retain the auxiliary.

At400, weight1 adds tree+.020100 dB versus weight0, rigid-.001800 and
hard-.001426. Weight4 adds tree+.062820, rigid-.007088 and hard-.016854.
Absolute tree gains are+.314869/.334970/.377689 for weights0/1/4. This shows
a strength response but also increasing background cost, not a solved canopy.
Reports:compare_v333_v338_0400.json and compare_v333_v339_0400.json. Continue
the bounded800-image tests and native transmitted-building audits.

v338 weight1 completed800 normally: tree+.508568 versus original, an additional
+.042482 over weight0; rigid-.009259, hard-.005592 (extra-.003155/-.007752).
Report:compare_v333_v338_0800.json. v340 read-only transmitted-radiance audit
launchedGPU2. v339 weight4 still finishing. No auxiliary configuration adopted.

v339 weight4 completed800 normally, both paired frozen checks pass. Additional
tree+.119254 over weight0 (absolute+.585340), rigid-.011766, hard-.027460.
738 hard is-.943106 versus original, another-.376899 versus control;909 hard
another-.301468. Do not adopt stronger weight based on tree mean. Report:
compare_v333_v339_0800.json. v341 transmitted-radiance audit launchedGPU1.

v340 weight1 audit completed:660 lower ROI PSNR12.825936->12.896877 versus
weight0; transmitted surface contribution.394646->.389727, volume.588323->
.593631. Floor-MSE.037895825->.037090980. This is real but small occlusion gain,
not a solution. Keep the current source/model unchanged pending all validation.

v341 weight4 audit completed:660 ROI PSNR12.996828, surface contribution.380166,
volume.603919 and floor-MSE.035613421. Compared with weight0 this is+.170892 dB
and-.014480 surface contribution, but738/909 building damage worsens materially.
The auxiliary has real limited optical effect; do not declare the persistent
see-through problem solved or adopt weight4. The next fixed-budget sampling
pair deliberately uses weight0 in both arms to isolate a different bottleneck.
