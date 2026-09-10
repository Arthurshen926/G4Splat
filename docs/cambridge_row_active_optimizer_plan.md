# Isolated nonzero-gradient-row Adam hypothesis

Not connected to the training script yet. Existing runs remain standard Adam.
Local installed torch.optim.Adam code confirms that dense zero gradients still
decay moments, advance the tensor-wide clock and may move parameters through
stored momentum. This is STANDARD ADAM, not evidence of an implementation bug.
Current sparse observations motivate testing a different update policy, not
declaring this the cause of canopy failure.

Proposed RowActiveAdam only updates a row if any gradient component in that
row is nonzero. Otherwise parameters, both moments and per-row clock remain
unchanged. This is NOT an exact visibility test: cancellation or an already
matched observation can also give zero gradients. It changes intermittent-row
optimization and potentially changes fit/generalization in either direction.
No new geometry, evidence authority, loss or opacity target is introduced.

Independent helper has checks for finite gradients before ANY group mutates,
per-row bias correction, and checkpoint-clock device/type restoration. CPU
tests compare always-active rows against standard Adam, exact inactive-state
preservation, saved-state/future-step replay, and nonfinite fail-before-update.
Needs evidence and a separate same-budget paired scene experiment before any
retention; do not add it to the running absolute/relative radiance comparison.

Four independent CPU optimizer tests pass. Still NOT integrated or running.
Extended the candidate-only fixed-state gradient audit to record per-view rows
with exactly zero current gradient but nonzero saved Adam momentum, including
evaluation-contribution-weighted fractions. All views use the same final state
and same saved moment: not a historical update replay and not proof of harm.
Source-position checkpoints can now use this candidate-only audit with restored
source positions held fixed; auxiliary-loss checkpoints remain rejected because
that gradient auditor does not model those extra losses.

Fixed an audit-only authority bug: candidate_gain0 was previously accepted but
the gradient replay always used gate1. It now fails closed instead of enabling
disabled candidates. Existing v328 and current candidate-gradient experiments
use gain1; no earlier reported comparison is invalidated by this guard. Added
regression test. No production trainer/renderer change.
