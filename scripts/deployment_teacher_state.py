"""Lean deployment metadata; full resumable checkpoints are not changed."""


def deployment_surface_capture(capture):
    """Keep the native positional layout and every rendering/ownership value."""
    if len(capture) not in (12,13):
        raise ValueError('Unexpected native surface capture schema')
    result=list(capture)
    # Native inference restores fields0:7 and optional metadata12. The Adam
    # state at10 is never read by that path and cannot make this export resume.
    result[10]={}
    return tuple(result)


def deployment_birth_audit(accumulator):
    """O(1) counters, without materializing the training-only candidate pool."""
    return dict(training_candidate_pool_exported=False,
                total_proposals=int(accumulator.total_proposals),
                total_births=int(accumulator.total_births),
                pending_midpoint_cells=len(accumulator.cells),
                pending_visual_hull_cells=len(accumulator.visual_hull_cells),
                consumed_first_hit_witnesses=len(accumulator.consumed_first_hit_witnesses))
