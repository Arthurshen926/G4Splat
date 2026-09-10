"""Photometric ownership for tree-only additions to an immutable background."""


def candidate_rgb_loss(rgb, target, reference, regions, *, noncanopy_target):
    if noncanopy_target not in ('source_reference', 'legacy_ground_truth'):
        raise ValueError('Explicit candidate RGB ownership policy required')
    if rgb.shape != target.shape or any(mask.shape != rgb.shape[1:] for mask in regions.values()):
        raise ValueError('Aligned RGB and semantic regions required')
    if noncanopy_target == 'source_reference' and (reference is None or reference.shape != rgb.shape):
        raise ValueError('Immutable original scene RGB required outside canopy')
    tree_error = (rgb-target).abs().mean(0)
    background_error = ((rgb-reference.detach()).abs().mean(0)
                        if noncanopy_target == 'source_reference' else tree_error)
    return sum(((tree_error if key == 'tree' else background_error)*regions[key]).sum()
               /regions[key].sum().clamp_min(1) for key in ('tree', 'rigid', 'sky'))
