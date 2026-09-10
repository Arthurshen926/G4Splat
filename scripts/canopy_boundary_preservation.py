"""Training-only rigid-side boundary RGB preservation; never an inference gate."""


def boundary_rgb_preservation(rgb, reference, rigid, boundary):
    if rgb.shape != reference.shape or rigid.shape != rgb.shape[1:] or boundary.shape != rigid.shape:
        raise ValueError('Aligned immutable reference and region masks required')
    if rigid.dtype != boundary.dtype or str(rigid.dtype) != 'torch.bool':
        raise ValueError('Boolean region masks required')
    if (boundary & ~rigid).any():
        raise ValueError('Boundary authority must remain on the rigid side')
    error = (rgb-reference.detach()).abs().mean(0)
    return (error*boundary).sum()/boundary.sum().clamp_min(1)
