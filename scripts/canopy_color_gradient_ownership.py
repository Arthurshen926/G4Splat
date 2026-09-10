"""Training-only color ownership; never changes forward radiance or inference."""
import torch


def backward_canopy_owned_colors(tree_loss, other_loss, *, color_parameters,
                                 optical_geometry_parameters, divisor=1):
    colors = tuple(color_parameters)
    others = tuple(optical_geometry_parameters)
    parameters = colors+others
    if (not colors or not others or divisor <= 0
            or len({id(p) for p in parameters}) != len(parameters)
            or not all(p.requires_grad for p in parameters)):
        raise ValueError('Disjoint trainable color and optical/geometry parameters required')
    # The same native forward graph supplies both losses. Color receives only
    # canopy RGB, whereas opacity/geometry receives the full unchanged objective.
    torch.autograd.backward(tree_loss/divisor, inputs=parameters, retain_graph=True)
    torch.autograd.backward(other_loss/divisor, inputs=others)
