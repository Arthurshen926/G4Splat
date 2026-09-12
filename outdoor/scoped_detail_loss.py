"""Keep local proposal fitting from reweighting unrelated scene geometry."""
import torch


def backward_detail_objective(global_loss,local_loss,proposal_parameters):
    parameters=tuple(proposal_parameters)
    if local_loss is None:
        global_loss.backward()
        return
    local_grad=torch.autograd.grad(local_loss,parameters,retain_graph=True,allow_unused=True)
    global_loss.backward()
    for parameter,gradient in zip(parameters,local_grad):
        if gradient is not None:
            if parameter.grad is None:parameter.grad=gradient.detach()
            else:parameter.grad.add_(gradient.detach())
