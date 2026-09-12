import torch
from outdoor.scoped_detail_loss import backward_detail_objective


def test_local_weight_does_not_drive_original_geometry():
    geometry=torch.tensor(2.,requires_grad=True);proposal=torch.tensor(3.,requires_grad=True)
    global_loss=(geometry+proposal)**2
    local_loss=100*(geometry-proposal)**2
    backward_detail_objective(global_loss,local_loss,[proposal])
    assert geometry.grad.item()==10
    assert proposal.grad.item()==210
