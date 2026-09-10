from types import SimpleNamespace
import torch
from outdoor.canopy_deformation_diagnostic import apply_dc_learning_rate_multiplier_


def test_dc_update_matches_separate_adam_learning_rate_without_changing_higher_sh():
    torch.manual_seed(34)
    features=torch.nn.Parameter(torch.randn(4,3,3))
    reference_dc=torch.nn.Parameter(features[:,0].detach().clone())
    reference_high=torch.nn.Parameter(features[:,1:].detach().clone())
    actual=torch.optim.Adam([features],lr=.0025,eps=1.e-15)
    reference=torch.optim.Adam([{'params':[reference_dc],'lr':.025},
                               {'params':[reference_high],'lr':.0025}],eps=1.e-15)
    eligible=torch.tensor([True,False,True,False]);initial=features.detach().clone()
    for _ in range(4):
        grad=torch.randn_like(features)*eligible[:,None,None]
        features.grad=grad.clone();reference_dc.grad=grad[:,0].clone();reference_high.grad=grad[:,1:].clone()
        before=features[:,0].detach().clone()
        actual.step();reference.step()
        apply_dc_learning_rate_multiplier_(SimpleNamespace(features=features),eligible,before,10.)
        torch.testing.assert_close(features[:,0],reference_dc,atol=2.e-6,rtol=2.e-6)
        torch.testing.assert_close(features[:,1:],reference_high,atol=0,rtol=0)
    torch.testing.assert_close(features[~eligible],initial[~eligible],atol=0,rtol=0)
