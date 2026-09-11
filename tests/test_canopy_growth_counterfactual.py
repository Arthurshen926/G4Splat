import pytest
import torch
from scripts.canopy_growth_counterfactual import eligible_shrink, shrink_probe


def evidence():
    return dict(report=dict(snapshot_sha256='x',training_views=[1,2],excluded_views=[3]),
                scale_code=torch.ones(3),tree_gradient=torch.tensor([1.,-1.,1.]),
                risk_gradient=torch.ones(3),tree_nonzero_views=torch.tensor([2,2,1]),
                risk_nonzero_views=torch.tensor([2,2,2]))


def test_scope_and_sign_selection():
    data=evidence();code=data['scale_code'].clone()
    assert eligible_shrink(data,code,'x',[1,2],[3]).tolist()==[True,False,False]
    with pytest.raises(ValueError):eligible_shrink(data,code,'y',[1,2],[3])
    with pytest.raises(ValueError):eligible_shrink(data,code+1,'x',[1,2],[3])
    with pytest.raises(ValueError):eligible_shrink(data,code,'x',[1,3],[2])


@pytest.mark.parametrize('fraction',[0.,.25,1.])
@pytest.mark.parametrize('fail',[False,True])
def test_finite_step_and_exception_restoration(fraction,fail):
    code=torch.nn.Parameter(torch.tensor([2.,1.,-1.]))
    original=code.detach().clone();mask=torch.tensor([True,False,False])
    try:
        with shrink_probe(code,mask,fraction):
            assert code[0]==2*(1-fraction)
            assert torch.equal(code[1:],original[1:])
            if fail:raise RuntimeError('render failed')
    except RuntimeError as exc:
        assert fail and str(exc)=='render failed'
    assert torch.equal(code,original)
