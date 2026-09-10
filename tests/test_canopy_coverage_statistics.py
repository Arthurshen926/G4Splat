import torch
from scripts.audit_canopy_occluder_coverage import alpha_statistics


def test_low_coverage_bins_are_strict_and_do_not_change_selection():
    values=torch.tensor([[0.,1.e-6,.01,.1,.5,.9]])
    mask=torch.ones_like(values,dtype=torch.bool)
    result=alpha_statistics(values,mask)
    assert [result[k] for k in ('below_1e_6_pixels','below_0_01_pixels',
                              'below_0_1_pixels','below_0_5_pixels','below_0_9_pixels')]==[1,2,3,4,5]
    assert result['pixels']==6 and mask.all()


def test_empty_region_stays_unknown_with_zero_counts():
    result=alpha_statistics(torch.ones(2,2),torch.zeros(2,2,dtype=torch.bool))
    assert all(value==0 for value in result.values())
