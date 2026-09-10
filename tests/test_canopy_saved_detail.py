import torch
from scripts.audit_canopy_saved_detail import detail_metrics


def test_identical_texture_has_zero_error_and_unit_similarity():
    image=torch.rand(3,24,24,generator=torch.Generator().manual_seed(4))
    result=detail_metrics(image,image,torch.ones(24,24,dtype=torch.bool))
    assert result['gradient_mae']==0
    assert result['gradient_strength_ratio']==1
    assert abs(result['interior_ssim']-1)<1.e-6


def test_flat_prediction_is_not_mistaken_for_good_detail():
    image=torch.rand(3,24,24,generator=torch.Generator().manual_seed(4))
    result=detail_metrics(torch.full_like(image,.5),image,torch.ones(24,24,dtype=torch.bool))
    assert result['gradient_mae']>.1
    assert result['gradient_strength_ratio']==0
    assert result['interior_ssim']<.1
