import pytest
from scripts.canopy_selected_optical_signal import summarize_signals


def test_fixed_signal_components_and_visibility_are_distinct():
    rows=[dict(total_gradient=[-2.,0.],weighted_floor_gradient=[-3.,0.],
               responsibility=[[1.,1.,0.,0.],[1.,0.,1.,0.]]),
          dict(total_gradient=[1.,0.],weighted_floor_gradient=[0.,0.],
               responsibility=[[0.,0.,0.,0.],[0.,0.,0.,0.]])]
    result=summarize_signals(rows)
    assert result['total_gradient_sum']==[-1.,0.]
    assert result['rgb_and_guard_gradient_sum']==[2.,0.]
    assert result['nonzero_gradient_views']==[2,0]
    assert result['contributing_views']==[1,1]


def test_malformed_signals_fail_closed():
    with pytest.raises(ValueError):
        summarize_signals([dict(total_gradient=[float('nan')],weighted_floor_gradient=[0.],
                               responsibility=[[1.,0.,0.,0.]])])
