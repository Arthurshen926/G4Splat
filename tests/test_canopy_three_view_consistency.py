import numpy as np
import pytest
from scripts.audit_canopy_three_view_consistency import three_view_cycle_error


def test_cycle_composes_flows_at_the_displaced_pixel():
    a=np.zeros((20,30,2),np.float32);a[...,0]=1
    b=np.zeros_like(a);b[...,1]=np.arange(30)[None]*.02
    c=a.copy();c[:,:-1]+=b[:,1:]
    assert three_view_cycle_error(a,b,c)[:,:-1].max()<1.e-6
    c[...,1]+=.7
    assert np.allclose(three_view_cycle_error(a,b,c)[:,:-1],.7,atol=1.e-6)


def test_cycle_rejects_unaligned_arrays():
    with pytest.raises(ValueError):three_view_cycle_error(np.zeros((2,3,2)),np.zeros((3,2,2)),np.zeros((2,3,2)))
