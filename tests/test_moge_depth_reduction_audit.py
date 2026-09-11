import numpy as np
import pytest
from scripts.audit_moge_depth_reduction import tile_statistics


def test_foreground_background_blend_detected():
    d=np.array([[1.,9.],[1.,9.]])
    s=tile_statistics(d,np.ones_like(d,dtype=bool),(1,1))
    assert s['mean'].item()==5
    assert s['minimum'].item()==1
    assert s['log_span'].item()==pytest.approx(np.log(9))


def test_invalid_depth_does_not_create_false_foreground():
    d=np.array([[0.,9.],[np.nan,9.]])
    s=tile_statistics(d,np.ones_like(d,dtype=bool),(1,1))
    assert s['valid_fraction'].item()==.5
    assert s['mean'].item()==9
    assert s['log_span'].item()==0
