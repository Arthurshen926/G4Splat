from types import SimpleNamespace
import numpy as np
import pytest
from scripts.native_resolution_crop_camera import crop_intrinsics


def test_crop_rays_equal_original_native_rays():
    c=SimpleNamespace(image_width=640,image_height=360,focal_x=550.,focal_y=552.,cx=319.,cy=181.)
    box=(900,600,1540,960);k=crop_intrinsics(c,(1080,1920),box)
    for x,y in [(0,0),(320,180),(639,359)]:
        ray=((x+.5-k['cx'])/k['fx'],(y+.5-k['cy'])/k['fy'])
        full=((x+900+.5-c.cx*3)/(c.focal_x*3),(y+600+.5-c.cy*3)/(c.focal_y*3))
        assert np.allclose(ray,full,atol=1e-12)
    assert c.cx==319. and k['width']==640


def test_invalid_crop_rejected():
    with pytest.raises(ValueError):crop_intrinsics(None,(1080,1920),(-1,0,10,10))
