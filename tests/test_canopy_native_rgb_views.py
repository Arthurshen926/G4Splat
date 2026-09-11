from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from scripts.canopy_native_rgb_views import NativeRGBViews


def test_native_rgb_rejects_nontraining_or_modified_image(tmp_path):
    path=tmp_path/'a.png';Image.fromarray(np.zeros((4,4,3),dtype=np.uint8)).save(path)
    view=SimpleNamespace(image_name='a');source=NativeRGBViews(tmp_path,[view],[0])
    with pytest.raises(ValueError,match='outside training'):
        source.get(SimpleNamespace(image_name='b'))
    Image.fromarray(np.ones((4,4,3),dtype=np.uint8)).save(path)
    with pytest.raises(ValueError,match='changed'):source.get(view)


def test_native_rgb_capacity_is_bounded(tmp_path):
    with pytest.raises(ValueError,match='bounded'):NativeRGBViews(tmp_path,[],[],capacity=1000)
