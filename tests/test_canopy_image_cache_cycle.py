import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'2d-gaussian-splatting'))
from outdoor.lazy_scene import _ImageLRU


def test_cohort_sized_cache_avoids_cyclic_redecode_without_changing_rgb():
    class Camera:
        def __init__(self, cache, index): self.cache, self.index, self.image, self.decodes = cache, index, None, 0
        def _drop_cached_image(self): self.image = None
        def read(self):
            if self.image is None:
                self.decodes += 1; self.image = torch.full((3, 2, 2), self.index/10.)
            self.cache.touch(self); return self.image
    def cycle(capacity):
        cache = _ImageLRU(capacity); cameras = [Camera(cache, i) for i in range(6)]
        values = [camera.read().clone() for _ in range(3) for camera in cameras]
        return values, sum(c.decodes for c in cameras)
    small, small_decodes = cycle(2); full, full_decodes = cycle(6)
    assert all(torch.equal(a, b) for a, b in zip(small, full))
    assert small_decodes == 18 and full_decodes == 6
