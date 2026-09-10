"""CPU LRU for a caller-guaranteed immutable reference renderer."""
from collections import OrderedDict
import torch


class FrozenReferenceCache:
    def __init__(self, capacity=0):
        if not isinstance(capacity, int) or not 0 <= capacity <= 512:
            raise ValueError('Bounded CPU reference cache required')
        self.capacity = capacity
        self.values = OrderedDict()
        self.hits = self.misses = 0

    @torch.no_grad()
    def get(self, key, render, device):
        if key in self.values:
            self.hits += 1
            value = self.values.pop(key)
            self.values[key] = value
        else:
            self.misses += 1
            value = render().detach()
            if value.ndim != 3 or value.shape[0] != 3 or not value.is_floating_point():
                raise ValueError('Unquantized CHW floating-point reference RGB required')
            if not self.capacity:
                return value
            value = value.to(device='cpu', copy=True)
            self.values[key] = value
            while len(self.values) > self.capacity:
                self.values.popitem(last=False)
        # Never expose the stored CPU tensor for in-place changes by consumers.
        return value.to(device=device, copy=True)

    def stats(self):
        return dict(capacity=self.capacity, entries=len(self.values), hits=self.hits, misses=self.misses,
                    bytes=sum(v.numel()*v.element_size() for v in self.values.values()))
