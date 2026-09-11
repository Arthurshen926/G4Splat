"""Bounded CPU cache sharing one immutable render's RGB and surface alpha.

Caller must bind the key to immutable model and camera state. This changes
neither the rendering model nor training objective; not yet wired into training.
"""
from collections import OrderedDict
import torch


class FrozenReferenceBundleCache:
    def __init__(self,capacity):
        if not isinstance(capacity,int) or not 0<=capacity<=512:
            raise ValueError('Bounded CPU cache capacity required')
        self.capacity=capacity;self.values=OrderedDict();self.hits=0;self.misses=0

    @torch.no_grad()
    def get(self,key,render,device):
        if key in self.values:
            self.hits+=1;packed=self.values.pop(key);self.values[key]=packed
        else:
            self.misses+=1;rgb,alpha=render()
            if (rgb.ndim!=3 or rgb.shape[0]!=3 or alpha.shape!=(1,*rgb.shape[1:])
                    or not rgb.is_floating_point() or rgb.dtype!=alpha.dtype or rgb.device!=alpha.device):
                raise ValueError('Aligned RGB and one-channel surface alpha required')
            packed=torch.cat((rgb.detach(),alpha.detach()),0)
            if self.capacity:
                packed=packed.to('cpu',copy=True);self.values[key]=packed
                while len(self.values)>self.capacity:self.values.popitem(last=False)
        result=packed.to(device=device,copy=True)
        return result[:3],result[3:]

    def stats(self):
        return dict(capacity=self.capacity,hits=self.hits,misses=self.misses,entries=len(self.values),
                    bytes=sum(t.numel()*t.element_size() for t in self.values.values()))
