"""Bounded CPU RGB cache for full-frame native-resolution training views."""
from collections import OrderedDict
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from scripts.native_resolution_crop_camera import NativeCropCamera


def file_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


class NativeRGBViews:
    def __init__(self,directory,views,training_indices,*,capacity=16):
        if not 1<=capacity<=64:raise ValueError('Native RGB cache must be bounded')
        self.directory=Path(directory);self.capacity=capacity;self.cache=OrderedDict()
        self.hashes={views[i].image_name:file_digest(self.directory/(views[i].image_name+'.png'))
                     for i in training_indices}

    def get(self,view):
        name=view.image_name
        if name not in self.hashes:raise ValueError('Native fitting camera outside training contract')
        if name in self.cache:
            rgb=self.cache.pop(name);self.cache[name]=rgb
        else:
            path=self.directory/(name+'.png')
            if file_digest(path)!=self.hashes[name]:raise ValueError('Native RGB changed after contract')
            with Image.open(path) as image:
                rgb=torch.from_numpy(np.array(image.convert('RGB'),copy=True)).permute(2,0,1).float()/255
            self.cache[name]=rgb
            while len(self.cache)>self.capacity:self.cache.popitem(last=False)
        h,w=rgb.shape[-2:]
        # Full image only: direct off-center crops are not renderer-equivalent yet.
        return NativeCropCamera(view,rgb,(0,0,w,h))
