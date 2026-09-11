"""Pixel-index camera for legacy ray helpers, never a rendering camera override.

Native ndc2Pix is ((ndc+1)*size-1)/2. With the unchanged calibrated projection,
pixel array index x corresponds to (x+.5-cx)/fx. Existing diagnostic ray and
photometric helpers use (x-cx)/fx instead. This narrow proxy converts their cx/cy
to array-index coordinates; it must only be passed to those helper functions.
"""


class NativePixelCamera:
    def __init__(self, camera):
        if isinstance(camera,NativePixelCamera):
            raise ValueError('Pixel-center conversion must not be applied twice')
        self.camera=camera
        self.cx=float(camera.cx)-.5
        self.cy=float(camera.cy)-.5

    def __getattr__(self,name):
        return getattr(self.camera,name)
