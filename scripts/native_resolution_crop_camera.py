"""Exact off-center camera for native RGB crops; no geometry mutation."""
import math


def crop_intrinsics(camera,native_shape,box):
    h,w=native_shape;left,top,right,bottom=box
    if not (0<=left<right<=w and 0<=top<bottom<=h):
        raise ValueError('Crop must be inside native image')
    sx=w/camera.image_width;sy=h/camera.image_height
    return dict(fx=camera.focal_x*sx,fy=camera.focal_y*sy,
                cx=camera.cx*sx-left,cy=camera.cy*sy-top,
                width=right-left,height=bottom-top)


class NativeCropCamera:
    def __init__(self,camera,native_rgb,box):
        from utils.graphics_utils import getProjectionMatrix
        self.camera=camera
        intr=crop_intrinsics(camera,native_rgb.shape[-2:],box)
        self.image_width=intr['width'];self.image_height=intr['height']
        self.focal_x=intr['fx'];self.focal_y=intr['fy'];self.cx=intr['cx'];self.cy=intr['cy']
        self.FoVx=2*math.atan(self.image_width/(2*self.focal_x))
        self.FoVy=2*math.atan(self.image_height/(2*self.focal_y))
        left,top,right,bottom=box
        self.original_image=native_rgb[:,top:bottom,left:right]
        self.gt_alpha_mask=None
        self.projection_matrix=getProjectionMatrix(znear=camera.znear,zfar=camera.zfar,
            fovX=self.FoVx,fovY=self.FoVy,fx=self.focal_x,fy=self.focal_y,cx=self.cx,cy=self.cy,
            image_width=self.image_width,image_height=self.image_height).T.to(camera.world_view_transform)
        self.full_proj_transform=camera.world_view_transform@self.projection_matrix

    def __getattr__(self,name):return getattr(self.camera,name)
