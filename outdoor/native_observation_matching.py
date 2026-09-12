"""Measured RGB patch association and fixed-camera ray triangulation."""
import numpy as np
from itertools import combinations


def rebind_angular_sigma(sigma, old_depth, new_depth):
    """Preserve measured source-camera angular footprint, not world size truth."""
    sigma,old_depth,new_depth=[np.asarray(x,dtype=np.float64) for x in (sigma,old_depth,new_depth)]
    if not (sigma.shape==old_depth.shape==new_depth.shape):raise ValueError('Aligned scale/depth rows required')
    if any(not np.isfinite(x).all() or (x<=0).any() for x in (sigma,old_depth,new_depth)):
        raise ValueError('Finite positive scale and depths required')
    return sigma*(new_depth/old_depth)


def epipolar_pixel_line(source_center, source_direction, target_center, target_rotation, fx, fy, cx, cy):
    """Line in integer-index pixel coordinates; cx/cy use that convention."""
    normal=np.cross(np.asarray(target_center)-source_center,source_direction)
    n=normal@np.asarray(target_rotation)
    return np.array([n[0]/fx,n[1]/fy,n[2]-n[0]*cx/fx-n[1]*cy/fy])


def triangulate_consensus(centers, directions, reprojection_errors, threshold=1.5):
    """Deterministic triplet consensus, requiring source observation zero.

    The callback returns native pixel errors, infinity for invalid projection.
    Refitting never exports observations which fail the final pixel threshold.
    """
    c=np.asarray(centers);d=np.asarray(directions)
    if len(c)<3:return None
    best=None
    for pair in combinations(range(1,len(c)),2):
        ids=np.array((0,)+pair)
        xyz=triangulate_rays(c[ids],d[ids])
        if xyz is None:continue
        errors=np.asarray(reprojection_errors(xyz));keep=np.isfinite(errors)&(errors<=threshold)
        if not keep[0] or keep.sum()<3:continue
        # Remove inconsistent observations after each refit, never silently keep
        # stale supports from the initial depth-projected packet.
        for _ in range(len(c)):
            xyz=triangulate_rays(c[keep],d[keep])
            if xyz is None:break
            errors=np.asarray(reprojection_errors(xyz))
            valid=keep&np.isfinite(errors)&(errors<=threshold)
            if np.array_equal(valid,keep):break
            keep=valid
            if not keep[0] or keep.sum()<3:
                xyz=None;break
        if xyz is None:continue
        score=(int(keep.sum()),-float(errors[keep].mean()))
        if best is None or score>best[0]:best=(score,xyz,keep,errors)
    return None if best is None else best[1:]


def sample_rgb(image,x,y):
    """Bilinear samples; caller must keep coordinates inside the image."""
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int)
    x1=np.minimum(x0+1,image.shape[1]-1);y1=np.minimum(y0+1,image.shape[0]-1)
    wx=(x-x0)[...,None];wy=(y-y0)[...,None]
    return ((1-wy)*((1-wx)*image[y0,x0]+wx*image[y0,x1])
            +wy*((1-wx)*image[y1,x0]+wx*image[y1,x1]))


def match_patch(source, source_xy, target, guess_xy, *, radius=5, patch_radius=2, epipolar_line=None, subpixel=False):
    """Return an unambiguous integer RGB match, or None; no depth target loss."""
    source=np.asarray(source);target=np.asarray(target)
    sx,sy=np.asarray(source_xy,dtype=float) if subpixel else np.rint(source_xy).astype(int)
    gx,gy=np.rint(guess_xy).astype(int)
    p=int(patch_radius)
    if min(sx,sy)<p or sx+p>=source.shape[1] or sy+p>=source.shape[0]:return None
    delta=np.arange(-p,p+1);oy,ox=np.meshgrid(delta,delta,indexing='ij')
    source_patch=(sample_rgb(source,sx+ox,sy+oy) if subpixel else source[sy+oy,sx+ox]).reshape(-1,3)
    if source_patch.std(0).mean()<.02:return None
    offsets=np.arange(-radius,radius+1);yy,xx=np.meshgrid(offsets,offsets,indexing='ij')
    x=gx+xx.ravel();y=gy+yy.ravel()
    valid=(x>=p)&(y>=p)&(x+p<target.shape[1])&(y+p<target.shape[0])
    if epipolar_line is not None:
        line=np.asarray(epipolar_line,dtype=float)
        norm=np.linalg.norm(line[:2])
        if line.shape!=(3,) or not np.isfinite(line).all() or norm<1e-12:return None
        valid &= np.abs(line[0]*x+line[1]*y+line[2])/norm<=1.5
    x=x[valid];y=y[valid]
    if not len(x):return None
    patches=target[y[:,None]+oy.ravel(),x[:,None]+ox.ravel()]
    sm=source_patch.mean(0,keepdims=True);tm=patches.mean(1,keepdims=True)
    sn=(source_patch-sm)/(source_patch.std(0,keepdims=True)+.05)
    tn=(patches-tm)/(patches.std(1,keepdims=True)+.05)
    cost=np.abs(tn-sn).mean((1,2))+.25*np.abs(tm-sm).mean((1,2))
    best=int(cost.argmin())
    alternative=(x-x[best])**2+(y-y[best])**2>4
    if not alternative.any() or cost[best]>.5 or cost[alternative].min()-cost[best]<.02:return None
    if subpixel:
        delta=np.arange(-.75,1.,.25);dy,dx=np.meshgrid(delta,delta,indexing='ij')
        xx=x[best]+dx.ravel();yy=y[best]+dy.ravel()
        valid=(xx>=p)&(yy>=p)&(xx+p<target.shape[1])&(yy+p<target.shape[0])
        valid &=(abs(xx-gx)<=radius)&(abs(yy-gy)<=radius)
        if epipolar_line is not None:valid &= abs(line[0]*xx+line[1]*yy+line[2])/norm<=1.5
        xx=xx[valid];yy=yy[valid]
        patches=sample_rgb(target,xx[:,None]+ox.ravel(),yy[:,None]+oy.ravel())
        tm=patches.mean(1,keepdims=True)
        tn=(patches-tm)/(patches.std(1,keepdims=True)+.05)
        refined=np.abs(tn-sn).mean((1,2))+.25*np.abs(tm-sm).mean((1,2))
        index=refined.argmin()
        return np.array([xx[index],yy[index]],dtype=float),float(refined[index])
    return np.array([x[best],y[best]],dtype=np.int32),float(cost[best])


def triangulate_rays(centers,directions):
    """Least-squares intersection; reject near-parallel or behind-camera fits."""
    c=np.asarray(centers,dtype=np.float64);d=np.asarray(directions,dtype=np.float64)
    if c.ndim!=2 or c.shape[1]!=3 or c.shape!=d.shape or len(c)<3:
        raise ValueError('At least three aligned ray observations required')
    if not np.isfinite(c).all() or not np.isfinite(d).all() or (np.linalg.norm(d,axis=-1)<1e-10).any():
        raise ValueError('Finite nonzero rays required')
    d=d/np.linalg.norm(d,axis=-1,keepdims=True)
    project=np.eye(3)[None]-d[:,:,None]*d[:,None,:]
    a=project.sum(0);eig=np.linalg.eigvalsh(a)
    if eig[0]<1e-6*eig[-1]:return None
    xyz=np.linalg.solve(a,np.einsum('nij,nj->i',project,c))
    if ((xyz-c)*d).sum(-1).min()<=0:return None
    return xyz
