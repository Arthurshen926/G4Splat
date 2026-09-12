"""Scientific depth evidence visualization; does not render reconstructed geometry."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from outdoor.moge3_depth_layers import reduce_depth_layers
from scripts.build_moge3_chart_base import _resize_scalar


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--depth',type=Path,required=True);p.add_argument('--rgb',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--roi',nargs=4,type=int,default=[60,855,1170,1065])
    a=p.parse_args()
    with np.load(a.depth) as z:
        depth=z['depth_m'];valid=z['valid_mask'].astype(bool)&z['refinement_valid_mask'].astype(bool)
    rgb=np.asarray(Image.open(a.rgb).convert('RGB'))
    if rgb.shape[:2]!=depth.shape:raise ValueError('RGB and depth must align')
    shape=(depth.shape[0]//3,depth.shape[1]//3)
    layers=reduce_depth_layers(depth,valid,shape);area,_=_resize_scalar(depth,valid,shape)
    retained=np.full_like(depth,np.nan)
    for name in ('front','back'):
        x=layers[name+'_native_x'];y=layers[name+'_native_y'];ok=x>=0
        retained[y[ok],x[ok]]=layers[name+'_depth'][ok]
    x0,y0,x1,y1=a.roi;sl=np.s_[y0:y1,x0:x1]
    raw=np.where(valid,depth,np.nan);lo,hi=np.nanpercentile(raw[sl],[2,98])
    up=lambda x:np.repeat(np.repeat(x,3,0),3,1)
    panels=[(rgb[sl],'Native RGB'),(raw[sl],'Native predicted depth (shared color scale)'),
            (up(area)[sl],'Area mean: foreground/background can mix'),
            (retained[sl],'Retained native sample rays (blank = no representative)'),
            (up(layers['mixed'])[sl],'Mixed cells: no continuous MoGe base replacement')]
    fig,axs=plt.subplots(5,1,figsize=(15,13))
    for ax,(data,title) in zip(axs,panels):
        if data.ndim==3:ax.imshow(data)
        elif data.dtype==bool:ax.imshow(data,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        else:ax.imshow(data,cmap='turbo_r',vmin=lo,vmax=hi,interpolation='nearest')
        ax.set_title(title);ax.axis('off')
    fig.suptitle('Evidence transfer only; not a reconstruction A/B or geometry truth')
    fig.tight_layout();a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    fig.savefig(a.output,dpi=140);plt.close(fig)


if __name__=='__main__':main()
