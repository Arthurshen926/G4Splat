"""Scientific overlay of measured depth disagreement on training RGB."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();m=json.loads((a.run/'manifest.json').read_text());c=json.loads((a.run/'cameras.json').read_text())
with np.load(a.audit/'depth_comparison.npz') as data:samples=data['samples']
fig,axes=plt.subplots(3,1,figsize=(16,27))
for ax,index in zip(axes,[559,641,665]):
    rgb=np.array(Image.open(Path(m['args']['source_path'])/'images'/(c[index]['img_name']+'.png')).convert('RGB'))
    rows=samples[(samples[:,0]==index)&(samples[:,5]>.9)]
    ax.imshow(rgb)
    dots=ax.scatter(rows[:,1],rows[:,2],c=np.log2(rows[:,4]/rows[:,3]),s=55,
                    cmap='coolwarm',vmin=-1,vmax=1,edgecolors='black',linewidths=.4)
    ax.set_title(f'Training view {index}: opaque surface samples; red = model farther than MoGe')
    ax.set_xlim(0,rgb.shape[1]);ax.set_ylim(rgb.shape[0],0);ax.axis('off')
fig.colorbar(dots,ax=axes,shrink=.45,label='log2(surface median camera-z / MoGe camera-z)')
fig.savefig(a.output,dpi=100,bbox_inches='tight');plt.close(fig)
