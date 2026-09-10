"""Plot measured reciprocal hypotheses on their input photograph."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--audit',type=Path,required=True)
p.add_argument('--source-map',type=Path,required=True)
p.add_argument('--images',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
d=json.loads(a.audit.read_text());camera=json.loads((a.source_map/'cameras.json').read_text())[d['sources'][0]]
points=np.load(a.source_map/'hypotheses_radius_0.4.npz',allow_pickle=False)
indices=np.asarray(d['summaries'][0]['indices'],dtype=np.int64)[:,0]
photo=Image.open(a.images/(camera['img_name']+'.png')).resize((camera['width'],camera['height']))
fig,ax=plt.subplots(figsize=(12,7));ax.imshow(photo)
scatter=ax.scatter(*points['uv'][indices].T,c=points['depth'][indices]/points['prior_depth'][indices],
                   s=18,cmap='coolwarm',vmin=.65,vmax=1.1,edgecolors='black',linewidths=.3)
fig.colorbar(scatter,ax=ax,label='Reciprocal depth / MoGe canopy prior')
ax.set_title(f'{len(indices)} reciprocal hypotheses (not certified leaf material)')
ax.set_axis_off();fig.tight_layout()
if a.output.exists():raise FileExistsError(a.output)
fig.savefig(a.output,dpi=150)
