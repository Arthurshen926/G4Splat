"""Scientific contact sheet of train-only source observations, not render A/B."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
for name in ['run','association','heldout','output']:p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();m=json.loads((a.run/'manifest.json').read_text());cams=json.loads((a.run/'cameras.json').read_text())
checks=defaultdict(list)
for r in json.loads(a.heldout.read_text())['rows']:checks[r['row']].append(r['heldout_error'])
ids={i for i,v in checks.items() if len(v)>=4 and all(e is not None and e<=1.5 for e in v)}
records=[r for r in json.loads((a.association/'audit.json').read_text())['observations'] if r['row'] in ids]
fig,axes=plt.subplots((len(records)+3)//4,4,figsize=(12,3*((len(records)+3)//4)))
for ax in axes.flat:ax.axis('off')
for ax,r in zip(axes.flat,records):
    o=r['observations'][0];i=o['view'];x,y=np.rint(o['xy']).astype(int)
    if i not in m['training_views']:raise ValueError('Training only')
    rgb=np.asarray(Image.open(Path(m['args']['source_path'])/'images'/(cams[i]['img_name']+'.png')).convert('RGB'))
    x0=max(0,x-64);y0=max(0,y-64);crop=rgb[y0:y+65,x0:x+65]
    ax.imshow(crop);ax.plot(x-x0,y-y0,'r+',markersize=9);ax.set_title(f'row {r["row"]}, train {i}, {len(r["observations"])} views')
fig.suptitle('Measured detail tracks: source RGB crops (NOT reconstruction)')
fig.tight_layout();fig.savefig(a.output,dpi=130);plt.close(fig)
