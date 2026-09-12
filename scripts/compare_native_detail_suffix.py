"""Matched surface-proposal A/B metrics and scientific native crop panels."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,required=True);p.add_argument('--area',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--control-variable',choices=('depth','surface_geometry','proposal_geometry'),default='depth')
    a=p.parse_args()
    manifests=[json.loads((x/'manifest.json').read_text()) for x in (a.native,a.area)]
    differing={'geometry'}
    if a.control_variable=='depth':
        if manifests[0]['geometry']!='native' or manifests[1]['geometry']!='area':raise ValueError('Wrong arm order')
    elif a.control_variable=='surface_geometry':
        if any(m['geometry']!='native' for m in manifests) or [m['surface_geometry'] for m in manifests]!=[True,False]:
            raise ValueError('Geometry experiment then fixed control required')
        differing={'surface_geometry','position_scale_rotation_frozen','source_geometry_frozen'}
        for m in manifests:
            if m['position_scale_rotation_frozen']==m['surface_geometry'] or m['source_geometry_frozen']==m['surface_geometry']:
                raise ValueError('Contradictory geometry policy')
    else:
        if any(m['geometry']!='native' or m['surface_geometry'] for m in manifests):raise ValueError('Fixed source/native proposal pair required')
        if [m['refine_proposals'] for m in manifests]!=[True,False]:raise ValueError('Wrong proposal arm order')
        if [m['proposal_reprojection_weight'] for m in manifests]!=[.001,0.]:raise ValueError('Unknown geometry recipe')
        differing={'refine_proposals','proposal_reprojection_weight'}
    if {k:v for k,v in manifests[0].items() if k not in differing}!={k:v for k,v in manifests[1].items() if k not in differing}:
        raise ValueError('Unmatched pair contract')
    for run in (a.native,a.area):
        if not (run/'suffix.pth').exists() or not json.loads((run/'frozen_parameter_audit.json').read_text())['unchanged']:
            raise ValueError('Both arms must complete and preserve the source')
    step=manifests[0]['steps']
    rows=[json.loads((run/f'eval_{step:06d}.json').read_text()) for run in (a.native,a.area)]
    if [r['index'] for r in rows[0]]!=[r['index'] for r in rows[1]]:raise ValueError('Different evaluation views')
    paired=[]
    for x,y in zip(*rows):
        if any(abs(x['source'][k]-y['source'][k])>1e-5 for k in x['source'] if x['source'][k] is not None):
            raise ValueError('Source reference mismatch')
        paired.append(dict(index=x['index'],native_minus_area={k:x['updated'][k]-y['updated'][k] for k in x['updated'] if x['updated'][k] is not None},
                           native_minus_source={k:x['updated'][k]-x['source'][k] for k in x['updated'] if x['updated'][k] is not None}))
    keys=('tree','rigid','hard')
    report=dict(scope=f'native_fullframe_{len(paired)}view_regression__not_geometry_verification',per_view=paired,
        native_minus_area={k:float(np.mean([r['native_minus_area'][k] for r in paired])) for k in keys},
        native_minus_source={k:float(np.mean([r['native_minus_source'][k] for r in paired])) for k in keys})
    report['control_variable']=a.control_variable
    report['evaluation_view_count']=len(paired)
    if a.control_variable!='depth':
        report['experiment_minus_control']=report.pop('native_minus_area')
        report['experiment_minus_source']=report.pop('native_minus_source')
        for row in paired:
            row['experiment_minus_control']=row.pop('native_minus_area')
            row['experiment_minus_source']=row.pop('native_minus_source')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'comparison.json').write_text(json.dumps(report,indent=2))
    for name,roi in [('railings',(60,855,1170,1065)),('canopy',(1410,615,1890,1005))]:
        x0,y0,x1,y1=roi
        native=np.asarray(Image.open(a.native/'view_660.png'));area=np.asarray(Image.open(a.area/'view_660.png'))
        if native.shape!=area.shape or native.shape[1]!=5760:raise ValueError('Expected native GT/source/update panels')
        gt=native[y0:y1,x0:x1];n=native[y0:y1,3840+x0:3840+x1];b=area[y0:y1,3840+x0:3840+x1]
        diff=np.abs(n.astype(float)-b.astype(float)).mean(-1)/255
        fig,axs=plt.subplots(4,1,figsize=(14,10))
        titles=['Reference RGB','Native-layer proposal geometry','Area-depth proposal geometry'] if a.control_variable=='depth' else ['Reference RGB','Existing surface geometry optimized','Existing surface geometry fixed']
        if a.control_variable=='proposal_geometry':titles=['Reference RGB','Detail positions/scales + reprojection optimized','Detail geometry fixed; optics only']
        for ax,im,title in zip(axs[:3],[gt,n,b],titles):
            ax.imshow(im);ax.set_title(title);ax.axis('off')
        im=axs[3].imshow(diff,vmin=0,vmax=.01,cmap='magma');axs[3].axis('off')
        axs[3].set_title('Experiment versus control: absolute RGB difference (0 to 0.01; saved PNG quantization applies)')
        fig.colorbar(im,ax=axs[3],fraction=.025);fig.tight_layout();fig.savefig(a.output/f'660_{name}_ab.png',dpi=120);plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k!='per_view'}),flush=True)


if __name__=='__main__':main()
