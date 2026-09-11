"""Read-only source-to-runtime depth audit; never assigns geometry authority."""
import argparse
import json
from pathlib import Path
import numpy as np


def tile_statistics(depth,valid,shape):
    h,w=shape
    if depth.ndim!=2 or valid.shape!=depth.shape or depth.shape[0]%h or depth.shape[1]%w:
        raise ValueError('Aligned integer area reduction required')
    dy,dx=depth.shape[0]//h,depth.shape[1]//w
    blocks=np.where(valid&np.isfinite(depth)&(depth>0),depth,np.nan).reshape(h,dy,w,dx).transpose(0,2,1,3).reshape(h,w,-1)
    count=np.isfinite(blocks).sum(-1)
    safe=np.where(count[...,None]>0,blocks,1.)
    low=np.nanmin(safe,-1);high=np.nanmax(safe,-1);mean=np.nanmean(safe,-1)
    return dict(valid_fraction=count/blocks.shape[-1],minimum=low,maximum=high,mean=mean,
                log_span=np.log(high/low),valid=count>0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-index',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();runtime=json.loads(a.runtime_index.read_text())
    source=json.loads(Path(runtime['source_index']).read_text())
    cache=np.load(a.runtime_index.parent/runtime['arrays']['depth_m']['path'],mmap_mode='r')
    regions={'seq2__frame00103':{'railing_rect':[20,285,390,355],'canopy_interior':[390,45,610,195],'canopy_leakage':[470,205,630,335]}}
    records=[]
    for stem,rois in regions.items():
        path=Path(source['records'][stem]['path'])
        with np.load(path,allow_pickle=False) as data:
            stats=tile_statistics(data['depth_m'],data['valid_mask'].astype(bool),runtime['raster_shape'])
        current=cache[runtime['camera_order'].index(stem)]
        for name,(x0,y0,x1,y1) in rois.items():
            sl=np.s_[y0:y1,x0:x1];ok=stats['valid'][sl];span=stats['log_span'][sl][ok]
            records.append(dict(image=stem,region=name,xyxy=[x0,y0,x1,y1],pixels=int(ok.sum()),
                depth_span_above_10pct_fraction=float((span>np.log(1.1)).mean()),
                depth_span_above_25pct_fraction=float((span>np.log(1.25)).mean()),
                depth_span_ratio_quantiles=np.exp(np.quantile(span,[.5,.9,.99,1])).tolist(),
                maximum_cached_vs_area_mean_error=float(np.abs(current[sl][ok]-stats['mean'][sl][ok]).max()),
                limitation='Rectangular diagnostic ROI, not a semantic mask or proof of correct depth'))
    result=dict(scope='read_only_raw_MoGe_to_runtime_reduction__not_geometry_truth',records=records,
                source_shape=[1080,1920],runtime_shape=runtime['raster_shape'],
                limitation='Depth agreement across views and absolute scene calibration not assessed here')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
