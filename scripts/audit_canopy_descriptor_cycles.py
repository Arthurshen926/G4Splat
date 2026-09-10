"""Three-view descriptor-cycle audit without epipolar preselection."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def close_cycles(ab, bc, ac, tolerance=1.):
    """Join reciprocal edges by native pixel identity, before geometric tests."""
    if not len(ab['left']) or not len(bc['left']) or not len(ac['left']):
        return np.zeros(0,dtype=np.int64),np.zeros(0,dtype=np.int64),np.zeros(0,dtype=np.int64)
    db,j=cKDTree(bc['left']).query(ab['right'])
    da,k=cKDTree(ac['left']).query(ab['left'])
    dc=np.linalg.norm(bc['right'][j]-ac['right'][k],axis=1)
    keep=(db<=tolerance)&(da<=tolerance)&(dc<=tolerance)
    i=np.flatnonzero(keep)
    return i,j[i],k[i]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('edges','closing-edges','cameras','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--triples',type=int,nargs='+')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from scripts.audit_canopy_descriptor_geometry import calibrated_projection, triangulate_matches
    from scripts.audit_canopy_epipolar_consistency import fundamental_from_cameras, sampson_pixel_distance
    cameras=json.loads(a.cameras.read_text())
    triples=[(658,659,661),(659,661,662),(686,687,688),(725,726,727),(582,583,584)]
    if a.triples is not None:
        from scripts.evaluate_canopy_validation import FIXED,ADDITIONAL
        if not a.triples or len(a.triples)%3 or set(a.triples)&set(FIXED+ADDITIONAL):
            raise ValueError('Complete non-validation camera triples required')
        triples=list(zip(a.triples[::3],a.triples[1::3],a.triples[2::3]))
        if len(set(triples))!=len(triples) or any(len(set(t))!=3 for t in triples):
            raise ValueError('Distinct three-camera cycles required')
    def read(i,j):
        paths=[d/f'matches_{i}_{j}.npz' for d in (a.edges,a.closing_edges)]
        existing=[p for p in paths if p.exists()]
        if len(existing)!=1: raise ValueError('Exactly one immutable edge required')
        return dict(np.load(existing[0]))
    records=[]
    for i,j,k in triples:
        ab,bc,ac=read(i,j),read(j,k),read(i,k)
        u,v,w=close_cycles(ab,bc,ac)
        left,middle,right=ab['left'][u],ab['right'][u],ac['right'][w]
        labels=np.where((ab['labels'][u]==bc['labels'][v])&(ab['labels'][u]==ac['labels'][w]),ab['labels'][u],0)
        points,geometric,error,angle=triangulate_matches(left,middle,cameras[i],cameras[j])
        matrix=calibrated_projection(cameras[k])
        projected=points@matrix[:,:3].T+matrix[:,3]
        third_error=np.linalg.norm(projected[:,:2]/np.maximum(projected[:,2:],1.e-12)-right,axis=1)
        epi=np.maximum(sampson_pixel_distance(left,middle,fundamental_from_cameras(cameras[i],cameras[j])),
                       sampson_pixel_distance(left,right,fundamental_from_cameras(cameras[i],cameras[k])))
        row={'triple':[i,j,k],'closed_cycles':len(u),'regions':{}}
        for name,code in [('canopy',1),('rigid',2)]:
            selected=labels==code
            values=epi[selected]
            row['regions'][name]={'tracks':int(selected.sum()),
                'median_max_pair_epipolar_px':float(np.median(values)) if len(values) else None,
                'p90_max_pair_epipolar_px':float(np.quantile(values,.9)) if len(values) else None,
                'median_third_view_reprojection_px':float(np.nanmedian(third_error[selected])) if len(values) else None,
                'all_three_geometric_candidates':int((selected&geometric&(third_error<1)&(projected[:,2]>0)).sum())}
        np.savez_compressed(a.output/f'tracks_{i}_{j}_{k}.npz',left=left,middle=middle,right=right,
            points=points,labels=labels,third_error=third_error,geometric=geometric,angle=angle)
        records.append(row);print(json.dumps(row),flush=True)
    (a.output/'audit.json').write_text(json.dumps({'scope':'RGB_descriptor_cycle_before_epipolar_filter__not_motion_ground_truth',
        'native_pixel_cycle_tolerance':1.,'records':records},indent=2))


if __name__=='__main__':main()
