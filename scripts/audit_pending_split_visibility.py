"""Read-only native A/B: unresolved split versus its immutable parent.

This is a counterfactual audit, NOT a production rendering workaround. The
context mutates a local diagnostic model only and restores it even on failure.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import torch
from PIL import Image
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import PROPOSAL_SPLIT, PROPOSAL_NONE, VERIFICATION_REJECTED
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.evaluate_canopy_validation import FIXED, ADDITIONAL, masked_metrics, tensor_digest


@contextmanager
def diagnostic_pending_parents(leaf):
    rows=torch.nonzero(leaf.proposal_kind==PROPOSAL_SPLIT,as_tuple=False).flatten()
    families=torch.unique(leaf.split_proposal_family_id[rows],sorted=True)
    snapshot_rows=leaf.split_parent_snapshot_indices(families)
    representatives=torch.stack([
        rows[leaf.split_proposal_family_id[rows]==family][0] for family in families
    ]) if len(families) else rows
    names={name.removeprefix('proposal_parent_') for name in leaf.split_parent_snapshot_names}
    names|={'proposal_kind','verification_state'}
    saved={name:getattr(leaf,name)[rows].detach().clone() for name in names}
    try:
        with torch.no_grad():
            leaf.verification_state[rows]=VERIFICATION_REJECTED
            for name in leaf.split_parent_snapshot_names:
                getattr(leaf,name.removeprefix('proposal_parent_'))[representatives]=getattr(leaf,name)[snapshot_rows]
            leaf.proposal_kind[representatives]=PROPOSAL_NONE
        yield {'pending_families':len(families),'pending_children':len(rows)}
    finally:
        with torch.no_grad():
            for name,value in saved.items():getattr(leaf,name)[rows]=value


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=640,white_background=True)
    for name in ('checkpoint','masks','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
    dataset=model.extract(a);dataset.model_path=str(a.output)
    teacher=load_hybrid_teacher(a.checkpoint,sh_degree=dataset.sh_degree)
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=4).getTrainCameras()
    masks=CambridgeMaskLookup(Path(dataset.source_path),a.masks)
    leaf=teacher.foliage
    names={name.removeprefix('proposal_parent_') for name in leaf.split_parent_snapshot_names}|{'proposal_kind','verification_state'}
    before={name:tensor_digest(getattr(leaf,name)) for name in names}
    results=[]
    for index in FIXED+ADDITIONAL:
        view=views[index];shape=(view.image_height,view.image_width)
        obj,sky,dist,tree=masks.get_index_masks(view.image_name,(0,1,2,3),shape,torch.device('cuda'))
        known=obj&sky&dist;canopy=known&~tree;rigid=known&tree
        target=view.original_image.cuda()
        native=teacher.render(view,task=None,conditioned=False)
        with diagnostic_pending_parents(leaf) as transaction:
            restored=teacher.render(view,task=None,conditioned=False)
        row={'index':index}
        for label,package in [('pending',native),('parent_fallback',restored)]:
            row[label]={k:masked_metrics(package['rgb'],target,m)['psnr'] for k,m in [('tree',canopy),('rigid',rigid)]}
            row[label]['canopy_alpha']=float(package['volume_alpha'].reshape_as(canopy)[canopy].mean())
        results.append(row)
        panel=torch.cat((target,native['rgb'],restored['rgb']),dim=2).clamp(0,1)
        Image.fromarray((panel.permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(a.output/f'view_{index}.png')
        print(json.dumps(row),flush=True)
    assert before=={name:tensor_digest(getattr(leaf,name)) for name in names}
    summary={label:{k:sum(row[label][k] for row in results)/len(results) for k in ['tree','rigid','canopy_alpha']} for label in ['pending','parent_fallback']}
    (a.output/'audit.json').write_text(json.dumps({'checkpoint':str(a.checkpoint),'scope':'read_only_counterfactual_parent_restoration_not_trained_result',
        'transaction':transaction,'summary':summary,'records':results,'all_tensors_restored':True},indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
