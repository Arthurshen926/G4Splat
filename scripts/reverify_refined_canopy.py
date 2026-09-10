"""Audit displaced diagnostic leaves without changing the scene during evidence collection.

Old camera IDs nominate candidates; they do not retain authority automatically.
All audits render the same frozen input scene. No iteration-order-dependent
removal/reinsertion and no recoloring or geometry changes are permitted.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / '2d-gaussian-splatting')]
import torch
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import (
    VolumetricFoliageModel, render_hybrid, static_detail_forward_visibility_gate,
    VERIFICATION_VERIFIED, VERIFICATION_UNVERIFIED, PROPOSAL_RAY_BIRTH,
)
from outdoor.canopy_support_expansion import (
    canopy_material_audit_fields, front_of_known_rigid_mask, record_distinct_camera_witnesses_,
)
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.evaluate_canopy_validation import FIXED, ADDITIONAL, tensor_digest


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    parser.set_defaults(data_device='cpu', resolution=640, white_background=True)
    for name in ('checkpoint', 'foliage', 'masks', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--candidate-scope', choices=('existing', 'all-canonical'), default='existing')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.35)
    dataset = model.extract(args)
    dataset.model_path = str(args.output)
    teacher = load_hybrid_teacher(args.checkpoint, sh_degree=dataset.sh_degree)
    capture = torch.load(args.foliage, map_location='cpu')
    if not capture.get('diagnostic_only') or Path(capture['source_checkpoint']).resolve() != args.checkpoint.resolve():
        raise ValueError('Source-matched diagnostic required')
    leaf = VolumetricFoliageModel(dataset.sh_degree, dynamic_rank=capture['foliage']['dynamic_rank'], device='cuda')
    leaf.restore(capture['foliage'])
    if leaf.dynamic_leaf_mask.any():
        raise ValueError('Static-only audit')
    invariants = {key: tensor_digest(getattr(leaf, key)) for key in
                  ('xyz', 'log_scales', 'quaternions', 'opacity_logits', 'features', 'support_camera_ids')}
    targets = leaf.verification_state == VERIFICATION_VERIFIED
    candidates = leaf.verified_camera_ids.clone()
    accepted = torch.zeros_like(candidates, dtype=torch.bool)
    new_table = torch.full_like(candidates, -1)
    views = LazyScene(dataset, GaussianModel(dataset.sh_degree), image_cache_size=6).getTrainCameras()
    masks = CambridgeMaskLookup(Path(dataset.source_path), args.masks)
    canonical = teacher.state['training_contract']['static_scene_canonical_rgb']['canonical_sequence']
    order = [i for i, view in enumerate(views) if i not in set(FIXED + ADDITIONAL)
             and view.image_name.startswith(canonical + '__')]
    records = []
    for index in order:
        view = views[index]
        camera = int(view.colmap_id)
        nominated = targets if args.candidate_scope == 'all-canonical' else targets & (candidates == camera).any(dim=1)
        if not nominated.any():
            continue
        shape = (view.image_height, view.image_width)
        obj, sky, distortion, tree = masks.get_index_masks(view.image_name, (0, 1, 2, 3), shape, torch.device('cuda'))
        canopy = obj & sky & distortion & ~tree
        rigid = obj & sky & distortion & tree
        known_sky = obj & distortion & ~sky
        package = render_hybrid(view, teacher.surface, leaf, background=torch.ones(3, device='cuda'),
            include_dynamic=False, optical_replacement_policy='disabled', structural_trainable_start=None,
            volume_gate=static_detail_forward_visibility_gate(leaf, camera, include_pending_exact=False),
            audit_fields=canopy_material_audit_fields(canopy.float(), rigid.float(), known_sky.float(), view.original_image.cuda()))
        response = package.responsibility[package.structural_count:]
        witness = nominated & (response[:, 1] > 1.e-5) & (response[:, 1] > response[:, 2] + response[:, 7])
        wall = render_hybrid(view, teacher.surface, leaf, background=torch.zeros(3, device='cuda'),
            include_dynamic=False, optical_replacement_policy='disabled', structural_trainable_start=None,
            volume_gate=torch.zeros(len(leaf), device='cuda'))
        witness &= front_of_known_rigid_mask(leaf.xyz, view, wall.median_depth, wall.surface_alpha)
        accepted |= witness[:, None] & (candidates == camera)
        if args.candidate_scope == 'all-canonical':
            record_distinct_camera_witnesses_(new_table, witness, camera)
        record = {'index': index, 'nominated': int(nominated.sum()), 'accepted': int(witness.sum())}
        records.append(record)
        print(json.dumps(record), flush=True)
        del package, wall
    # Commit only after every camera has seen exactly the same frozen scene.
    table = torch.where(accepted, candidates, torch.full_like(candidates, -1))
    if args.candidate_scope == 'all-canonical':
        table = new_table
    counts = (table >= 0).sum(dim=1)
    retained = targets & (counts >= 2)
    rejected = targets & ~retained
    leaf.verified_camera_ids[targets] = table[targets]
    leaf.verified_camera_count[targets] = counts[targets].to(leaf.verified_camera_count.dtype)
    leaf.verified_sequence_count[targets] = (counts[targets] > 0).to(leaf.verified_sequence_count.dtype)
    leaf.verification_state[rejected] = VERIFICATION_UNVERIFIED
    leaf.proposal_kind[rejected] = PROPOSAL_RAY_BIRTH
    leaf.birth_iteration[rejected] = int(teacher.state['iteration'])
    assert invariants == {key: tensor_digest(getattr(leaf, key)) for key in invariants}
    audit = {'previously_verified': int(targets.sum()), 'retained': int(retained.sum()),
             'candidate_scope': args.candidate_scope,
             'rejected': int(rejected.sum()), 'frozen_scene_witness_audit': True,
             'production_ready': False, 'requires_native_quality_validation_after_removal': True,
             'invariants': invariants, 'records': records, 'excluded_view_indices': sorted(FIXED + ADDITIONAL)}
    torch.save({**capture, 'scope': 'fresh_dense_canonical_canopy_only__frozen_scene_reverification__nonresumable',
                'source_foliage': str(args.foliage.resolve()), 'reverification_audit': audit,
                'foliage': leaf.capture()}, args.output / 'diagnostic_foliage_capture.pth')
    (args.output / 'audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps({key: value for key, value in audit.items() if key not in ('records', 'invariants')}), flush=True)


if __name__ == '__main__':
    main()
