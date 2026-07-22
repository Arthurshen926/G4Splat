from pathlib import Path

from view_quality_control.provenance import audit_input_provenance


def _audit_provenance(dataset: Path, mask: Path) -> dict:
    return audit_input_provenance(
        dataset,
        mask,
        quality_mask_indices=[0, 1, 2],
        thing_mask_index=0,
        sky_mask_index=1,
        tree_mask_index=3,
        pose_clusters=8,
        neighbor_count=8,
        max_image_width=960,
        max_reject_fraction=0.05,
    )


def test_audit_provenance_changes_for_source_image_or_audit_knob(tmp_path: Path):
    dataset = tmp_path / "train"
    image_dir = dataset / "images"
    sparse = dataset / "sparse" / "0"
    image_dir.mkdir(parents=True)
    sparse.mkdir(parents=True)
    (image_dir / "a.png").write_bytes(b"image-a")
    (dataset / "name_mapping.json").write_text('{"a.png": "seq/a.png"}')
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (sparse / name).write_bytes(name.encode("utf-8"))
    mask = tmp_path / "masks.pkl"
    mask.write_bytes(b"mask")

    baseline = _audit_provenance(dataset, mask)
    (image_dir / "a.png").write_bytes(b"image-a-replaced")
    changed_image = _audit_provenance(dataset, mask)
    assert changed_image != baseline

    changed_knob = audit_input_provenance(
        dataset,
        mask,
        quality_mask_indices=[0, 1, 2],
        thing_mask_index=0,
        sky_mask_index=1,
        tree_mask_index=3,
        pose_clusters=9,
        neighbor_count=8,
        max_image_width=960,
        max_reject_fraction=0.05,
    )
    assert changed_knob != changed_image
