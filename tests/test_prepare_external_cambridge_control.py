import json
from pathlib import Path
import pickle

import numpy as np
from PIL import Image
import torch

from scripts.prepare_external_cambridge_control import build_external_control


def _write_source(root: Path) -> tuple[Path, Path]:
    images = root / "images"
    images.mkdir(parents=True)
    for name in ("seq1__frame00001.png", "seq2__frame00002.png"):
        (images / name).write_bytes(b"rgb")
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "images.bin").write_bytes(b"sparse")
    mapping = {
        "seq1__frame00001.png": "seq1/frame00001.png",
        "seq2__frame00002.png": "seq2/frame00002.png",
    }
    (root / "name_mapping.json").write_text(json.dumps(mapping))
    (root / "source_split.txt").write_text("seq1__frame00001.png\nseq2__frame00002.png\n")
    mask_path = root.parent / "external_masks.pkl"
    masks = {
        source_name: tuple(torch.ones((3, 4), dtype=torch.bool) for _ in range(3))
        for source_name in mapping.values()
    }
    with mask_path.open("wb") as handle:
        pickle.dump(masks, handle)
    return root, mask_path


def test_external_control_rekeys_exactly_the_staged_all_train_masks(tmp_path: Path):
    staged, masks = _write_source(tmp_path / "staged")
    output = tmp_path / "external_control"

    manifest = build_external_control(
        staged_dataset=staged,
        source_mask_pickle=masks,
        output=output,
    )

    assert manifest["staged_rgb_count"] == 2
    assert manifest["selected_mask_count"] == 2
    assert manifest["all_train_split"] == "no_dataset_test_txt"
    assert (output / "images" / "seq1__frame00001.png").is_symlink()
    assert (output / "sparse").is_symlink()
    assert (output / "name_mapping.json").is_symlink()
    assert not (output / "dataset_test.txt").exists()
    with (output / "images" / "masks.pkl").open("rb") as handle:
        selected = pickle.load(handle)
    assert set(selected) == {"seq1__frame00001.png", "seq2__frame00002.png"}
    assert all(len(channels) == 3 for channels in selected.values())


def test_external_control_can_materialize_shared_bilinear_targets(tmp_path: Path):
    staged, masks = _write_source(tmp_path / "staged")
    pixels = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    for name in ("seq1__frame00001.png", "seq2__frame00002.png"):
        Image.fromarray(pixels, mode="RGB").save(staged / "images" / name)
    output = tmp_path / "external_control_shared"

    manifest = build_external_control(
        staged_dataset=staged,
        source_mask_pickle=masks,
        output=output,
        canonical_size=(3, 2),
    )

    image = Image.open(output / "images" / "seq1__frame00001.png")
    assert image.size == (3, 2)
    assert not (output / "images" / "seq1__frame00001.png").is_symlink()
    assert manifest["rgb_target_storage"] == "shared_uint8_torch_bilinear_align_corners_false"
    assert manifest["canonical_image_size_wh"] == [3, 2]
    assert manifest["canonical_source_size_counts_wh"] == {"6x4": 2}
