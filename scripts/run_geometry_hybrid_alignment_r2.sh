#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/root/miniconda3/envs/g4splat/lib

scene=/root/G4Splat/output_cambridge_qc_n24_30k/geometry_hybrid_r2/mast3r_sfm
log=/root/G4Splat/output_cambridge_qc_n24_30k/geometry_hybrid_r2/alignment.log

exec /root/miniconda3/bin/conda run --no-capture-output -n g4splat \
  python /root/G4Splat/scripts/align_charts.py \
  --source_path "$scene" \
  --mast3r_scene "$scene" \
  --output_path "$scene" \
  --config semantic_masked_strong_lowmem \
  --depth_model depthanythingv2 \
  --depthanythingv2_checkpoint_dir /root/MAtCha/Depth-Anything-V2/checkpoints \
  --depthanything_encoder vitl \
  --cambridge_mask_pickle /mnt/pool/sqy/Cambridge_stdloc/StMarysChurch/processed/masks.pkl \
  --cambridge_mask_dataset_path /root/G4Splat/output_cambridge_qc_n24_30k/prepared/StMarysChurch/dataset_qc_tree_v4/train_opt \
  --cambridge_mask_indices 0 1 2 \
  >> "$log" 2>&1
