#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-../plantseg}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/plantseg}"
ASSET_CACHE_DIR="${ASSET_CACHE_DIR:-pretrained_assets}"
BATCH_SIZE="${BATCH_SIZE:-4}"
IMG_SIZE="${IMG_SIZE:-448}"
WORKERS="${WORKERS:-4}"

mkdir -p "${OUTPUT_DIR}"
export USE_TF=0

python train.py \
  --model caris \
  --dataset plantseg \
  --model_id plantseg \
  --batch-size "${BATCH_SIZE}" \
  --pin_mem \
  --print-freq 50 \
  --workers "${WORKERS}" \
  --lr 1e-4 \
  --wd 1e-2 \
  --swin_type base \
  --warmup \
  --warmup_ratio 1e-3 \
  --warmup_iters 1500 \
  --clip_grads \
  --clip_value 1.0 \
  --epochs 50 \
  --img_size "${IMG_SIZE}" \
  --output-dir "${OUTPUT_DIR}" \
  --plantseg_root "${DATA_ROOT}" \
  --caption_index 3 \
  --asset_cache_dir "${ASSET_CACHE_DIR}" \
  --auto_download_assets
