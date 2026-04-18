#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-../plantseg}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/plantseg}"
ASSET_CACHE_DIR="${ASSET_CACHE_DIR:-pretrained_assets}"
BATCH_SIZE="${BATCH_SIZE:-4}"
IMG_SIZE="${IMG_SIZE:-448}"
WORKERS="${WORKERS:-4}"
RESUME_PATH="${RESUME_PATH:-${OUTPUT_DIR}/model_best_plantseg.pth}"

mkdir -p "${OUTPUT_DIR}"
export USE_TF=0

python eval.py \
  --model caris \
  --dataset plantseg \
  --split test \
  --batch-size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --device cuda:0 \
  --img_size "${IMG_SIZE}" \
  --resume "${RESUME_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --plantseg_root "${DATA_ROOT}" \
  --caption_index 3 \
  --asset_cache_dir "${ASSET_CACHE_DIR}" \
  --auto_download_assets
