#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${ENV_FILE:-envs/server_4090.yml}"
ENV_NAME="${ENV_NAME:-caris-plantseg}"

conda env create -n "${ENV_NAME}" -f "${ENV_FILE}" || conda env update -n "${ENV_NAME}" -f "${ENV_FILE}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

pip install --upgrade pip
conda install -y intel-openmp "mkl=2024.0"
pip install openmim
mim install "mmcv-full>=1.3.17,<1.5.0"
pip install -r requirements.server.txt
