#!/usr/bin/env bash
# Minimal SAR projection launcher.
#
# Example:
#   BASE_MODEL=/path/to/base RL_MODEL=/path/to/rl SAVE_PATH=checkpoints/sar-model \
#     bash scripts/run_sar_projection.sh

set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-python}"
BASE_MODEL="${BASE_MODEL:?Set BASE_MODEL to the base model path or HF id}"
RL_MODEL="${RL_MODEL:?Set RL_MODEL to the RL model path or HF id}"
SAVE_PATH="${SAVE_PATH:?Set SAVE_PATH to the output checkpoint directory}"
SVD_RANK="${SVD_RANK:-1000000}"
DELTA_FRACTION="${DELTA_FRACTION:-0.01}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TARGET_MODULES="${TARGET_MODULES:-q_proj k_proj v_proj o_proj gate_proj up_proj down_proj}"

export CUDA_VISIBLE_DEVICES

echo "[INFO] Starting SAR projection"
echo "[INFO] BASE_MODEL=${BASE_MODEL}"
echo "[INFO] RL_MODEL=${RL_MODEL}"
echo "[INFO] SAVE_PATH=${SAVE_PATH}"
echo "[INFO] DELTA_FRACTION=${DELTA_FRACTION}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

"$PYTHON_EXE" scripts/svd_rewire_parallel.py \
    --base_model "$BASE_MODEL" \
    --rl_model "$RL_MODEL" \
    --save_path "$SAVE_PATH" \
    --svd_rank "$SVD_RANK" \
    --delta_fraction "$DELTA_FRACTION" \
    --target_modules $TARGET_MODULES \
    --trust_remote_code

echo "[SUCCESS] SAR model saved to ${SAVE_PATH}"

