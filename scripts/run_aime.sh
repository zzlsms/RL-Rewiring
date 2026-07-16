#!/usr/bin/env bash
# Launch vLLM servers and distributed AIME evaluation workers.

set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-python}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a local checkpoint or Hugging Face model id}"
SERVED_NAME="${SERVED_NAME:-sar-model}"
OUT_DIR="${OUT_DIR:-outputs/aime24}"

NODE_RANK="${NODE_RANK:-0}"
TOTAL_NODES="${TOTAL_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
TP_SIZE="${TP_SIZE:-1}"

if (( GPUS_PER_NODE < 1 || TP_SIZE < 1 || GPUS_PER_NODE % TP_SIZE != 0 )); then
    echo "[ERROR] GPUS_PER_NODE must be positive and divisible by TP_SIZE." >&2
    exit 2
fi
if (( NODE_RANK < 0 || NODE_RANK >= TOTAL_NODES )); then
    echo "[ERROR] NODE_RANK must satisfy 0 <= NODE_RANK < TOTAL_NODES." >&2
    exit 2
fi

INSTANCES_PER_NODE=$((GPUS_PER_NODE / TP_SIZE))
NUM_SHARDS=$((TOTAL_NODES * INSTANCES_PER_NODE))

NUM_PROBLEMS="${NUM_PROBLEMS:--1}"
K_VALUE="${K_VALUE:-32}"
SUB_BATCH_SIZE="${SUB_BATCH_SIZE:-16}"
DATASET="${DATASET:-HuggingFaceH4/aime_2024}"
SPLIT="${SPLIT:-train}"
DTYPE="${DTYPE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
BASE_PORT="${BASE_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-28672}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
LOG_DIR="${OUT_DIR}/logs/node_${NODE_RANK}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

SERVER_PIDS=()
EVAL_PIDS=()

cleanup() {
    for pid in "${SERVER_PIDS[@]:-}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "[INFO] node=${NODE_RANK}/${TOTAL_NODES}, local_instances=${INSTANCES_PER_NODE}, shards=${NUM_SHARDS}"
echo "[INFO] Starting vLLM servers (tensor parallel size=${TP_SIZE})..."

for ((i = 0; i < INSTANCES_PER_NODE; i++)); do
    PORT=$((BASE_PORT + i))
    GPU_START=$((i * TP_SIZE))
    GPU_END=$((GPU_START + TP_SIZE - 1))
    GPU_IDS=$(seq -s, "$GPU_START" "$GPU_END")
    GLOBAL_SHARD_ID=$((NODE_RANK * INSTANCES_PER_NODE + i))

    CUDA_VISIBLE_DEVICES="$GPU_IDS" env -u RANK -u WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT \
        "$PYTHON_EXE" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "$SERVED_NAME" \
        --tensor-parallel-size "$TP_SIZE" \
        --port "$PORT" \
        --dtype "$DTYPE" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --max-model-len "$MAX_MODEL_LEN" \
        --enable-prefix-caching \
        --trust-remote-code \
        > "${LOG_DIR}/vllm_${PORT}.log" 2>&1 &

    SERVER_PIDS+=("$!")
    echo "[LAUNCH] gpu=${GPU_IDS} port=${PORT} shard=${GLOBAL_SHARD_ID}"
done

echo "[INFO] Waiting for vLLM health checks..."
START_TIME=$SECONDS
while true; do
    READY_COUNT=0
    for ((i = 0; i < INSTANCES_PER_NODE; i++)); do
        PORT=$((BASE_PORT + i))
        CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/health" || true)
        if [[ "$CODE" == "200" ]]; then
            READY_COUNT=$((READY_COUNT + 1))
        fi
    done

    if (( READY_COUNT == INSTANCES_PER_NODE )); then
        echo "[SUCCESS] All vLLM servers are ready."
        break
    fi

    for pid in "${SERVER_PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[ERROR] A vLLM server exited. Inspect ${LOG_DIR}/vllm_*.log." >&2
            exit 1
        fi
    done

    if (( SECONDS - START_TIME >= STARTUP_TIMEOUT )); then
        echo "[ERROR] vLLM startup timed out after ${STARTUP_TIMEOUT}s." >&2
        exit 1
    fi

    echo "[WAIT] ${READY_COUNT}/${INSTANCES_PER_NODE} ready..."
    sleep 10
done

echo "[INFO] Launching AIME evaluation workers..."
for ((i = 0; i < INSTANCES_PER_NODE; i++)); do
    PORT=$((BASE_PORT + i))
    GLOBAL_SHARD_ID=$((NODE_RANK * INSTANCES_PER_NODE + i))

    "$PYTHON_EXE" scripts/evaluation_aime24_distributed.py \
        --model_id "$MODEL_PATH" \
        --served_model_name "$SERVED_NAME" \
        --vllm_url "http://localhost:${PORT}" \
        --dataset "$DATASET" \
        --split "$SPLIT" \
        --shard_id "$GLOBAL_SHARD_ID" \
        --num_shards "$NUM_SHARDS" \
        --num_problems "$NUM_PROBLEMS" \
        --k "$K_VALUE" \
        --sub_batch_size "$SUB_BATCH_SIZE" \
        --max_tokens "$MAX_TOKENS" \
        --request_timeout "$REQUEST_TIMEOUT" \
        --output_dir "$OUT_DIR" \
        > "${LOG_DIR}/eval_shard_${GLOBAL_SHARD_ID}.log" 2>&1 &
    EVAL_PIDS+=("$!")
done

STATUS=0
for pid in "${EVAL_PIDS[@]}"; do
    if ! wait "$pid"; then
        STATUS=1
    fi
done

if (( STATUS != 0 )); then
    echo "[ERROR] At least one evaluation worker failed. Inspect ${LOG_DIR}/eval_shard_*.log." >&2
    exit "$STATUS"
fi

echo "[SUCCESS] Node ${NODE_RANK} finished its evaluation shards."
