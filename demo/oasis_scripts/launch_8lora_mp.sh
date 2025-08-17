#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/cpfs04/shared/kilab/hf-hub/Qwen2.5-7B-Instruct"
BASE_PORT=8001
SERVE_NAME="qwen-2"

# 推荐的小环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

for i in $(seq 0 7); do
  PORT=$((BASE_PORT + i))
  LOG="vllm_gpu${i}.log"

  echo "Launching GPU ${i} on port ${PORT} -> ${LOG}"
  CUDA_VISIBLE_DEVICES=${i} \
  vllm serve "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --served-model-name "${SERVE_NAME}" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --enforce-eager False \
    --max-num-seqs 512 \
    > "${LOG}" 2>&1 &

  # 给每个实例一点冷启动时间，避免IO争用
  sleep 2
done

echo "All 8 instances launched (ports ${BASE_PORT}..$((BASE_PORT+7)))."
echo "LoRA 路由建议：lora_id = 1..8  ->  port = 8000 + lora_id"
