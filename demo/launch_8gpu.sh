#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/cpfs04/shared/kilab/hf-hub/Qwen2.5-7B-Instruct"
BASE_PORT=8001

# 可选：稳定性/网络设置（本机单机多实例通常不需要 NCCL）
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
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --enforce-eager False \
    --max-num-seqs 512 \
    --served-model-name qwen-2 \
    > "${LOG}" 2>&1 &

  # 给每个实例一点冷启动时间，避免抢占 IO
  sleep 2
done

echo "All 8 instances launched."
