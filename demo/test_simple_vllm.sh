#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/cpfs04/shared/kilab/hf-hub/Qwen2.5-7B-Instruct"
PORT=8001

echo "🧪 简单vLLM测试..."

# 停止现有进程
pkill -f "vllm serve" || true
sleep 2

echo "启动vLLM实例..."

CUDA_VISIBLE_DEVICES=0 \
vllm serve "${MODEL_PATH}" \
  --port "${PORT}" \
  --gpu-memory-utilization 0.3 \
  > test.log 2>&1 &

echo "等待启动..."
sleep 20

echo "检查状态..."
if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
  echo "✅ 成功!"
else
  echo "❌ 失败"
  echo "日志:"
  tail -10 test.log
fi

pkill -f "vllm serve" || true
