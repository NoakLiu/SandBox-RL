#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/cpfs04/shared/kilab/hf-hub/Qwen2.5-7B-Instruct"
PORT=8001

echo "🧪 测试单个vLLM实例启动..."

# 停止现有的vLLM进程
echo "🛑 停止现有vLLM进程..."
pkill -f "vllm serve" || true
sleep 3

echo "📡 启动测试实例在端口 ${PORT}..."

CUDA_VISIBLE_DEVICES=0 \
vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.3 \
  --max-num-seqs 64 \
  --served-model-name qwen-2 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > test_vllm.log 2>&1 &

echo "⏳ 等待实例启动..."
sleep 30

echo "🔍 检查实例状态..."
if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
  echo "✅ 测试实例启动成功！"
  echo "📊 健康检查响应:"
  curl -s "http://localhost:${PORT}/health" | jq . || curl -s "http://localhost:${PORT}/health"
else
  echo "❌ 测试实例启动失败"
  echo "📝 查看日志:"
  tail -20 test_vllm.log
fi

echo ""
echo "🧹 清理测试实例..."
pkill -f "vllm serve" || true
