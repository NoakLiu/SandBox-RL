#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/cpfs04/shared/kilab/hf-hub/Qwen2.5-7B-Instruct"
BASE_PORT=8001

echo "🧹 清理vLLM缓存和进程..."

# 停止所有vLLM进程
echo "🛑 停止vLLM进程..."
pkill -f "vllm serve" || true
sleep 3

# 清理编译缓存
echo "🗑️ 清理编译缓存..."
rm -rf ~/.cache/vllm/torch_compile_cache || true
rm -rf ~/.cache/torch/compiled_cache || true

echo "⏳ 等待清理完成..."
sleep 5

echo "🚀 启动单个测试实例..."

# 启动单个测试实例
CUDA_VISIBLE_DEVICES=0 \
vllm serve "${MODEL_PATH}" \
  --port "${BASE_PORT}" \
  --gpu-memory-utilization 0.3 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --served-model-name qwen-2 \
  > test_clean.log 2>&1 &

echo "⏳ 等待实例启动..."
sleep 30

echo "🔍 检查实例状态..."
if curl -s "http://localhost:${BASE_PORT}/health" > /dev/null 2>&1; then
  echo "✅ 清理后启动成功！"
  echo "📊 健康检查响应:"
  curl -s "http://localhost:${BASE_PORT}/health"
else
  echo "❌ 启动失败"
  echo "📝 查看日志:"
  tail -20 test_clean.log
fi

echo ""
echo "🧹 清理测试实例..."
pkill -f "vllm serve" || true
