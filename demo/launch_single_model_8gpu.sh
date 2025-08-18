#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/cpfs04/shared/kilab/hf-hub/Qwen2.5-7B-Instruct"
PORT=8001

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

echo "🚀 启动单模型+8GPU分布式vLLM实例 (张量并行模式)..."

# 停止现有的vLLM进程
echo "🛑 停止现有vLLM进程..."
pkill -f "vllm serve" || true
sleep 3

# 清理编译缓存
echo "🗑️ 清理编译缓存..."
rm -rf ~/.cache/vllm/torch_compile_cache || true
rm -rf ~/.cache/torch/compiled_cache || true

echo "📡 启动单模型+8GPU vLLM实例..."
echo "   模型: ${MODEL_PATH}"
echo "   端口: ${PORT}"
echo "   张量并行: 8"

# 启动单模型+8GPU实例
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --served-model-name qwen-2 \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.4 \
  --max-num-seqs 128 \
  --enable-lora \
  --max-lora-rank 64 \
  --max-loras 16

echo "✅ 单模型+8GPU启动完成"
echo "⏳ 等待实例启动..."
sleep 30

echo "🔍 检查实例状态..."
if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
  echo "✅ 健康 - http://localhost:${PORT}"
else
  echo "❌ 未响应"
fi

echo ""
echo "🎉 单模型+8GPU分布式vLLM启动完成！"
echo "📊 配置信息:"
echo "   - 端口: http://localhost:${PORT}"
echo "   - 张量并行: 8 GPU"
echo "   - LoRA支持: 启用"
echo "   - 最大LoRA数: 16"
echo "   - 最大LoRA rank: 64"
echo ""
echo "🔧 关键特性:"
echo "   - 单进程管理8个GPU"
echo "   - 支持8个LoRA独立热更新"
echo "   - 张量并行加速推理"
echo "   - 禁用编译避免缓存问题"
echo ""
echo "📝 日志文件: vllm_single_8gpu.log"
echo "🔍 查看日志: tail -f vllm_single_8gpu.log"
