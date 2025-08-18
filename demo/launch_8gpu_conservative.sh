#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/cpfs04/shared/kilab/hf-hub/Qwen2.5-7B-Instruct"
BASE_PORT=8001

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "🚀 启动8GPU分布式vLLM实例 (保守内存配置)..."

# 停止现有的vLLM进程
echo "🛑 停止现有vLLM进程..."
pkill -f "vllm serve" || true
sleep 3

# 启动8个vLLM实例，使用非常保守的内存配置
for i in $(seq 0 7); do
  PORT=$((BASE_PORT + i))
  LOG="vllm_gpu${i}.log"
  
  echo "📡 启动GPU ${i} 在端口 ${PORT} -> ${LOG}"
  
  CUDA_VISIBLE_DEVICES=${i} \
  vllm serve "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.4 \
    --max-num-seqs 128 \
    --served-model-name qwen-2 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    > "${LOG}" 2>&1 &
  
  echo "✅ GPU ${i} 启动完成"
  sleep 8  # 给每个实例更多启动时间
done

echo "⏳ 等待所有实例启动..."
sleep 45

echo "🔍 检查实例状态..."
for i in $(seq 0 7); do
  PORT=$((BASE_PORT + i))
  echo -n "GPU ${i} (端口 ${PORT}): "
  
  if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "✅ 健康"
  else
    echo "❌ 未响应"
  fi
done

echo ""
echo "🎉 8GPU分布式vLLM启动完成！"
echo "📊 端口映射:"
for i in $(seq 0 7); do
  PORT=$((BASE_PORT + i))
  echo "   GPU ${i} -> http://localhost:${PORT}"
done
echo ""
echo "🔧 保守配置说明:"
echo "   - GPU内存利用率: 40% (非常保守，确保稳定性)"
echo "   - 最大序列长度: 8192 (最小化内存占用)"
echo "   - 最大并发序列: 128 (减少内存压力)"
echo ""
echo "📝 日志文件: vllm_gpu0.log 到 vllm_gpu7.log"
