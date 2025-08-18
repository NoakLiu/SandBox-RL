#!/usr/bin/env bash

# 基础版本：不加载LoRA，确保vLLM能正常启动
echo "🚀 启动单模型+8GPU vLLM (基础版本，无LoRA)..."

# 停止现有进程
pkill -f "vllm serve" || true
sleep 3

# 启动vLLM，不加载LoRA
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve /cpfs04/shared/kilab/hf-hub/Qwen2.5-7B-Instruct \
  --port 8001 \
  --served-model-name qwen-2 \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.4 \
  --max-num-seqs 128

echo "✅ vLLM启动完成"
