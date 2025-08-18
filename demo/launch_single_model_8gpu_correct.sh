#!/usr/bin/env bash

# 根据vLLM官方文档的正确LoRA启动方式
echo "🚀 启动单模型+8GPU vLLM (正确LoRA配置)..."

# 停止现有进程
pkill -f "vllm serve" || true
sleep 3

# 创建LoRA模块配置
# 使用--lora-modules参数预加载LoRA适配器
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve /cpfs04/shared/kilab/hf-hub/Qwen2.5-7B-Instruct \
  --port 8001 \
  --served-model-name qwen-2 \
  --tensor-parallel-size 4 \
  --enable-lora \
  --max-lora-rank 64 \
  --max-loras 16 \
  --lora-modules \
    lora1=/cpfs04/shared/kilab/liudong/lora1 \
    lora2=/cpfs04/shared/kilab/liudong/lora2 \
    lora3=/cpfs04/shared/kilab/liudong/lora3 \
    lora4=/cpfs04/shared/kilab/liudong/lora4 \
    lora5=/cpfs04/shared/kilab/liudong/lora5 \
    lora6=/cpfs04/shared/kilab/liudong/lora6 \
    lora7=/cpfs04/shared/kilab/liudong/lora7 \
    lora8=/cpfs04/shared/kilab/liudong/lora8

echo "✅ vLLM启动完成"
