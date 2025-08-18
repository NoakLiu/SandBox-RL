#!/usr/bin/env bash

# 替代方案：使用不同的张量并行配置
# 方案1：使用tensor_parallel_size=4，但启用pipeline_parallel_size=2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve /cpfs04/shared/kilab/hf-hub/Qwen2.5-7B-Instruct --port 8001 --served-model-name qwen-2 --tensor-parallel-size 4 --pipeline-parallel-size 2 --enable-lora --max-lora-rank 64 --max-loras 16
