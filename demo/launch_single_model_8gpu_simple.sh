#!/usr/bin/env bash

# 简单的一行命令启动单模型+8GPU vLLM
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve /cpfs04/shared/kilab/hf-hub/Qwen2.5-7B-Instruct --port 8001 --served-model-name qwen-2 --tensor-parallel-size 8 --enable-lora --max-lora-rank 64 --max-loras 16
