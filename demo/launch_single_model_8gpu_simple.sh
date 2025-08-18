#!/usr/bin/env bash

# 简单的一行命令启动单模型+8GPU vLLM
# 注意：Qwen2.5-7B-Instruct有28个注意力头，不能被8整除，所以使用tensor_parallel_size=4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve /cpfs04/shared/kilab/hf-hub/Qwen2.5-7B-Instruct --port 8001 --served-model-name qwen-2 --tensor-parallel-size 4 --enable-lora --max-lora-rank 64 --max-loras 16
