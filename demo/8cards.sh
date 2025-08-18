# GPU 0
CUDA_VISIBLE_DEVICES=0 \
vllm serve /cpfs04/shared/kilab/hf-hub/Qwen2.5-7B-Instruct \
  --port 8001 \
  --served-model-name qwen-2 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes > vllm_gpu0.log 2>&1 &

# GPU 1
CUDA_VISIBLE_DEVICES=1 \
vllm serve /cpfs04/shared/kilab/hf-hub/Qwen2.5-7B-Instruct \
  --port 8002 \
  --served-model-name qwen-2 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes > vllm_gpu1.log 2>&1 &

# GPU 2
CUDA_VISIBLE_DEVICES=2 \
vllm serve /cpfs04/shared/kilab/hf-hub/Qwen2.5-7B-Instruct \
  --port 8003 \
  --served-model-name qwen-2 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes > vllm_gpu2.log 2>&1 &

# GPU 3
CUDA_VISIBLE_DEVICES=3 \
vllm serve /cpfs04/shared/kilab/hf-hub/Qwen2.5-7B-Instruct \
  --port 8004 \
  --served-model-name qwen-2 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes > vllm_gpu3.log 2>&1 &

# GPU 4
CUDA_VISIBLE_DEVICES=4 \
vllm serve /cpfs04/shared/kilab/hf-hub/Qwen2.5-7B-Instruct \
  --port 8005 \
  --served-model-name qwen-2 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes > vllm_gpu4.log 2>&1 &

# GPU 5
CUDA_VISIBLE_DEVICES=5 \
vllm serve /cpfs04/shared/kilab/hf-hub/Qwen2.5-7B-Instruct \
  --port 8006 \
  --served-model-name qwen-2 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes > vllm_gpu5.log 2>&1 &

# GPU 6
CUDA_VISIBLE_DEVICES=6 \
vllm serve /cpfs04/shared/kilab/hf-hub/Qwen2.5-7B-Instruct \
  --port 8007 \
  --served-model-name qwen-2 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes > vllm_gpu6.log 2>&1 &

# GPU 7
CUDA_VISIBLE_DEVICES=7 \
vllm serve /cpfs04/shared/kilab/hf-hub/Qwen2.5-7B-Instruct \
  --port 8008 \
  --served-model-name qwen-2 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes > vllm_gpu7.log 2>&1 &
