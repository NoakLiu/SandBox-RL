#!/usr/bin/env bash
set -euo pipefail

# 尝试优雅结束所有 vLLM serve 进程
pkill -f "vllm serve" || true

echo "✅ All vLLM instances stopped."
