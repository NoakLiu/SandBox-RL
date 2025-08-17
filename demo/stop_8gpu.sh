#!/usr/bin/env bash
set -euo pipefail

# 尝试优雅停止
pkill -f "vllm serve" || true

# 如需强制（通常不需要），可以取消下面注释
# pkill -9 -f "vllm serve" || true

echo "Stopped all vLLM instances."
