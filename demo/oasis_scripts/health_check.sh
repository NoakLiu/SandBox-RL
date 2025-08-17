#!/usr/bin/env bash
set -euo pipefail

BASE_PORT=8001
NUM_INSTANCES=8

for i in $(seq 0 $((NUM_INSTANCES-1))); do
  PORT=$((BASE_PORT + i))
  echo -n "Port $PORT: "
  curl --http1.1 -s "http://127.0.0.1:${PORT}/health" || echo "unreachable"
  echo
done
