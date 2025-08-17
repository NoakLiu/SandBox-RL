#!/usr/bin/env bash
set -euo pipefail
BASE_PORT=8001

for i in $(seq 0 7); do
  PORT=$((BASE_PORT + i))
  echo -n "Checking :${PORT} ... "
  OUT=$(curl --http1.1 -s "http://127.0.0.1:${PORT}/health" || true)
  echo "$OUT"
done
