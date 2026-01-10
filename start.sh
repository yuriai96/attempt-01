#!/usr/bin/env bash
set -eo pipefail

log() {
  echo "[`date '+%Y-%m-%d %H:%M:%S'`] $1"
}

log "===== Container starting ====="

# -------- Start render-service (background) --------
log "Starting render-service (uvicorn) on port 8000"
cd /workspace/external/404-gen-subnet/render-service
uvicorn render_service:app --host 0.0.0.0 --port 8000 &
RENDER_PID=$!
log "render-service started (PID=$RENDER_PID)"

# -------- Start vLLM (background) --------
log "Starting vLLM on port 8001"
/opt/vllm-env/bin/vllm serve zai-org/GLM-4.1V-9B-Thinking \
  --gpu-memory-utilization 0.4 \
  --port 8001 &
VLLM_PID=$!
log "Waiting for vLLM to be ready on port 8001 (PID=$VLLM_PID)"
until (echo > /dev/tcp/127.0.0.1/8001) >/dev/null 2>&1; do
  sleep 1
done
log "vLLM is ready!"


# -------- Run serve.py (FOREGROUND, LAST) --------
log "Starting serve.py (foreground) at /workspace"
cd /workspace
exec python serve.py
