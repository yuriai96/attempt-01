#!/bin/bash
set -e

# --- Cấu hình mặc định ---
VLLM_HOST=${VLLM_HOST:-"0.0.0.0"}
VLLM_PORT=${VLLM_PORT:-8095}
VLLM_MODEL=${VLLM_MODEL:-"THUDM/GLM-4.1V-9B-Thinking"}
GPU_UTIL=${VLLM_GPU_MEMORY_UTILIZATION:-0.275}
API_KEY=${VLLM_API_KEY:-"local"}

echo "-----------------------------------------------------"
echo "🚀 STARTING VLLM SERVER (Isolated Env)"
echo "   Model: $VLLM_MODEL"
echo "   Host : $VLLM_HOST"
echo "   Port : $VLLM_PORT"
echo "   GPU Util: $GPU_UTIL"
echo "-----------------------------------------------------"

# 1. Khởi chạy vLLM ở chế độ background (&)
/opt/vllm-env/bin/vllm serve "$VLLM_MODEL" \
  --revision "17193d2147da3acd0da358eb251ef862b47e7545" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --api-key "$API_KEY" \
  --max-model-len 8096 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max_num_seqs 2 &

# Lưu lại Process ID của vLLM
VLLM_PID=$!

# 2. Vòng lặp đợi vLLM khởi động xong (Health Check)
echo "⏳ Waiting for vLLM to become ready..."
MAX_RETRIES=150
COUNTER=0

while [ "$COUNTER" -lt "$MAX_RETRIES" ]; do
  if curl -s -f "http://localhost:$VLLM_PORT/health" > /dev/null; then
    echo "✅ vLLM is READY!"
    break
  fi

  echo "   ... loading model ($COUNTER/$MAX_RETRIES)"
  sleep 5
  COUNTER=$((COUNTER + 1))
done

if [ "$COUNTER" -eq "$MAX_RETRIES" ]; then
  echo "❌ vLLM failed to start within timeout."
  echo "   Hint: check vLLM logs / stdout, or /var/log/vllm.log if you redirect logs there."
  kill "$VLLM_PID" || true
  exit 1
fi


echo "🔥 Warming up vLLM with a small chat request..."
WARMUP_RETRIES=10
WARMUP_COUNTER=0

while [ "$WARMUP_COUNTER" -lt "$WARMUP_RETRIES" ]; do
  if curl -sS -f --max-time 30 "http://localhost:$VLLM_PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -d '{
      "model": "'"$VLLM_MODEL"'",
      "messages": [{"role":"user","content":"Hello, are you running?"}],
      "max_tokens": 16,
      "temperature": 0
    }' > /dev/null; then
      echo "✅ Warmup done!"
      break
  fi

  echo "   ... warmup retry ($WARMUP_COUNTER/$WARMUP_RETRIES)"
  sleep 3
  WARMUP_COUNTER=$((WARMUP_COUNTER + 1))
done

if [ "$WARMUP_COUNTER" -eq "$WARMUP_RETRIES" ]; then
  echo "⚠️ Warmup failed after retries, continuing anyway..."
fi

echo "-----------------------------------------------------"
echo "🚀 STARTING MAIN FASTAPI SERVICE (Base Env)"
echo "-----------------------------------------------------"

# 3. Khởi chạy App chính (Foreground)
exec python serve.py