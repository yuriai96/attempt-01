#!/bin/bash
set -e

VLLM_HOST=${VLLM_HOST:-"0.0.0.0"}
VLLM_PORT=${VLLM_PORT:-8095}
MAX_MODEL_LEN=5120
GPU_UTIL=0.275

VLLM_MODEL=${VLLM_MODEL:-"zai-org/GLM-4.1V-9B-Thinking"}
API_KEY=${VLLM_API_KEY:-"local"}

echo "-----------------------------------------------------"
echo "🚀 STARTING VLLM SERVER"
echo "   Model: $VLLM_MODEL"
echo "   Port: $VLLM_PORT"
echo "   GPU Util: $GPU_UTIL"
echo "-----------------------------------------------------"

vllm serve "zai-org/GLM-4.1V-9B-Thinking" \
    --port $VLLM_PORT \
    --api-key $API_KEY \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_UTIL \
    --tensor-parallel-size 1 \
    --max_num_seqs 2
