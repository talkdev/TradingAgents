#!/bin/bash
# FlashInfer sampler workaround
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "[+] Starting vLLM server with Qwen2.5-72B-Instruct-AWQ..."

/root/TradingAI/venv/bin/vllm serve /root/TradingAI/models/Qwen2.5-72B-Instruct-AWQ \
    --host 127.0.0.1 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --max-num-seqs 6 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --api-key local-dummy-key