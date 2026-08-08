#!/usr/bin/env bash

set -Eeuo pipefail

###############################################################################
# TradingAgents + Local Qwen2.5-72B + vLLM
#
# GPU:
#   RTX PRO 6000 96GB
#
# Repository:
#   https://github.com/talkdev/TradingAgents
#
# This script:
#   1. Installs system dependencies
#   2. Clones your TradingAgents repository
#   3. Creates Python virtual environment
#   4. Installs TradingAgents
#   5. Installs vLLM
#   6. Downloads Qwen2.5-72B-Instruct-AWQ
#   7. Configures TradingAgents for local vLLM
#   8. Starts Qwen
#   9. Tests Qwen
#  10. Runs trading.py
###############################################################################

echo
echo "=============================================================="
echo " TradingAgents Local AI Deployment"
echo " RTX PRO 6000 96GB"
echo " Qwen2.5-72B-Instruct-AWQ"
echo "=============================================================="
echo

###############################################################################
# CONFIGURATION
###############################################################################

BASE_DIR="${HOME}/TradingAI"

REPO_URL="https://github.com/talkdev/TradingAgents.git"

TRADINGAGENTS_DIR="${BASE_DIR}/TradingAgents"
VENV_DIR="${BASE_DIR}/venv"
MODEL_DIR="${BASE_DIR}/models/Qwen2.5-72B-Instruct-AWQ"
REPORT_DIR="${TRADINGAGENTS_DIR}/reports"

MODEL_REPO="Qwen/Qwen2.5-72B-Instruct-AWQ"
MODEL_NAME="Qwen2.5-72B-Instruct-AWQ"

VLLM_HOST="127.0.0.1"
VLLM_PORT="8000"

GPU_MEMORY_UTILIZATION="0.92"
MAX_MODEL_LEN="16384"
MAX_NUM_SEQS="6"

MAX_WORKERS="6"

###############################################################################
# 1. GPU CHECK
###############################################################################

echo "[1/10] Checking GPU..."

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "ERROR: NVIDIA driver / nvidia-smi not available."
    exit 1
fi

nvidia-smi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo
echo "Detected GPU count: ${GPU_COUNT}"

if [ "${GPU_COUNT}" -ne 1 ]; then
    echo
    echo "WARNING: This setup is designed for 1 GPU."
    echo "Detected ${GPU_COUNT} GPUs."
fi

###############################################################################
# 2. SYSTEM PACKAGES
###############################################################################

echo
echo "[2/10] Installing system packages..."

sudo apt-get update

sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    tmux \
    htop \
    nvtop \
    jq

###############################################################################
# 3. CREATE BASE DIRECTORY
###############################################################################

echo
echo "[3/10] Creating directories..."

mkdir -p "${BASE_DIR}"
mkdir -p "${BASE_DIR}/models"
mkdir -p "${REPORT_DIR}"

###############################################################################
# 4. CLONE YOUR REPOSITORY
###############################################################################

echo
echo "[4/10] Cloning TradingAgents repository..."

if [ -d "${TRADINGAGENTS_DIR}/.git" ]; then

    echo "Repository already exists."
    echo "Updating repository..."

    cd "${TRADINGAGENTS_DIR}"

    git fetch origin
    git checkout main
    git pull --ff-only

else

    rm -rf "${TRADINGAGENTS_DIR}"

    git clone \
        "${REPO_URL}" \
        "${TRADINGAGENTS_DIR}"

fi

echo
echo "Repository:"
echo "${TRADINGAGENTS_DIR}"

cd "${TRADINGAGENTS_DIR}"

echo
echo "Latest commit:"
git log -1 --oneline

###############################################################################
# 5. PYTHON ENVIRONMENT
###############################################################################

echo
echo "[5/10] Creating Python virtual environment..."

if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

python --version

pip install --upgrade pip setuptools wheel

###############################################################################
# 6. PYTORCH + vLLM + TRADINGAGENTS
###############################################################################

echo
echo "[6/10] Installing Python dependencies..."

pip install \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

echo
echo "Installing vLLM..."

pip install vllm

echo
echo "Installing TradingAgents..."

cd "${TRADINGAGENTS_DIR}"

pip install -e .

echo
echo "Installing Hugging Face tools..."

pip install \
    huggingface_hub \
    transformers \
    accelerate \
    safetensors \
    sentencepiece

###############################################################################
# 7. VERIFY PYTORCH / GPU
###############################################################################

echo
echo "Checking PyTorch..."

python - <<'PY'

import torch

print()
print("==============================================")
print(" PyTorch / GPU")
print("==============================================")

print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print("GPU             :", torch.cuda.get_device_name(0))

vram = torch.cuda.get_device_properties(0).total_memory / 1024**3

print("VRAM            :", round(vram, 2), "GB")

if vram < 80:
    raise RuntimeError(
        f"GPU has only {vram:.1f} GB VRAM. "
        "This deployment expects a high-memory GPU."
    )

print("==============================================")
print()

PY

###############################################################################
# 8. DOWNLOAD QWEN
###############################################################################

echo
echo "[7/10] Downloading Qwen2.5-72B-Instruct-AWQ..."

if [ ! -f "${MODEL_DIR}/config.json" ]; then

    echo
    echo "Downloading:"
    echo "${MODEL_REPO}"
    echo

    huggingface-cli download \
        "${MODEL_REPO}" \
        --local-dir "${MODEL_DIR}"

else

    echo "Qwen model already downloaded."
    echo "Skipping download."

fi

###############################################################################
# 9. CONFIGURE TRADINGAGENTS
###############################################################################

echo
echo "[8/10] Configuring TradingAgents..."

cd "${TRADINGAGENTS_DIR}"

cat > .env <<EOF
###############################################################################
# Local Qwen/vLLM configuration
###############################################################################

OPENAI_API_KEY=dummy

OPENAI_COMPATIBLE_API_KEY=dummy

TRADINGAGENTS_LLM_PROVIDER=openai_compatible

TRADINGAGENTS_DEEP_THINK_LLM=${MODEL_NAME}

TRADINGAGENTS_QUICK_THINK_LLM=${MODEL_NAME}

TRADINGAGENTS_LLM_BACKEND_URL=http://${VLLM_HOST}:${VLLM_PORT}/v1

TRADINGAGENTS_OUTPUT_LANGUAGE=English

TRADINGAGENTS_MAX_DEBATE_ROUNDS=2

TRADINGAGENTS_MAX_RISK_ROUNDS=1

TRADINGAGENTS_TEMPERATURE=0.1

MAX_WORKERS=${MAX_WORKERS}

OPENAI_BASE_URL=http://${VLLM_HOST}:${VLLM_PORT}/v1
EOF

chmod 600 .env

###############################################################################
# FIX USER SCRIPT CONFIGURATION
###############################################################################

echo
echo "Checking trading.py..."

if [ ! -f "${TRADINGAGENTS_DIR}/trading.py" ]; then

    echo
    echo "ERROR: trading.py was not found in repository."
    echo
    exit 1

fi

###############################################################################
# Patch Windows paths in trading.py
###############################################################################

python - <<PY

from pathlib import Path

path = Path("${TRADINGAGENTS_DIR}/trading.py")

text = path.read_text(encoding="utf-8")

# Windows TradingAgents path
text = text.replace(
    r"C:\Users\Administrator\Desktop\TradingAgents",
    "${TRADINGAGENTS_DIR}"
)

# Windows reports path
text = text.replace(
    r"C:\Users\Administrator\Desktop\TradingAgents\reports",
    "${REPORT_DIR}"
)

# Local model instead of OpenAI models
text = text.replace(
    'config["deep_think_llm"] = "gpt-5.5"',
    'config["deep_think_llm"] = "${MODEL_NAME}"'
)

text = text.replace(
    'config["quick_think_llm"] = "gpt-5.4-mini"',
    'config["quick_think_llm"] = "${MODEL_NAME}"'
)

# Local provider
text = text.replace(
    'config["llm_provider"] = "openai"',
    'config["llm_provider"] = "openai_compatible"'
)

# Direct summary call
text = text.replace(
    'model="gpt-4o"',
    'model="${MODEL_NAME}"'
)

# Remove hard-coded OpenAI key if present.
import re

text = re.sub(
    r'os\.environ\["OPENAI_API_KEY"\]\s*=\s*os\.getenv\([\s\S]*?\)\s*',
    'os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "dummy")\n',
    text,
    count=1
)

path.write_text(text, encoding="utf-8")

print("trading.py configured for local Qwen.")

PY

###############################################################################
# 10. START vLLM
###############################################################################

echo
echo "[9/10] Starting Qwen vLLM..."

cat > "${BASE_DIR}/start_vllm.sh" <<EOF
#!/usr/bin/env bash

set -Eeuo pipefail

source "${VENV_DIR}/bin/activate"

exec vllm serve "${MODEL_DIR}" \\
    --host "${VLLM_HOST}" \\
    --port "${VLLM_PORT}" \\
    --tensor-parallel-size 1 \\
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \\
    --max-model-len ${MAX_MODEL_LEN} \\
    --max-num-seqs ${MAX_NUM_SEQS} \\
    --enable-prefix-caching
EOF

chmod +x "${BASE_DIR}/start_vllm.sh"

if pgrep -f "vllm serve" >/dev/null 2>&1; then

    echo "vLLM already running."

else

    nohup "${BASE_DIR}/start_vllm.sh" \
        > "${BASE_DIR}/vllm.log" \
        2>&1 &

    echo "vLLM started."

fi

###############################################################################
# WAIT FOR SERVER
###############################################################################

echo
echo "Waiting for Qwen..."

READY=0

for i in $(seq 1 120); do

    if curl -sf \
        "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" \
        >/dev/null 2>&1; then

        READY=1
        break

    fi

    printf "."
    sleep 5

done

echo

if [ "${READY}" -ne 1 ]; then

    echo
    echo "ERROR: Qwen did not start successfully."
    echo
    echo "Last 100 lines of vLLM log:"
    echo
    tail -100 "${BASE_DIR}/vllm.log"

    exit 1

fi

###############################################################################
# TEST QWEN
###############################################################################

echo
echo "[10/10] Testing Qwen..."

curl -sf \
    "http://${VLLM_HOST}:${VLLM_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL_NAME}\",
        \"messages\": [
            {
                \"role\": \"user\",
                \"content\": \"Reply with exactly QWEN_READY\"
            }
        ],
        \"temperature\": 0,
        \"max_tokens\": 20
    }" | jq .

###############################################################################
# RUN TRADINGAGENTS
###############################################################################

echo
echo
echo "=============================================================="
echo " QWEN IS READY"
echo "=============================================================="
echo
echo "Repository : ${TRADINGAGENTS_DIR}"
echo "Model      : ${MODEL_NAME}"
echo "GPU        : RTX PRO 6000 96GB"
echo "Tensor TP  : 1"
echo "Max Seq    : 6"
echo "Workers    : 6"
echo "API        : http://${VLLM_HOST}:${VLLM_PORT}/v1"
echo
echo "Starting TradingAgents..."
echo "=============================================================="
echo

cd "${TRADINGAGENTS_DIR}"

source "${VENV_DIR}/bin/activate"

export OPENAI_API_KEY=dummy
export OPENAI_COMPATIBLE_API_KEY=dummy

export OPENAI_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"

export TRADINGAGENTS_LLM_PROVIDER="openai_compatible"

export TRADINGAGENTS_DEEP_THINK_LLM="${MODEL_NAME}"

export TRADINGAGENTS_QUICK_THINK_LLM="${MODEL_NAME}"

export TRADINGAGENTS_LLM_BACKEND_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"

export TRADINGAGENTS_OUTPUT_LANGUAGE="English"

python trading.py

echo
echo "=============================================================="
echo " TradingAgents execution completed"
echo "=============================================================="