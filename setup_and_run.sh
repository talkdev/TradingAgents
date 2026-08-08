#!/usr/bin/env bash
set -Eeuo pipefail

BASE_DIR="/root/TradingAI"
REPO_DIR="${BASE_DIR}/TradingAgents"
REPO_URL="https://github.com/talkdev/TradingAgents.git"
REPO_BRANCH="main"
VENV="${BASE_DIR}/venv"
MODEL_NAME="Qwen2.5-72B-Instruct-AWQ"
MODEL_REPO="Qwen/Qwen2.5-72B-Instruct-AWQ"
MODEL_DIR="${BASE_DIR}/models/${MODEL_NAME}"
PYTHON="${VENV}/bin/python"
PIP="${VENV}/bin/pip"
VLLM="${VENV}/bin/vllm"
HF="${VENV}/bin/hf"
HOST="127.0.0.1"
PORT="8000"
API_KEY="local-dummy-key"
TP_SIZE=1
MAX_MODEL_LEN=16384
MAX_NUM_SEQS=6
MAX_WORKERS=6
GPU_UTILIZATION=0.90
VLLM_PID="${BASE_DIR}/vllm.pid"
VLLM_LOG="${BASE_DIR}/vllm.log"

log() {
    echo
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

die() {
    echo
    echo "ERROR: $*"
    exit 1
}

[ "$(id -u)" -eq 0 ] || die "Run this script as root."

mkdir -p "${BASE_DIR}" "${BASE_DIR}/models"

# -----------------------------------------------------------------------------
# GPU
# -----------------------------------------------------------------------------

log "Checking GPU"

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found."

nvidia-smi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

echo "GPU       : ${GPU_NAME}"
echo "GPU count : ${GPU_COUNT}"
echo "VRAM      : ${GPU_VRAM} MiB"

[ "${GPU_COUNT}" -eq 1 ] || die "This setup requires exactly one GPU."

# -----------------------------------------------------------------------------
# System packages
# -----------------------------------------------------------------------------

log "Installing system packages"

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git curl wget jq procps psmisc util-linux \
    build-essential python3 python3-pip python3-venv python3-dev

# -----------------------------------------------------------------------------
# Repository
# -----------------------------------------------------------------------------

log "Preparing TradingAgents repository"

if [ -d "${REPO_DIR}/.git" ]; then
    cd "${REPO_DIR}"
    echo "Existing repository found."
    if git diff --quiet && git diff --cached --quiet; then
        git fetch origin
        git checkout "${REPO_BRANCH}"
        git pull --ff-only origin "${REPO_BRANCH}"
    else
        echo "WARNING: Local Git changes detected."
        echo "Keeping local changes; repository will not be reset."
    fi
else
    if [ -e "${REPO_DIR}" ]; then
        die "${REPO_DIR} exists but is not a Git repository."
    fi
    git clone --branch "${REPO_BRANCH}" --single-branch "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"

echo
echo "Repository : ${REPO_DIR}"
echo "Remote     : $(git remote get-url origin)"
echo "Commit     : $(git rev-parse --short HEAD)"

# -----------------------------------------------------------------------------
# Python environment
# -----------------------------------------------------------------------------

log "Preparing Python environment"

if [ ! -x "${PYTHON}" ]; then
    python3 -m venv "${VENV}"
fi

"${PYTHON}" --version

"${PIP}" install --upgrade pip wheel
"${PIP}" install --force-reinstall "setuptools>=77.0.3,<81.0.0"

# -----------------------------------------------------------------------------
# PyTorch
# -----------------------------------------------------------------------------

log "Checking PyTorch"

if ! "${PYTHON}" -c "import torch" >/dev/null 2>&1; then
    "${PIP}" install torch
fi

"${PYTHON}" - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available.")

print("GPU:", torch.cuda.get_device_name(0))
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")
PY

# -----------------------------------------------------------------------------
# vLLM
# -----------------------------------------------------------------------------

log "Checking vLLM"

if ! "${PYTHON}" -c "import vllm" >/dev/null 2>&1; then
    "${PIP}" install vllm
fi

"${PIP}" install --force-reinstall "setuptools>=77.0.3,<81.0.0"

# -----------------------------------------------------------------------------
# Hugging Face CLI
# -----------------------------------------------------------------------------

log "Installing Hugging Face CLI"

"${PIP}" install --upgrade "huggingface_hub[cli]"

[ -x "${HF}" ] || die "Hugging Face CLI not found at ${HF}."

"${HF}" --version

# -----------------------------------------------------------------------------
# TradingAgents dependencies
# -----------------------------------------------------------------------------

log "Installing TradingAgents"

cd "${REPO_DIR}"

"${PIP}" install \
    transformers \
    accelerate \
    safetensors \
    sentencepiece

"${PIP}" install -e .

# TradingAgents installation may change setuptools.
"${PIP}" install --force-reinstall "setuptools>=77.0.3,<81.0.0"

echo
echo "Installed versions:"
"${PIP}" show setuptools | grep '^Version:' || true
"${PIP}" show torch | grep '^Version:' || true
"${PIP}" show vllm | grep '^Version:' || true

# -----------------------------------------------------------------------------
# Qwen model
# -----------------------------------------------------------------------------

log "Checking Qwen model"

if [ -f "${MODEL_DIR}/config.json" ]; then
    echo "Existing Qwen model found:"
    echo "${MODEL_DIR}"
    du -sh "${MODEL_DIR}"
else
    log "Qwen model not found. Downloading ${MODEL_REPO}"
    mkdir -p "${MODEL_DIR}"
    "${HF}" download \
        "${MODEL_REPO}" \
        --local-dir "${MODEL_DIR}"
fi

[ -f "${MODEL_DIR}/config.json" ] || die "Qwen model is incomplete."

echo
echo "Qwen model:"
echo "${MODEL_DIR}"
du -sh "${MODEL_DIR}"

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

log "Creating TradingAgents environment"

cd "${REPO_DIR}"

cat > .env <<EOF
OPENAI_API_KEY=${API_KEY}
OPENAI_COMPATIBLE_API_KEY=${API_KEY}
OPENAI_BASE_URL=http://${HOST}:${PORT}/v1
TRADINGAGENTS_LLM_PROVIDER=openai_compatible
TRADINGAGENTS_DEEP_THINK_LLM=${MODEL_NAME}
TRADINGAGENTS_QUICK_THINK_LLM=${MODEL_NAME}
TRADINGAGENTS_LLM_BACKEND_URL=http://${HOST}:${PORT}/v1
TRADINGAGENTS_OUTPUT_LANGUAGE=English
MAX_WORKERS=${MAX_WORKERS}
EOF

chmod 600 .env

# -----------------------------------------------------------------------------
# Stop Qwen
# -----------------------------------------------------------------------------

log "Creating stop_qwen.sh"

cat > "${BASE_DIR}/stop_qwen.sh" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

BASE_DIR="/root/TradingAI"
PID_FILE="${BASE_DIR}/vllm.pid"

echo "Stopping Qwen/vLLM..."

if [ -f "${PID_FILE}" ]; then
    PID=$(cat "${PID_FILE}" 2>/dev/null || true)
    if [ -n "${PID}" ] && kill -0 "${PID}" 2>/dev/null; then
        echo "Stopping PID ${PID}"
        kill -TERM "${PID}" 2>/dev/null || true
        sleep 5
        kill -KILL "${PID}" 2>/dev/null || true
    fi
    rm -f "${PID_FILE}"
fi

pkill -TERM -f 'vllm serve' 2>/dev/null || true
pkill -TERM -f 'VLLM::EngineCore' 2>/dev/null || true
sleep 3
pkill -KILL -f 'vllm serve' 2>/dev/null || true
pkill -KILL -f 'VLLM::EngineCore' 2>/dev/null || true
sleep 3

nvidia-smi
EOF

chmod 755 "${BASE_DIR}/stop_qwen.sh"

# -----------------------------------------------------------------------------
# Start Qwen
# -----------------------------------------------------------------------------

log "Creating start_qwen.sh"

cat > "${BASE_DIR}/start_qwen.sh" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail

BASE_DIR="${BASE_DIR}"
VLLM="${VLLM}"
MODEL_DIR="${MODEL_DIR}"
MODEL_NAME="${MODEL_NAME}"
HOST="${HOST}"
PORT="${PORT}"
API_KEY="${API_KEY}"
PID_FILE="${VLLM_PID}"
LOG_FILE="${VLLM_LOG}"

export VLLM_USE_FLASHINFER_SAMPLER=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

if curl -fsS --max-time 5 \
    "http://${HOST}:${PORT}/v1/models" \
    -H "Authorization: Bearer ${API_KEY}" \
    >/dev/null 2>&1; then
    log "Qwen is already running."
    exit 0
fi

log "Cleaning stale vLLM processes."

if [ -f "${PID_FILE}" ]; then
    PID=$(cat "${PID_FILE}" 2>/dev/null || true)
    if [ -n "\${PID}" ] && kill -0 "\${PID}" 2>/dev/null; then
        kill -TERM "\${PID}" 2>/dev/null || true
        sleep 5
        kill -KILL "\${PID}" 2>/dev/null || true
    fi
    rm -f "\${PID_FILE}"
fi

pkill -TERM -f 'vllm serve' 2>/dev/null || true
pkill -TERM -f 'VLLM::EngineCore' 2>/dev/null || true
sleep 3
pkill -KILL -f 'vllm serve' 2>/dev/null || true
pkill -KILL -f 'VLLM::EngineCore' 2>/dev/null || true
sleep 3

GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)

echo "GPU memory currently used: ${GPU_USED} MiB"

if [ "${GPU_USED}" -gt 5000 ]; then
    echo "ERROR: GPU still has ${GPU_USED} MiB allocated."
    nvidia-smi
    exit 1
fi

[ -f "${MODEL_DIR}/config.json" ] || {
    echo "ERROR: Model not found at ${MODEL_DIR}"
    exit 1
}

for GPU_UTIL in 0.90 0.85 0.80; do
    log "Starting Qwen with GPU utilization ${GPU_UTIL}"
    : > "\${LOG_FILE}"

    nohup "\${VLLM}" serve "\${MODEL_DIR}" \
        --host "\${HOST}" \
        --port "\${PORT}" \
        --api-key "\${API_KEY}" \
        --served-model-name "\${MODEL_NAME}" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization "\${GPU_UTIL}" \
        --max-model-len 16384 \
        --max-num-seqs 6 \
        --enable-prefix-caching \
        >"\${LOG_FILE}" 2>&1 &

    PID=\$!
    echo "\${PID}" > "\${PID_FILE}"

    log "vLLM PID: \${PID}"

    READY=0

    for i in \$(seq 1 180); do
        if curl -fsS --max-time 5 \
            "http://\${HOST}:\${PORT}/v1/models" \
            -H "Authorization: Bearer \${API_KEY}" \
            >/dev/null 2>&1; then
            READY=1
            break
        fi

        if ! kill -0 "\${PID}" 2>/dev/null; then
            break
        fi

        sleep 2
    done

    if [ "\${READY}" -eq 1 ]; then
        log "Qwen API is ready."

        RESPONSE=\$(curl -fsS --max-time 120 \
            "http://\${HOST}:\${PORT}/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -H "Authorization: Bearer \${API_KEY}" \
            -d '{
                "model": "'"${MODEL_NAME}"'",
                "messages": [
                    {
                        "role": "user",
                        "content": "Reply with exactly QWEN_READY"
                    }
                ],
                "temperature": 0,
                "max_tokens": 20
            }')

        echo "\${RESPONSE}" | jq .

        if echo "\${RESPONSE}" | grep -q "QWEN_READY"; then
            echo
            echo "============================================================"
            echo " QWEN READY"
            echo "============================================================"
            echo "Model       : \${MODEL_NAME}"
            echo "Tensor      : 1"
            echo "Context     : 16384"
            echo "Sequences   : 6"
            echo "GPU Util    : \${GPU_UTIL}"
            echo "API         : http://\${HOST}:\${PORT}/v1"
            echo "PID         : \${PID}"
            echo "Log         : \${LOG_FILE}"
            echo "============================================================"
            exit 0
        fi
    fi

    echo
    echo "Qwen startup failed."
    tail -100 "\${LOG_FILE}" || true

    "\${BASE_DIR}/stop_qwen.sh" || true

    GPU_USED=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)

    if [ "\${GPU_USED}" -gt 5000 ]; then
        echo "GPU remains occupied after cleanup."
        nvidia-smi
        exit 1
    fi
done

echo
echo "Qwen failed after all startup attempts."
echo "Full log: ${LOG_FILE}"
tail -200 "${LOG_FILE}" || true
exit 1
EOF

chmod 755 "${BASE_DIR}/start_qwen.sh"

# -----------------------------------------------------------------------------
# Status
# -----------------------------------------------------------------------------

log "Creating status.sh"

cat > "${BASE_DIR}/status.sh" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

echo "================ GPU ================"
nvidia-smi

echo
echo "================ vLLM ================"
ps -ef | grep -i '[v]llm' || true

echo
echo "================ API ================"

if curl -fsS --max-time 5 \
    http://127.0.0.1:8000/v1/models \
    -H 'Authorization: Bearer local-dummy-key' \
    >/tmp/qwen_models.json 2>/dev/null; then
    jq . /tmp/qwen_models.json
    echo
    echo "STATUS: READY"
else
    echo "STATUS: NOT READY"
fi
EOF

chmod 755 "${BASE_DIR}/status.sh"

# -----------------------------------------------------------------------------
# TradingAgents runner
# -----------------------------------------------------------------------------

log "Creating run_trading.sh"

cat > "${BASE_DIR}/run_trading.sh" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail

BASE_DIR="${BASE_DIR}"
REPO_DIR="${REPO_DIR}"
PYTHON="${PYTHON}"
API_URL="http://${HOST}:${PORT}/v1"
API_KEY="${API_KEY}"
MODEL_NAME="${MODEL_NAME}"

if ! curl -fsS --max-time 5 \
    "${API_URL}/models" \
    -H "Authorization: Bearer ${API_KEY}" \
    >/dev/null 2>&1; then
    echo "Qwen is not running. Starting it..."
    "${BASE_DIR}/start_qwen.sh"
fi

curl -fsS --max-time 10 \
    "${API_URL}/models" \
    -H "Authorization: Bearer ${API_KEY}" \
    >/dev/null \
    || {
        echo "ERROR: Qwen API is unavailable."
        exit 1
    }

cd "${REPO_DIR}"

export OPENAI_API_KEY="${API_KEY}"
export OPENAI_BASE_URL="${API_URL}"
export OPENAI_COMPATIBLE_API_KEY="${API_KEY}"
export TRADINGAGENTS_LLM_PROVIDER="openai_compatible"
export TRADINGAGENTS_DEEP_THINK_LLM="${MODEL_NAME}"
export TRADINGAGENTS_QUICK_THINK_LLM="${MODEL_NAME}"
export TRADINGAGENTS_LLM_BACKEND_URL="${API_URL}"
export MAX_WORKERS="${MAX_WORKERS}"

echo
echo "============================================================"
echo " Starting TradingAgents"
echo "============================================================"
echo "Repository : ${REPO_DIR}"
echo "Model      : ${MODEL_NAME}"
echo "Workers    : ${MAX_WORKERS}"
echo "API        : ${API_URL}"
echo "============================================================"
echo

exec "${PYTHON}" trading.py
EOF

chmod 755 "${BASE_DIR}/run_trading.sh"

# -----------------------------------------------------------------------------
# Validate
# -----------------------------------------------------------------------------

log "Validating installation"

"${PYTHON}" - <<'PY'
import torch
import vllm
import setuptools

print("setuptools:", setuptools.__version__)
print("torch:", torch.__version__)
print("vLLM:", vllm.__version__)
print("CUDA:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable.")

print("GPU:", torch.cuda.get_device_name(0))
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")
PY

# -----------------------------------------------------------------------------
# Secret check
# -----------------------------------------------------------------------------

log "Checking repository for hard-coded OpenAI keys"

if grep -R \
    --exclude-dir=.git \
    --exclude-dir=venv \
    --exclude-dir=.venv \
    -nE 'sk-(proj-)?[A-Za-z0-9_-]{20,}' \
    "${REPO_DIR}" \
    >/tmp/tradingagents_secret_scan.txt 2>/dev/null; then
    echo
    cat /tmp/tradingagents_secret_scan.txt
    echo
    die "Possible hard-coded OpenAI API key found in repository."
fi

echo "No obvious sk-* API key found."

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo
echo "============================================================"
echo " SETUP COMPLETE"
echo "============================================================"
echo
echo "Installation : ${BASE_DIR}"
echo "Repository   : ${REPO_DIR}"
echo "Python       : ${VENV}"
echo "Qwen model   : ${MODEL_DIR}"
echo
echo "Scripts:"
echo "  ${BASE_DIR}/start_qwen.sh"
echo "  ${BASE_DIR}/stop_qwen.sh"
echo "  ${BASE_DIR}/status.sh"
echo "  ${BASE_DIR}/run_trading.sh"
echo
echo "============================================================"

# -----------------------------------------------------------------------------
# Start Qwen and validate inference
# -----------------------------------------------------------------------------

log "Starting Qwen"

"${BASE_DIR}/start_qwen.sh"

echo
echo "============================================================"
echo " QWEN SETUP COMPLETE"
echo "============================================================"
echo
echo "Start Qwen:"
echo "  ${BASE_DIR}/start_qwen.sh"
echo
echo "Stop Qwen:"
echo "  ${BASE_DIR}/stop_qwen.sh"
echo
echo "Status:"
echo "  ${BASE_DIR}/status.sh"
echo
echo "Run TradingAgents:"
echo "  ${BASE_DIR}/run_trading.sh"
echo
echo "============================================================"