#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:?Usage: start_agent.sh <model_name> <gpu_id> [--no-wait]}"
GPU_ID="${2:-0}"
NO_WAIT="${3:-}"
DTYPE="${AGENT_DTYPE:-auto}"
BACKEND="${AGENT_BACKEND:-transformers}"
MODEL_FILE="${AGENT_MODEL_FILE:-}"
SESSION_NAME="autoresearch-agent"
PORT=8000

# Activate conda env — always prefer miniforge3
if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
    for _CP in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
        if [[ -f "${_CP}/etc/profile.d/conda.sh" ]]; then
            source "${_CP}/etc/profile.d/conda.sh"
            break
        fi
    done
fi
conda activate autoresearch
[[ -f "$CONDA_PREFIX/etc/conda/activate.d/cuda-libs.sh" ]] && source "$CONDA_PREFIX/etc/conda/activate.d/cuda-libs.sh"

# Kill existing session if running (allows model switch)
if screen -list | grep -q "$SESSION_NAME"; then
    screen -S "$SESSION_NAME" -X quit 2>/dev/null || true
    sleep 2
    echo "Stopped existing agent session"
fi

if [[ "$BACKEND" == "llamacpp" ]]; then
    # llama.cpp backend — download GGUF from HuggingFace, serve via llama-cpp-python
    MODELS_DIR="$HOME/autoresearch/models"
    mkdir -p "$MODELS_DIR"
    GGUF_PATH="$MODELS_DIR/$MODEL_FILE"

    # Download + serve inside screen (download can take 10+ min for large models)
    screen -dmS "$SESSION_NAME" bash -c "source \$HOME/miniforge3/etc/profile.d/conda.sh 2>/dev/null || eval \"\$(conda shell.bash hook)\"; conda activate autoresearch; [[ -f \$CONDA_PREFIX/etc/conda/activate.d/cuda-libs.sh ]] && source \$CONDA_PREFIX/etc/conda/activate.d/cuda-libs.sh; if [ ! -f $GGUF_PATH ]; then echo 'Downloading $MODEL_FILE...'; python -c \"from huggingface_hub import hf_hub_download; hf_hub_download('$MODEL', '$MODEL_FILE', local_dir='$MODELS_DIR')\"; echo 'Downloaded'; fi; CUDA_VISIBLE_DEVICES=$GPU_ID python -m llama_cpp.server --model $GGUF_PATH --port $PORT --n_gpu_layers -1 --chat_format chatml 2>&1 | tee ~/autoresearch/agent.log"
else
    # transformers + FastAPI backend
    if [[ "$GPU_ID" == *","* ]]; then
        DEVICE_ARG="auto"
    else
        DEVICE_ARG="cuda:0"
    fi

    screen -dmS "$SESSION_NAME" bash -c "source \$HOME/miniforge3/etc/profile.d/conda.sh 2>/dev/null || eval \"\$(conda shell.bash hook)\"; conda activate autoresearch; [[ -f \$CONDA_PREFIX/etc/conda/activate.d/cuda-libs.sh ]] && source \$CONDA_PREFIX/etc/conda/activate.d/cuda-libs.sh; CUDA_VISIBLE_DEVICES=$GPU_ID python -m autoresearch.server.model_server --model '$MODEL' --port $PORT --device $DEVICE_ARG --dtype $DTYPE 2>&1 | tee ~/autoresearch/agent.log"
fi

echo "Agent starting in screen '$SESSION_NAME' on GPU $GPU_ID"
echo "Model: $MODEL (backend: $BACKEND)"
echo "API: http://localhost:$PORT"

# If --no-wait, exit immediately
if [[ "$NO_WAIT" == "--no-wait" ]]; then
    echo "Started (not waiting for health check)"
    exit 0
fi

# Wait for ready (up to 10 min)
echo "Waiting for model to load..."
for i in $(seq 1 120); do
    HEALTH="$(curl -s http://localhost:$PORT/health 2>/dev/null || curl -s http://localhost:$PORT/v1/models 2>/dev/null || echo "")"
    if [[ -n "$HEALTH" ]] && [[ "$HEALTH" != *"error"* ]]; then
        echo "Agent is ready!"
        exit 0
    fi
    sleep 5
done
echo "WARNING: Agent not ready within 10 min. Check: screen -r $SESSION_NAME"
exit 0
