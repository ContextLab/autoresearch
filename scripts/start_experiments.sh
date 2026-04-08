#!/usr/bin/env bash
set -euo pipefail
GPU_IDS="${1:?Usage: start_experiments.sh <gpu_ids>}"
INSTALL_DIR="$HOME/autoresearch"

# Activate conda env — always prefer miniforge3 (installed by setup_env.sh)
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

IFS=',' read -ra GPUS <<< "$GPU_IDS"
for GPU_ID in "${GPUS[@]}"; do
    SESSION_NAME="autoresearch-gpu${GPU_ID}"
    if screen -list | grep -q "$SESSION_NAME"; then
        echo "Session '$SESSION_NAME' already running — skipping"
        continue
    fi
    mkdir -p "$INSTALL_DIR/checkpoints/gpu${GPU_ID}"
    screen -dmS "$SESSION_NAME" bash -c "source \$HOME/miniforge3/etc/profile.d/conda.sh 2>/dev/null || eval \"\$(conda shell.bash hook)\"; conda activate autoresearch; [[ -f \$CONDA_PREFIX/etc/conda/activate.d/cuda-libs.sh ]] && source \$CONDA_PREFIX/etc/conda/activate.d/cuda-libs.sh; cd $INSTALL_DIR; CUDA_VISIBLE_DEVICES=$GPU_ID python -m autoresearch.server.runner --gpu-id $GPU_ID --checkpoint-dir $INSTALL_DIR/checkpoints/gpu${GPU_ID} 2>&1 | tee $INSTALL_DIR/checkpoints/gpu${GPU_ID}/runner.log"
    echo "Started: $SESSION_NAME"
done
