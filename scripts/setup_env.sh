#!/usr/bin/env bash
# Idempotent server environment setup for autoresearch.
# Designed to run ON the remote server via SSH.
# Handles: fresh installs, partial installs, corrupted envs, stale processes.
# Uses: Miniforge (provides mamba+conda), uv (fast pip replacement)
set -euo pipefail

ENV_NAME="autoresearch"
PYTHON_VERSION="3.12"
UV_TMPDIR="$HOME/.uv-tmp"
CLEAN=false

# Parse flags
for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN=true ;;
    esac
done

echo "=== autoresearch environment setup ==="

# ── 0. Lockfile — prevent concurrent runs ────────────────────────────────
LOCKFILE="$HOME/.autoresearch-setup.lock"
if [[ -f "$LOCKFILE" ]]; then
    LOCK_PID="$(cat "$LOCKFILE" 2>/dev/null || true)"
    if [[ -n "$LOCK_PID" ]] && kill -0 "$LOCK_PID" 2>/dev/null; then
        echo "Setup already running (PID $LOCK_PID). Waiting for it to finish..."
        for i in $(seq 1 360); do
            sleep 10
            if ! kill -0 "$LOCK_PID" 2>/dev/null; then
                echo "Previous run finished."
                break
            fi
        done
        if kill -0 "$LOCK_PID" 2>/dev/null; then
            echo "Previous run still going after 1 hour. Killing it."
            kill "$LOCK_PID" 2>/dev/null || true
            sleep 2
        fi
    fi
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

# Kill stale install processes from previous crashed runs
STALE_PIDS="$(pgrep -u "$(whoami)" -f 'pip.*install\|uv.*install' 2>/dev/null || true)"
if [[ -n "$STALE_PIDS" ]]; then
    echo "Killing stale install processes: $STALE_PIDS"
    kill $STALE_PIDS 2>/dev/null || true
    sleep 2
fi

# ── 1. Detect OS and architecture ───────────────────────────────────────
OS_RAW="$(uname -s)"
ARCH_RAW="$(uname -m)"

case "$OS_RAW" in
    Linux*)
        if grep -qi microsoft /proc/version 2>/dev/null; then
            echo "Detected OS: WSL (Linux)"
        else
            echo "Detected OS: Linux"
        fi
        MINIFORGE_OS="Linux"
        ;;
    Darwin*)
        echo "Detected OS: macOS"
        MINIFORGE_OS="MacOSX"
        ;;
    *)
        echo "Error: Unsupported OS: $OS_RAW" >&2
        exit 1
        ;;
esac

case "$ARCH_RAW" in
    x86_64)  MINIFORGE_ARCH="x86_64" ;;
    aarch64) MINIFORGE_ARCH="aarch64" ;;
    arm64)
        if [[ "$MINIFORGE_OS" == "Linux" ]]; then
            MINIFORGE_ARCH="aarch64"
        else
            MINIFORGE_ARCH="arm64"
        fi
        ;;
    *)
        echo "Error: Unsupported architecture: $ARCH_RAW" >&2
        exit 1
        ;;
esac

echo "Architecture: $ARCH_RAW"

# ── 2. Install Miniforge (provides conda + mamba) if not present ────────
# Try sourcing from common locations first
for CONDA_PREFIX in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    if [[ -f "${CONDA_PREFIX}/etc/profile.d/conda.sh" ]]; then
        source "${CONDA_PREFIX}/etc/profile.d/conda.sh"
        break
    fi
done

if ! type conda &>/dev/null; then
    echo "Installing Miniforge (conda + mamba)..."
    INSTALLER="Miniforge3-${MINIFORGE_OS}-${MINIFORGE_ARCH}.sh"
    URL="https://github.com/conda-forge/miniforge/releases/latest/download/${INSTALLER}"
    curl -fsSL "$URL" -o "/tmp/${INSTALLER}"
    bash "/tmp/${INSTALLER}" -b -p "$HOME/miniforge3"
    rm -f "/tmp/${INSTALLER}"
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    echo "Miniforge installed"
else
    echo "conda already available — skipping Miniforge install"
fi

# Ensure mamba is available
# Miniforge includes mamba; system conda installs may not
if ! type mamba &>/dev/null; then
    # Try installing mamba via conda (may fail on system installs without base env)
    if conda install -y -n base -c conda-forge mamba 2>/dev/null; then
        echo "mamba installed into base"
    else
        # Fall back to using conda directly (slower but works)
        echo "Cannot install mamba (system conda without base env) — using conda instead"
        mamba() { conda "$@"; }
    fi
fi

if ! type conda &>/dev/null; then
    echo "Error: conda not available after installation" >&2
    exit 1
fi

# ── 3. Create or repair conda env (using mamba for speed) ───────────────
eval "$(conda shell.bash hook)"

ENV_OK=false
ENV_LIST="$(conda env list 2>/dev/null || true)"

# --clean flag: force remove and recreate
if [[ "$CLEAN" == true ]] && echo "$ENV_LIST" | grep -q "$ENV_NAME"; then
    echo "Clean mode: removing existing env '$ENV_NAME'..."
    conda remove -n "$ENV_NAME" --all -y 2>/dev/null || rm -rf "$HOME/.conda/envs/$ENV_NAME"
    ENV_LIST="$(conda env list 2>/dev/null || true)"
fi

if echo "$ENV_LIST" | grep -q "$ENV_NAME"; then
    if conda run -n "$ENV_NAME" python --version &>/dev/null; then
        ENV_OK=true
        echo "conda env '$ENV_NAME' already exists and is functional — skipping"
    else
        echo "conda env '$ENV_NAME' is corrupted — removing and recreating"
        conda remove -n "$ENV_NAME" --all -y 2>/dev/null || rm -rf "$HOME/.conda/envs/$ENV_NAME"
    fi
fi

if [[ "$ENV_OK" == false ]]; then
    echo "Creating conda env '$ENV_NAME' with Python $PYTHON_VERSION (via mamba)..."
    mamba create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
    echo "conda env '$ENV_NAME' created"
fi

# ── 4. Activate the env ─────────────────────────────────────────────────
conda activate "$ENV_NAME"
echo "Activated env: $ENV_NAME (Python $(python --version 2>&1 | awk '{print $2}'))"

# ── 5. Install uv (fast pip replacement) ────────────────────────────────
if command -v uv &>/dev/null; then
    echo "uv already installed — skipping"
else
    echo "Installing uv..."
    pip install --quiet uv
    echo "uv installed"
fi

# Use home directory for temp/build files (server /tmp is often small)
export TMPDIR="$UV_TMPDIR"
mkdir -p "$TMPDIR"

# ── 5b. Install system tools via conda ─────────────────────────────────
# ffmpeg/ffprobe needed by multitrans benchmark (audio processing)
if command -v ffprobe &>/dev/null; then
    echo "ffmpeg already installed — skipping"
else
    echo "Installing ffmpeg via conda..."
    conda install -y -n "$ENV_NAME" -c conda-forge ffmpeg
    echo "ffmpeg installed"
fi

# espeak-ng needed by phonemizer (IPA transcription)
if command -v espeak-ng &>/dev/null || command -v espeak &>/dev/null; then
    echo "espeak already installed — skipping"
else
    echo "Installing espeak-ng via conda..."
    conda install -y -n "$ENV_NAME" -c conda-forge espeak-ng
    echo "espeak-ng installed"
fi

# ── 6. Install packages via uv ──────────────────────────────────────────
# Install torch with CUDA 12.8 wheels (compatible with vllm prebuilt wheels)
if python -c "import torch" &>/dev/null; then
    echo "torch already installed — skipping"
else
    echo "Installing torch (cu128) via uv... this may take several minutes"
    uv pip install torch --index-url https://download.pytorch.org/whl/cu128
    echo "torch installed"
fi

# Install numpy (torch needs it at runtime)
if python -c "import numpy" &>/dev/null; then
    echo "numpy already installed — skipping"
else
    uv pip install numpy
fi

# Install remaining packages (pin compatible versions)
# vllm + transformers must be version-compatible
PACKAGES=(
    "openai>=1.0"
    "fastapi>=0.100"
    "uvicorn>=0.20"
    "git+https://github.com/huggingface/transformers.git"
    "accelerate>=0.20"
    "bitsandbytes>=0.41"
    "huggingface-hub>=0.20"
)

# Install llama-cpp-python with CUDA support (use prebuilt wheel)
if python -c "import llama_cpp" &>/dev/null 2>&1; then
    echo "llama-cpp-python already installed — skipping"
else
    echo "Installing llama-cpp-python with CUDA (prebuilt)..."
    uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu128 --no-cache-dir 2>/dev/null \
        || uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --no-cache-dir 2>/dev/null \
        || { echo "Prebuilt wheel not found, building from source..."; \
             export CUDACXX="$(which nvcc 2>/dev/null || find /usr/local/cuda*/bin -name nvcc 2>/dev/null | head -1)"; \
             export PATH="$(dirname "$CUDACXX"):$PATH"; \
             CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --no-cache-dir; }
    echo "llama-cpp-python installed"
fi

# Install packages (uv is fast and handles already-satisfied deps)
echo "Installing/verifying packages via uv..."
uv pip install "${PACKAGES[@]}"
echo "Packages installed"

# Verify transformers has gemma4 support
if ! python -c "from transformers import AutoModelForCausalLM" &>/dev/null 2>&1; then
    echo "Reinstalling transformers from source..."
    uv pip install --force-reinstall "git+https://github.com/huggingface/transformers.git"
    echo "transformers reinstalled"
fi

# Ensure CUDA shared libs are findable (nvidia pip packages put them in site-packages)
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
CUDA_LIB_DIRS=(
    "$SITE_PACKAGES/nvidia/cusparselt/lib"
    "$SITE_PACKAGES/nvidia/cublas/lib"
    "$SITE_PACKAGES/nvidia/cudnn/lib"
    "$SITE_PACKAGES/nvidia/cuda_runtime/lib"
)
EXTRA_LD=""
for d in "${CUDA_LIB_DIRS[@]}"; do
    [[ -d "$d" ]] && EXTRA_LD="${EXTRA_LD:+$EXTRA_LD:}$d"
done
if [[ -n "$EXTRA_LD" ]]; then
    # Write an activation script so LD_LIBRARY_PATH is set on conda activate
    ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
    mkdir -p "$ACTIVATE_DIR"
    cat > "$ACTIVATE_DIR/cuda-libs.sh" <<SCRIPT
#!/bin/bash
export LD_LIBRARY_PATH="$EXTRA_LD\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
SCRIPT
    chmod +x "$ACTIVATE_DIR/cuda-libs.sh"
    # Also set for current session
    export LD_LIBRARY_PATH="$EXTRA_LD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "CUDA library paths configured"
fi

# ── 7. Verify installation ──────────────────────────────────────────────
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'  torch {torch.__version__}')"
python -c "import openai; print(f'  openai {openai.__version__}')"
python -c "import transformers; print(f'  transformers {transformers.__version__}')"
python -c "import fastapi; print(f'  fastapi {fastapi.__version__}')"
python -c "from transformers import AutoModelForCausalLM; print('  transformers: OK')" || { echo "  transformers: BROKEN — setup failed"; exit 1; }

# ── 8. Create directories ──────────────────────────────────────────────
mkdir -p "$HOME/autoresearch/checkpoints"

echo ""
echo "=== Setup complete ==="
