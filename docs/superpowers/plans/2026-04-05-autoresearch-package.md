# Autoresearch Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform autoresearch into a deployable Python package that runs autonomous AI research experiments on any SSH GPU server, with a local HF model as the experiment agent and Claude Code as the research architect.

**Architecture:** Python package with CLI (`autoresearch deploy/status/sync/stop`), deterministic shell scripts for repeatable server operations, and a Claude Code skill for interactive setup. Server-side: vLLM serves agent model, experiment runners loop autonomously in screen sessions.

**Tech Stack:** Python 3.11, Click (CLI), PyYAML (config), vLLM (agent serving), paramiko or sshpass (SSH), screen (remote sessions)

**Spec:** `docs/superpowers/specs/2026-04-05-autoresearch-package-design.md`

---

## File Map

### New files (create)

| File | Responsibility |
|-|-|
| `src/autoresearch/__init__.py` | Package init, version |
| `src/autoresearch/cli.py` | Click CLI: deploy, status, sync, stop, resume |
| `src/autoresearch/config.py` | Load/save credentials.yaml + project config |
| `src/autoresearch/ssh.py` | SSH command execution, file transfer, screen management |
| `src/autoresearch/deploy.py` | Orchestrates full deployment to server |
| `src/autoresearch/server/__init__.py` | Server subpackage |
| `src/autoresearch/server/agent.py` | Agent harness: construct prompts, parse responses, apply diffs |
| `src/autoresearch/server/runner.py` | Experiment loop: train -> evaluate -> keep/discard |
| `src/autoresearch/server/vllm_manager.py` | vLLM start/stop/health check |
| `src/autoresearch/server/monitor.py` | Read checkpoints, format status output |
| `src/autoresearch/program/__init__.py` | Program subpackage |
| `src/autoresearch/program/builder.py` | Construct program.md from various input paths |
| `src/autoresearch/program/templates/default.md` | Default program.md template |
| `scripts/ssh.sh` | SSH wrapper using sshpass + credentials.yaml |
| `scripts/setup_env.sh` | Idempotent server env setup (conda, deps, data) |
| `scripts/start_agent.sh` | Start vLLM in screen session |
| `scripts/start_experiments.sh` | Start experiment runners in screen sessions |
| `scripts/status.sh` | Read checkpoint state, report status |
| `scripts/sync.sh` | rsync checkpoints + results locally |
| `scripts/stop.sh` | Kill screen sessions, stop vLLM |
| `skill/SKILL.md` | Claude Code skill definition |
| `tests/test_config.py` | Config loading/saving tests |
| `tests/test_ssh.py` | SSH connection tests (real server) |
| `tests/test_server_setup.py` | Server setup script tests (real server) |
| `tests/test_agent.py` | Agent harness tests |
| `tests/test_runner.py` | Experiment runner tests |
| `tests/test_program_builder.py` | Program.md builder tests |
| `tests/test_integration.py` | End-to-end integration tests |

### Modified files

| File | Change |
|-|-|
| `pyproject.toml` | Add deps (click, pyyaml, openai, paramiko), entry points, src layout |
| `.gitignore` | Add results/, checkpoints/, .autoresearch/ (already has credentials.yaml) |

---

## Task 1: Package Scaffolding + Config Module

**Files:**
- Modify: `pyproject.toml`
- Create: `src/autoresearch/__init__.py`
- Create: `src/autoresearch/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test for config loading**

```python
# tests/test_config.py
import os
import tempfile
import pytest
from pathlib import Path


def test_load_credentials_from_yaml():
    """Load credentials from a real YAML file."""
    from autoresearch.config import load_credentials

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(
            "servers:\n"
            "  default:\n"
            "    host: example.com\n"
            "    port: 22\n"
            "    username: testuser\n"
            "    password: testpass\n"
        )
        f.flush()
        creds = load_credentials(f.name)

    os.unlink(f.name)
    assert creds["host"] == "example.com"
    assert creds["port"] == 22
    assert creds["username"] == "testuser"
    assert creds["password"] == "testpass"


def test_load_credentials_missing_file():
    """Raise FileNotFoundError for missing credentials."""
    from autoresearch.config import load_credentials

    with pytest.raises(FileNotFoundError):
        load_credentials("/nonexistent/path.yaml")


def test_load_credentials_named_server():
    """Load a named server profile."""
    from autoresearch.config import load_credentials

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(
            "servers:\n"
            "  default:\n"
            "    host: default.com\n"
            "    port: 22\n"
            "    username: user1\n"
            "    password: pass1\n"
            "  gpu-box:\n"
            "    host: gpu.example.com\n"
            "    port: 2222\n"
            "    username: user2\n"
            "    password: pass2\n"
        )
        f.flush()
        creds = load_credentials(f.name, server="gpu-box")

    os.unlink(f.name)
    assert creds["host"] == "gpu.example.com"
    assert creds["port"] == 2222


def test_save_credentials():
    """Save credentials to YAML and read back."""
    from autoresearch.config import save_credentials, load_credentials

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "creds.yaml")
        save_credentials(
            path,
            server="default",
            host="myhost.com",
            port=22,
            username="myuser",
            password="mypass",
        )
        creds = load_credentials(path)
        assert creds["host"] == "myhost.com"
        assert creds["username"] == "myuser"


def test_load_project_config():
    """Load project config with GPU assignments and model."""
    from autoresearch.config import load_project_config, save_project_config

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.yaml")
        save_project_config(
            path,
            agent_model="google/gemma-4-27b-it",
            agent_gpu=0,
            experiment_gpus=[1, 2, 3],
        )
        config = load_project_config(path)
        assert config["agent_model"] == "google/gemma-4-27b-it"
        assert config["agent_gpu"] == 0
        assert config["experiment_gpus"] == [1, 2, 3]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /path/to/autoresearch
uv run pytest tests/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'autoresearch'`

- [ ] **Step 3: Update pyproject.toml for src layout**

```toml
[project]
name = "autoresearch"
version = "0.2.0"
description = "Autonomous AI research on remote GPU servers"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "click>=8.1",
    "pyyaml>=6.0",
    "openai>=1.0",
    "kernels>=0.11.7",
    "matplotlib>=3.10.8",
    "numpy>=2.2.6",
    "pandas>=2.3.3",
    "pyarrow>=21.0.0",
    "requests>=2.32.0",
    "rustbpe>=0.1.0",
    "tiktoken>=0.11.0",
    "torch==2.9.1",
]

[project.scripts]
autoresearch = "autoresearch.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.uv.sources]
torch = [
    { index = "pytorch-cu128" },
]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Create package __init__.py**

```python
# src/autoresearch/__init__.py
"""Autoresearch: Autonomous AI research on remote GPU servers."""

__version__ = "0.2.0"
```

- [ ] **Step 5: Implement config module**

```python
# src/autoresearch/config.py
"""Load and save credentials and project configuration."""

from pathlib import Path
from typing import Any

import yaml


def load_credentials(path: str | Path, server: str = "default") -> dict[str, Any]:
    """Load SSH credentials for a named server from a YAML file.

    Returns dict with keys: host, port, username, password.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Credentials file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    servers = data.get("servers", {})
    if server not in servers:
        raise KeyError(f"Server '{server}' not found in {path}. Available: {list(servers.keys())}")

    entry = servers[server]
    return {
        "host": entry["host"],
        "port": entry.get("port", 22),
        "username": entry["username"],
        "password": entry.get("password", ""),
    }


def save_credentials(
    path: str | Path,
    server: str,
    host: str,
    port: int,
    username: str,
    password: str,
) -> None:
    """Save SSH credentials for a named server to a YAML file.

    Merges with existing servers if file already exists.
    """
    path = Path(path)
    data: dict = {"servers": {}}
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {"servers": {}}

    data["servers"][server] = {
        "host": host,
        "port": port,
        "username": username,
        "password": password,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def save_project_config(
    path: str | Path,
    agent_model: str,
    agent_gpu: int,
    experiment_gpus: list[int],
    **extra: Any,
) -> None:
    """Save project configuration (GPU assignments, model, etc.)."""
    path = Path(path)
    config = {
        "agent_model": agent_model,
        "agent_gpu": agent_gpu,
        "experiment_gpus": experiment_gpus,
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def load_project_config(path: str | Path) -> dict[str, Any]:
    """Load project configuration from YAML."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Project config not found: {path}")

    with open(path) as f:
        return yaml.safe_load(f)
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
uv run pytest tests/test_config.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/autoresearch/__init__.py src/autoresearch/config.py tests/test_config.py pyproject.toml
git commit -m "feat: add package scaffolding and config module"
```

---

## Task 2: SSH Module

**Files:**
- Create: `src/autoresearch/ssh.py`
- Create: `tests/test_ssh.py`

- [ ] **Step 1: Write failing tests for SSH module**

```python
# tests/test_ssh.py
"""SSH module tests — runs against real server from credentials.yaml."""

import os
import pytest
from pathlib import Path

CREDS_PATH = Path(__file__).parent.parent / "credentials.yaml"
SKIP_NO_SERVER = pytest.mark.skipif(
    not CREDS_PATH.exists(),
    reason="No credentials.yaml — skipping real server tests",
)


@SKIP_NO_SERVER
def test_ssh_run_command():
    """Run a simple command on the real server."""
    from autoresearch.config import load_credentials
    from autoresearch.ssh import SSHClient

    creds = load_credentials(CREDS_PATH)
    client = SSHClient(**creds)
    result = client.run("echo hello")
    assert "hello" in result.stdout
    assert result.returncode == 0


@SKIP_NO_SERVER
def test_ssh_run_command_failure():
    """Run a command that fails and check return code."""
    from autoresearch.config import load_credentials
    from autoresearch.ssh import SSHClient

    creds = load_credentials(CREDS_PATH)
    client = SSHClient(**creds)
    result = client.run("exit 42")
    assert result.returncode == 42


@SKIP_NO_SERVER
def test_ssh_detect_os():
    """Detect remote OS."""
    from autoresearch.config import load_credentials
    from autoresearch.ssh import SSHClient

    creds = load_credentials(CREDS_PATH)
    client = SSHClient(**creds)
    os_info = client.run("uname -s")
    assert os_info.stdout.strip() in ("Linux", "Darwin")


@SKIP_NO_SERVER
def test_ssh_detect_gpus():
    """Detect GPUs on remote server."""
    from autoresearch.config import load_credentials
    from autoresearch.ssh import SSHClient

    creds = load_credentials(CREDS_PATH)
    client = SSHClient(**creds)
    result = client.run(
        "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader"
    )
    assert result.returncode == 0
    lines = result.stdout.strip().split("\n")
    assert len(lines) >= 1  # at least one GPU


@SKIP_NO_SERVER
def test_ssh_file_transfer(tmp_path):
    """Upload a file and verify it exists remotely."""
    from autoresearch.config import load_credentials
    from autoresearch.ssh import SSHClient

    creds = load_credentials(CREDS_PATH)
    client = SSHClient(**creds)

    local_file = tmp_path / "test_upload.txt"
    local_file.write_text("autoresearch test")
    remote_path = "/tmp/autoresearch_test_upload.txt"

    client.upload(str(local_file), remote_path)
    result = client.run(f"cat {remote_path}")
    assert "autoresearch test" in result.stdout

    # Cleanup
    client.run(f"rm -f {remote_path}")


@SKIP_NO_SERVER
def test_ssh_screen_management():
    """Start and stop a screen session."""
    from autoresearch.config import load_credentials
    from autoresearch.ssh import SSHClient

    creds = load_credentials(CREDS_PATH)
    client = SSHClient(**creds)
    session_name = "autoresearch_test_session"

    # Start a screen session
    client.start_screen(session_name, "sleep 300")
    result = client.run(f"screen -list | grep {session_name}")
    assert session_name in result.stdout

    # Stop it
    client.stop_screen(session_name)
    result = client.run(f"screen -list | grep {session_name}")
    assert session_name not in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_ssh.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'autoresearch.ssh'`

- [ ] **Step 3: Implement SSH module**

```python
# src/autoresearch/ssh.py
"""SSH client for remote server operations via sshpass + subprocess."""

import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommandResult:
    stdout: str
    stderr: str
    returncode: int


class SSHClient:
    """SSH client wrapping sshpass + ssh/scp for remote operations."""

    def __init__(self, host: str, username: str, password: str, port: int = 22):
        self.host = host
        self.username = username
        self.password = password
        self.port = port

        if not shutil.which("sshpass"):
            raise RuntimeError(
                "sshpass not found. Install it: brew install sshpass (macOS) "
                "or apt install sshpass (Linux)"
            )

    def _ssh_base(self) -> list[str]:
        return [
            "sshpass", "-p", self.password,
            "ssh",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10",
            "-p", str(self.port),
            f"{self.username}@{self.host}",
        ]

    def run(self, command: str, timeout: int = 120) -> CommandResult:
        """Run a command on the remote server."""
        proc = subprocess.run(
            [*self._ssh_base(), command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return CommandResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )

    def upload(self, local_path: str, remote_path: str) -> None:
        """Upload a file or directory to the remote server via scp."""
        is_dir = Path(local_path).is_dir()
        cmd = [
            "sshpass", "-p", self.password,
            "scp",
            "-o", "StrictHostKeyChecking=accept-new",
            "-P", str(self.port),
        ]
        if is_dir:
            cmd.append("-r")
        cmd.extend([local_path, f"{self.username}@{self.host}:{remote_path}"])
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)

    def download(self, remote_path: str, local_path: str) -> None:
        """Download a file or directory from the remote server via scp."""
        cmd = [
            "sshpass", "-p", self.password,
            "scp",
            "-o", "StrictHostKeyChecking=accept-new",
            "-P", str(self.port),
            "-r",
            f"{self.username}@{self.host}:{remote_path}",
            local_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)

    def rsync(self, local_path: str, remote_path: str, direction: str = "up") -> None:
        """Rsync files between local and remote.

        direction: "up" (local->remote) or "down" (remote->local)
        """
        ssh_cmd = (
            f"sshpass -p {self.password} ssh "
            f"-o StrictHostKeyChecking=accept-new -p {self.port}"
        )
        if direction == "up":
            src = local_path
            dst = f"{self.username}@{self.host}:{remote_path}"
        else:
            src = f"{self.username}@{self.host}:{remote_path}"
            dst = local_path

        subprocess.run(
            ["rsync", "-avz", "--progress", "-e", ssh_cmd, src, dst],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )

    def start_screen(self, session_name: str, command: str) -> None:
        """Start a detached screen session running a command."""
        self.run(f"screen -dmS {session_name} bash -c '{command}'")

    def stop_screen(self, session_name: str) -> None:
        """Kill a screen session by name."""
        self.run(f"screen -S {session_name} -X quit")

    def list_screens(self) -> list[str]:
        """List active screen session names."""
        result = self.run("screen -list")
        lines = result.stdout.strip().split("\n")
        sessions = []
        for line in lines:
            line = line.strip()
            if "." in line and ("Detached" in line or "Attached" in line):
                # Format: "12345.session_name\t(Detached)"
                parts = line.split("\t")[0].split(".")
                if len(parts) >= 2:
                    sessions.append(".".join(parts[1:]))
        return sessions

    def detect_gpus(self) -> list[dict]:
        """Detect NVIDIA GPUs on the remote server.

        Returns list of dicts with keys: index, name, memory_mb.
        """
        result = self.run(
            "nvidia-smi --query-gpu=index,name,memory.total "
            "--format=csv,noheader,nounits"
        )
        if result.returncode != 0:
            return []

        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_mb": int(float(parts[2])),
                })
        return gpus

    def detect_os(self) -> str:
        """Detect remote OS. Returns 'linux', 'macos', or 'windows'."""
        result = self.run("uname -s")
        uname = result.stdout.strip().lower()
        if "linux" in uname:
            return "linux"
        elif "darwin" in uname:
            return "macos"
        else:
            # Check for WSL
            result2 = self.run("cat /proc/version 2>/dev/null")
            if "microsoft" in result2.stdout.lower():
                return "windows"
            return "linux"  # fallback
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_ssh.py -v
```

Expected: All 6 tests PASS (or skipped if no credentials.yaml).

- [ ] **Step 5: Commit**

```bash
git add src/autoresearch/ssh.py tests/test_ssh.py
git commit -m "feat: add SSH client module with real server tests"
```

---

## Task 3: Deterministic Shell Scripts — SSH Wrapper + Server Setup

**Files:**
- Create: `scripts/ssh.sh`
- Create: `scripts/setup_env.sh`
- Create: `tests/test_server_setup.py`

- [ ] **Step 1: Create scripts/ssh.sh**

```bash
#!/usr/bin/env bash
# scripts/ssh.sh — SSH wrapper that reads credentials.yaml
# Usage: ./scripts/ssh.sh [command...]
# If no command given, opens interactive SSH session.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CREDS_FILE="${PROJECT_DIR}/credentials.yaml"

if [ ! -f "$CREDS_FILE" ]; then
    echo "ERROR: credentials.yaml not found at $CREDS_FILE" >&2
    exit 1
fi

# Parse YAML (simple — works for our flat structure)
HOST=$(grep -A5 'default:' "$CREDS_FILE" | grep 'host:' | head -1 | awk '{print $2}')
PORT=$(grep -A5 'default:' "$CREDS_FILE" | grep 'port:' | head -1 | awk '{print $2}')
USERNAME=$(grep -A5 'default:' "$CREDS_FILE" | grep 'username:' | head -1 | awk '{print $2}')
PASSWORD=$(grep -A5 'default:' "$CREDS_FILE" | grep 'password:' | head -1 | awk '{print $2}')

PORT="${PORT:-22}"

if [ $# -eq 0 ]; then
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=accept-new -p "$PORT" "${USERNAME}@${HOST}"
else
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=accept-new -p "$PORT" "${USERNAME}@${HOST}" "$@"
fi
```

- [ ] **Step 2: Create scripts/setup_env.sh**

```bash
#!/usr/bin/env bash
# scripts/setup_env.sh — Idempotent server environment setup
# Detects OS, installs conda if needed, creates env, installs deps.
# Usage: Run on the remote server (or via: ./scripts/ssh.sh 'bash -s' < scripts/setup_env.sh)

set -euo pipefail

ENV_NAME="autoresearch"
PYTHON_VERSION="3.11"
INSTALL_DIR="$HOME/autoresearch"

echo "=== Autoresearch Server Setup ==="

# --- Step 1: Detect OS and architecture ---
OS="$(uname -s)"
ARCH="$(uname -m)"
echo "Detected OS: $OS, Arch: $ARCH"

case "$OS" in
    Linux)
        if grep -qi microsoft /proc/version 2>/dev/null; then
            PLATFORM="wsl"
        else
            PLATFORM="linux"
        fi
        ;;
    Darwin) PLATFORM="macos" ;;
    *)      echo "Unsupported OS: $OS" >&2; exit 1 ;;
esac

# --- Step 2: Install conda (Miniforge) if not present ---
if command -v conda &>/dev/null; then
    echo "conda already installed: $(conda --version)"
else
    echo "Installing Miniforge..."
    case "${PLATFORM}-${ARCH}" in
        linux-x86_64|wsl-x86_64)  INSTALLER="Miniforge3-Linux-x86_64.sh" ;;
        linux-aarch64)             INSTALLER="Miniforge3-Linux-aarch64.sh" ;;
        macos-x86_64)              INSTALLER="Miniforge3-MacOSX-x86_64.sh" ;;
        macos-arm64)               INSTALLER="Miniforge3-MacOSX-arm64.sh" ;;
        *)                         echo "Unsupported platform: ${PLATFORM}-${ARCH}" >&2; exit 1 ;;
    esac

    MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/${INSTALLER}"
    curl -fsSL "$MINIFORGE_URL" -o /tmp/miniforge_installer.sh
    bash /tmp/miniforge_installer.sh -b -p "$HOME/miniforge3"
    rm /tmp/miniforge_installer.sh

    # Initialize conda for current shell
    eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
    conda init bash
    echo "Miniforge installed at $HOME/miniforge3"
fi

# Ensure conda is in PATH for this session
if ! command -v conda &>/dev/null; then
    eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
fi

# --- Step 3: Create conda environment if not exists ---
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda env '$ENV_NAME' already exists"
else
    echo "Creating conda env '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# Activate the environment
conda activate "$ENV_NAME"
echo "Active Python: $(python --version) at $(which python)"

# --- Step 4: Install uv if not present ---
if command -v uv &>/dev/null; then
    echo "uv already installed: $(uv --version)"
else
    echo "Installing uv..."
    pip install uv
fi

# --- Step 5: Install vLLM and core deps ---
echo "Installing/updating core dependencies..."
pip install --upgrade vllm openai torch transformers

# --- Step 6: Create project directory ---
mkdir -p "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR/checkpoints"

echo ""
echo "=== Setup complete ==="
echo "Conda env: $ENV_NAME"
echo "Python: $(python --version)"
echo "Project dir: $INSTALL_DIR"
echo "To activate: conda activate $ENV_NAME"
```

- [ ] **Step 3: Make scripts executable**

```bash
chmod +x scripts/ssh.sh scripts/setup_env.sh
```

- [ ] **Step 4: Write test for server setup**

```python
# tests/test_server_setup.py
"""Test server setup script on real server."""

import pytest
from pathlib import Path

CREDS_PATH = Path(__file__).parent.parent / "credentials.yaml"
SKIP_NO_SERVER = pytest.mark.skipif(
    not CREDS_PATH.exists(),
    reason="No credentials.yaml — skipping real server tests",
)


@SKIP_NO_SERVER
def test_ssh_script_runs(tmp_path):
    """Verify scripts/ssh.sh can connect and run a command."""
    import subprocess

    result = subprocess.run(
        ["bash", "scripts/ssh.sh", "echo CONNECTION_OK"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "CONNECTION_OK" in result.stdout


@SKIP_NO_SERVER
def test_setup_env_is_idempotent():
    """Run setup_env.sh twice — second run should be fast and succeed."""
    from autoresearch.config import load_credentials
    from autoresearch.ssh import SSHClient

    creds = load_credentials(CREDS_PATH)
    client = SSHClient(**creds)

    # Upload and run setup script
    client.upload("scripts/setup_env.sh", "/tmp/setup_env.sh")
    result = client.run("bash /tmp/setup_env.sh", timeout=600)
    assert result.returncode == 0
    assert "Setup complete" in result.stdout

    # Run again — should succeed quickly
    result2 = client.run("bash /tmp/setup_env.sh", timeout=300)
    assert result2.returncode == 0
    assert "already" in result2.stdout.lower()
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_server_setup.py -v --timeout=600
```

Expected: PASS (may take several minutes on first run for conda/pip installs).

- [ ] **Step 6: Commit**

```bash
git add scripts/ssh.sh scripts/setup_env.sh tests/test_server_setup.py
git commit -m "feat: add SSH wrapper and idempotent server setup scripts"
```

---

## Task 4: vLLM Management

**Files:**
- Create: `src/autoresearch/server/__init__.py`
- Create: `src/autoresearch/server/vllm_manager.py`
- Create: `scripts/start_agent.sh`
- Create: `scripts/stop.sh`

- [ ] **Step 1: Create server subpackage init**

```python
# src/autoresearch/server/__init__.py
"""Server-side components for autoresearch."""
```

- [ ] **Step 2: Create scripts/start_agent.sh**

```bash
#!/usr/bin/env bash
# scripts/start_agent.sh — Start vLLM agent in a screen session
# Usage: start_agent.sh <model_name> <gpu_id>
# Example: start_agent.sh google/gemma-4-27b-it 0

set -euo pipefail

MODEL="${1:?Usage: start_agent.sh <model_name> <gpu_id>}"
GPU_ID="${2:-0}"
SESSION_NAME="autoresearch-agent"
PORT=8000

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate autoresearch

# Check if already running
if screen -list | grep -q "$SESSION_NAME"; then
    echo "Agent session '$SESSION_NAME' is already running."
    echo "Stop it first with: screen -S $SESSION_NAME -X quit"
    exit 1
fi

echo "Starting vLLM agent on GPU $GPU_ID with model $MODEL..."

screen -dmS "$SESSION_NAME" bash -c "
    eval \"\$(conda shell.bash hook)\"
    conda activate autoresearch
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m vllm.entrypoints.openai.api_server \
        --model '$MODEL' \
        --port $PORT \
        --gpu-memory-utilization 0.90 \
        --max-model-len 8192 \
        --trust-remote-code \
        2>&1 | tee ~/autoresearch/agent.log
"

echo "vLLM agent starting in screen session '$SESSION_NAME'"
echo "  Model: $MODEL"
echo "  GPU: $GPU_ID"
echo "  API: http://localhost:$PORT"
echo "  Log: ~/autoresearch/agent.log"
echo ""
echo "Check status: screen -r $SESSION_NAME"

# Wait for server to be ready (up to 5 minutes)
echo "Waiting for vLLM to be ready..."
for i in $(seq 1 60); do
    if curl -s http://localhost:$PORT/health 2>/dev/null | grep -q "ok\|healthy\|200"; then
        echo "vLLM is ready!"
        exit 0
    fi
    sleep 5
done

echo "WARNING: vLLM did not become ready within 5 minutes."
echo "Check logs: screen -r $SESSION_NAME"
exit 1
```

- [ ] **Step 3: Create scripts/stop.sh**

```bash
#!/usr/bin/env bash
# scripts/stop.sh — Stop autoresearch sessions
# Usage: stop.sh [all|agent|experiments|gpu<N>]

set -euo pipefail

TARGET="${1:-all}"

stop_screen() {
    local name="$1"
    if screen -list | grep -q "$name"; then
        screen -S "$name" -X quit
        echo "Stopped: $name"
    else
        echo "Not running: $name"
    fi
}

case "$TARGET" in
    all)
        echo "Stopping all autoresearch sessions..."
        stop_screen "autoresearch-agent"
        for session in $(screen -list | grep "autoresearch-gpu" | awk '{print $1}' | cut -d. -f2-); do
            stop_screen "$session"
        done
        echo "All sessions stopped."
        ;;
    agent)
        stop_screen "autoresearch-agent"
        ;;
    experiments)
        for session in $(screen -list | grep "autoresearch-gpu" | awk '{print $1}' | cut -d. -f2-); do
            stop_screen "$session"
        done
        ;;
    gpu*)
        GPU_NUM="${TARGET#gpu}"
        stop_screen "autoresearch-gpu${GPU_NUM}"
        ;;
    *)
        echo "Usage: stop.sh [all|agent|experiments|gpu<N>]" >&2
        exit 1
        ;;
esac
```

- [ ] **Step 4: Implement vllm_manager.py**

```python
# src/autoresearch/server/vllm_manager.py
"""vLLM lifecycle management — start, stop, health check via SSH."""

from autoresearch.ssh import SSHClient


class VLLMManager:
    """Manage vLLM server on a remote GPU."""

    def __init__(self, ssh: SSHClient, model: str, gpu_id: int = 0, port: int = 8000):
        self.ssh = ssh
        self.model = model
        self.gpu_id = gpu_id
        self.port = port

    def start(self) -> str:
        """Start vLLM agent on remote server. Returns status message."""
        # Upload start script
        self.ssh.upload("scripts/start_agent.sh", "~/autoresearch/start_agent.sh")
        result = self.ssh.run(
            f"bash ~/autoresearch/start_agent.sh '{self.model}' {self.gpu_id}",
            timeout=360,
        )
        return result.stdout

    def stop(self) -> str:
        """Stop vLLM agent on remote server."""
        result = self.ssh.run("screen -S autoresearch-agent -X quit")
        return "Agent stopped" if result.returncode == 0 else result.stderr

    def health(self) -> bool:
        """Check if vLLM is responding."""
        result = self.ssh.run(f"curl -s http://localhost:{self.port}/health")
        return result.returncode == 0 and result.stdout.strip() != ""

    def is_running(self) -> bool:
        """Check if the screen session exists."""
        result = self.ssh.run("screen -list | grep autoresearch-agent")
        return "autoresearch-agent" in result.stdout
```

- [ ] **Step 5: Make scripts executable**

```bash
chmod +x scripts/start_agent.sh scripts/stop.sh
```

- [ ] **Step 6: Commit**

```bash
git add src/autoresearch/server/__init__.py src/autoresearch/server/vllm_manager.py \
    scripts/start_agent.sh scripts/stop.sh
git commit -m "feat: add vLLM management and start/stop scripts"
```

---

## Task 5: Agent Harness

**Files:**
- Create: `src/autoresearch/server/agent.py`
- Create: `tests/test_agent.py`

The agent harness constructs prompts for the vLLM model and parses its responses into code edits.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agent.py
"""Tests for the agent harness — prompt construction and response parsing."""

import textwrap


def test_build_experiment_prompt():
    """Build a prompt from program.md, train.py, and results history."""
    from autoresearch.server.agent import build_experiment_prompt

    program_md = "## Goal\nMinimize val_bpb\n## Constraints\nOnly edit train.py"
    train_py = "DEPTH = 8\nASPECT_RATIO = 8"
    results_history = "commit\tval_bpb\tstatus\tdescription\nabc1234\t0.998\tkeep\tbaseline"

    prompt = build_experiment_prompt(program_md, train_py, results_history)

    assert "Minimize val_bpb" in prompt
    assert "DEPTH = 8" in prompt
    assert "baseline" in prompt
    assert "0.998" in prompt


def test_parse_code_edit_fenced():
    """Parse a fenced code block response into a full file replacement."""
    from autoresearch.server.agent import parse_code_edit

    response = textwrap.dedent("""\
        I'll increase the depth to 10.

        ```python
        DEPTH = 10
        ASPECT_RATIO = 8
        ```

        This should improve val_bpb.
    """)

    edit = parse_code_edit(response)
    assert "DEPTH = 10" in edit
    assert "ASPECT_RATIO = 8" in edit


def test_parse_code_edit_diff_format():
    """Parse a diff-style response (search/replace blocks)."""
    from autoresearch.server.agent import parse_code_edit

    response = textwrap.dedent("""\
        Let me change the depth.

        <<<SEARCH
        DEPTH = 8
        ===
        DEPTH = 10
        >>>REPLACE

        This increases model capacity.
    """)

    edit = parse_code_edit(response)
    assert edit is not None
    assert "SEARCH" in response  # confirms we got a diff


def test_apply_diff_to_file():
    """Apply a search/replace diff to file content."""
    from autoresearch.server.agent import apply_diff

    original = "DEPTH = 8\nASPECT_RATIO = 8\nTOTAL_BATCH_SIZE = 2**19"
    diff_text = "<<<SEARCH\nDEPTH = 8\n===\nDEPTH = 10\n>>>REPLACE"

    result = apply_diff(original, diff_text)
    assert "DEPTH = 10" in result
    assert "ASPECT_RATIO = 8" in result  # unchanged


def test_extract_experiment_description():
    """Extract the experiment description from agent response."""
    from autoresearch.server.agent import extract_description

    response = "I'll increase DEPTH from 8 to 10 to add model capacity.\n\n```python\n..."
    desc = extract_description(response)
    assert len(desc) > 0
    assert len(desc) <= 200  # should be concise
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_agent.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement agent harness**

```python
# src/autoresearch/server/agent.py
"""Agent harness: construct prompts, call vLLM, parse code edits."""

import re
from openai import OpenAI


def build_experiment_prompt(
    program_md: str,
    train_py: str,
    results_history: str,
    iteration: int = 0,
) -> str:
    """Build the prompt sent to the agent LLM for the next experiment."""
    return f"""You are an autonomous AI research agent. Your job is to modify train.py to improve val_bpb (lower is better).

## Research Program
{program_md}

## Current train.py
```python
{train_py}
```

## Experiment History
```
{results_history}
```

## Iteration
This is experiment #{iteration + 1}.

## Instructions
1. Analyze the current code and past results.
2. Propose ONE focused change to train.py.
3. Explain your reasoning in 1-2 sentences.
4. Output the COMPLETE modified train.py in a ```python fenced code block.
   OR use <<<SEARCH / === / >>>REPLACE blocks for targeted edits.
5. Keep changes small and focused. One idea per experiment.
"""


def parse_code_edit(response: str) -> str | None:
    """Parse the agent's response to extract code edits.

    Supports two formats:
    1. Fenced code block (```python ... ```) — full file replacement
    2. SEARCH/REPLACE blocks — targeted diffs

    Returns the extracted code/diff string, or None if unparseable.
    """
    # Try fenced code block first
    fenced = re.findall(r"```python\n(.*?)```", response, re.DOTALL)
    if fenced:
        # Return the longest fenced block (likely the full file)
        return max(fenced, key=len).strip()

    # Try SEARCH/REPLACE format
    if "<<<SEARCH" in response and ">>>REPLACE" in response:
        return response  # return full response, apply_diff will extract blocks

    return None


def apply_diff(original: str, diff_text: str) -> str:
    """Apply SEARCH/REPLACE diff blocks to original file content.

    Diff format:
    <<<SEARCH
    old code
    ===
    new code
    >>>REPLACE
    """
    result = original
    pattern = r"<<<SEARCH\n(.*?)\n===\n(.*?)\n>>>REPLACE"
    matches = re.findall(pattern, diff_text, re.DOTALL)

    for search, replace in matches:
        if search.strip() in result:
            result = result.replace(search.strip(), replace.strip(), 1)

    return result


def extract_description(response: str) -> str:
    """Extract a concise experiment description from the agent's response.

    Takes the first sentence/line before any code block, truncated to 200 chars.
    """
    # Get text before the first code block
    parts = re.split(r"```", response, maxsplit=1)
    text = parts[0].strip()

    # Take first meaningful line
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and len(line) > 10:
            return line[:200]

    return text[:200] if text else "no description"


def call_agent(
    prompt: str,
    base_url: str = "http://localhost:8000/v1",
    model: str = "default",
    max_tokens: int = 8192,
    temperature: float = 0.7,
) -> str:
    """Call the vLLM agent and return the response text."""
    client = OpenAI(base_url=base_url, api_key="not-needed")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_agent.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/autoresearch/server/agent.py tests/test_agent.py
git commit -m "feat: add agent harness with prompt construction and response parsing"
```

---

## Task 6: Experiment Runner

**Files:**
- Create: `src/autoresearch/server/runner.py`
- Create: `scripts/start_experiments.sh`
- Create: `tests/test_runner.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_runner.py
"""Tests for the experiment runner logic."""

import json
import os
import tempfile


def test_parse_training_results():
    """Parse val_bpb and peak_vram_mb from a run.log."""
    from autoresearch.server.runner import parse_results

    log_content = """
some output
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
"""
    results = parse_results(log_content)
    assert results["val_bpb"] == 0.9979
    assert results["peak_vram_mb"] == 45060.2
    assert results["status"] == "ok"


def test_parse_training_results_crash():
    """Detect a crash from a run.log without val_bpb."""
    from autoresearch.server.runner import parse_results

    log_content = """
Traceback (most recent call last):
  File "train.py", line 123
RuntimeError: CUDA out of memory
"""
    results = parse_results(log_content)
    assert results["status"] == "crash"
    assert results["val_bpb"] == 0.0


def test_update_results_tsv():
    """Append a result row to results.tsv."""
    from autoresearch.server.runner import update_results_tsv

    with tempfile.TemporaryDirectory() as tmpdir:
        tsv_path = os.path.join(tmpdir, "results.tsv")

        # First write creates header
        update_results_tsv(tsv_path, "abc1234", 0.998, 44.0, "keep", "baseline")
        update_results_tsv(tsv_path, "def5678", 0.993, 44.2, "keep", "increase LR")

        with open(tsv_path) as f:
            lines = f.readlines()

        assert lines[0].startswith("commit\t")
        assert "abc1234" in lines[1]
        assert "def5678" in lines[2]
        assert len(lines) == 3


def test_checkpoint_write_and_read():
    """Write and read a checkpoint state.json."""
    from autoresearch.server.runner import write_checkpoint, read_checkpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state.json")

        write_checkpoint(path, gpu_id=1, iteration=42, best_val_bpb=0.982,
                         current_commit="abc1234", best_commit="def5678",
                         status="running")

        state = read_checkpoint(path)
        assert state["gpu_id"] == 1
        assert state["iteration"] == 42
        assert state["best_val_bpb"] == 0.982
        assert state["status"] == "running"


def test_should_keep_result():
    """Decide whether to keep or discard based on val_bpb."""
    from autoresearch.server.runner import should_keep

    assert should_keep(new_bpb=0.990, best_bpb=0.995) is True   # improved
    assert should_keep(new_bpb=0.995, best_bpb=0.990) is False  # worse
    assert should_keep(new_bpb=0.990, best_bpb=0.990) is False  # equal = discard
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_runner.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement runner module**

```python
# src/autoresearch/server/runner.py
"""Experiment runner: the autonomous train-evaluate-decide loop."""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


def parse_results(log_content: str) -> dict:
    """Parse val_bpb and peak_vram_mb from training log output.

    Returns dict with keys: val_bpb, peak_vram_mb, status.
    """
    val_match = re.search(r"^val_bpb:\s+([0-9.]+)", log_content, re.MULTILINE)
    vram_match = re.search(r"^peak_vram_mb:\s+([0-9.]+)", log_content, re.MULTILINE)

    if val_match:
        return {
            "val_bpb": float(val_match.group(1)),
            "peak_vram_mb": float(vram_match.group(1)) if vram_match else 0.0,
            "status": "ok",
        }
    else:
        return {
            "val_bpb": 0.0,
            "peak_vram_mb": 0.0,
            "status": "crash",
        }


def update_results_tsv(
    tsv_path: str,
    commit: str,
    val_bpb: float,
    memory_gb: float,
    status: str,
    description: str,
) -> None:
    """Append a result row to results.tsv. Creates file with header if needed."""
    path = Path(tsv_path)
    header = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header)

    with open(path, "a") as f:
        f.write(f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n")


def write_checkpoint(
    path: str,
    gpu_id: int,
    iteration: int,
    best_val_bpb: float,
    current_commit: str,
    best_commit: str,
    status: str,
) -> None:
    """Write checkpoint state to JSON."""
    state = {
        "gpu_id": gpu_id,
        "iteration": iteration,
        "best_val_bpb": best_val_bpb,
        "current_commit": current_commit,
        "best_commit": best_commit,
        "status": status,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def read_checkpoint(path: str) -> dict:
    """Read checkpoint state from JSON."""
    with open(path) as f:
        return json.load(f)


def should_keep(new_bpb: float, best_bpb: float) -> bool:
    """Decide whether to keep a result. Lower val_bpb is better."""
    return new_bpb < best_bpb
```

- [ ] **Step 4: Create scripts/start_experiments.sh**

```bash
#!/usr/bin/env bash
# scripts/start_experiments.sh — Start experiment runners in screen sessions
# Usage: start_experiments.sh <gpu_ids_comma_separated>
# Example: start_experiments.sh 1,2,3,4,5,6,7

set -euo pipefail

GPU_IDS="${1:?Usage: start_experiments.sh <gpu_ids_comma_separated>}"
INSTALL_DIR="$HOME/autoresearch"

eval "$(conda shell.bash hook)"
conda activate autoresearch

IFS=',' read -ra GPUS <<< "$GPU_IDS"

for GPU_ID in "${GPUS[@]}"; do
    SESSION_NAME="autoresearch-gpu${GPU_ID}"

    if screen -list | grep -q "$SESSION_NAME"; then
        echo "Session '$SESSION_NAME' already running — skipping"
        continue
    fi

    echo "Starting experiment runner on GPU $GPU_ID..."
    mkdir -p "$INSTALL_DIR/checkpoints/gpu${GPU_ID}"

    screen -dmS "$SESSION_NAME" bash -c "
        eval \"\$(conda shell.bash hook)\"
        conda activate autoresearch
        cd $INSTALL_DIR
        CUDA_VISIBLE_DEVICES=$GPU_ID python -m autoresearch.server.runner \
            --gpu-id $GPU_ID \
            --checkpoint-dir $INSTALL_DIR/checkpoints/gpu${GPU_ID} \
            2>&1 | tee $INSTALL_DIR/checkpoints/gpu${GPU_ID}/runner.log
    "

    echo "Started: $SESSION_NAME"
done

echo ""
echo "Running experiments:"
screen -list | grep "autoresearch-gpu" || echo "  (none)"
```

- [ ] **Step 5: Make script executable and run tests**

```bash
chmod +x scripts/start_experiments.sh
uv run pytest tests/test_runner.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/autoresearch/server/runner.py scripts/start_experiments.sh tests/test_runner.py
git commit -m "feat: add experiment runner and start_experiments script"
```

---

## Task 7: Program.md Builder

**Files:**
- Create: `src/autoresearch/program/__init__.py`
- Create: `src/autoresearch/program/builder.py`
- Create: `src/autoresearch/program/templates/default.md`
- Create: `tests/test_program_builder.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_program_builder.py
"""Tests for program.md builder."""


def test_build_from_description():
    """Build program.md from a text description."""
    from autoresearch.program.builder import build_from_description

    result = build_from_description(
        goal="Minimize val_bpb for a small GPT on FineWeb-Edu",
        constraints=["Only edit train.py", "5-minute time budget"],
        directions=["Try different depths", "Experiment with learning rates"],
    )

    assert "## Goal" in result
    assert "Minimize val_bpb" in result
    assert "## Constraints" in result
    assert "Only edit train.py" in result
    assert "## Research Directions" in result


def test_build_from_existing_file(tmp_path):
    """Load and validate an existing program.md."""
    from autoresearch.program.builder import load_program

    program_file = tmp_path / "program.md"
    program_file.write_text("# Research Program\n\n## Goal\nTest goal\n")

    result = load_program(str(program_file))
    assert "Test goal" in result


def test_default_template_is_valid():
    """The default template has all required sections."""
    from autoresearch.program.builder import get_default_template

    template = get_default_template()
    required_sections = ["## Goal", "## Setup", "## Constraints",
                         "## Experimentation", "## Research Directions",
                         "## Output Format"]
    for section in required_sections:
        assert section in template, f"Missing section: {section}"


def test_build_from_description_uses_template():
    """Built program.md includes all standard sections."""
    from autoresearch.program.builder import build_from_description

    result = build_from_description(
        goal="Test goal",
        constraints=["No new deps"],
        directions=["Try X"],
    )

    assert "## Goal" in result
    assert "## Constraints" in result
    assert "## Research Directions" in result
    assert "## Output Format" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_program_builder.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create default template**

```markdown
<!-- src/autoresearch/program/templates/default.md -->
# Research Program

## Goal
{goal}

## Setup
- Training script: `train.py`
- Evaluation metric: val_bpb (bits per byte, lower is better)
- Time budget: 5 minutes per experiment (wall clock training time)
- Data: FineWeb-Edu dataset (pre-downloaded via prepare.py)

## Constraints
{constraints}

## Experimentation
- Only modify `train.py` — all other files are read-only
- Each experiment: modify code -> commit -> train -> evaluate -> keep/discard
- Keep if val_bpb improves (lower is better); discard if equal or worse
- Log all results to results.tsv
- Simplicity criterion: prefer simpler code at equal val_bpb

## Research Directions
{directions}

## Output Format
After each training run, the script prints:
```
---
val_bpb:          <value>
training_seconds: <value>
peak_vram_mb:     <value>
```

Log to results.tsv (tab-separated):
```
commit	val_bpb	memory_gb	status	description
```
```

- [ ] **Step 4: Implement builder module**

```python
# src/autoresearch/program/__init__.py
"""Program.md construction and management."""
```

```python
# src/autoresearch/program/builder.py
"""Build program.md files from various input sources."""

from pathlib import Path


TEMPLATE_DIR = Path(__file__).parent / "templates"


def get_default_template() -> str:
    """Return the default program.md template."""
    template_path = TEMPLATE_DIR / "default.md"
    return template_path.read_text()


def load_program(path: str) -> str:
    """Load a program.md file from disk."""
    return Path(path).read_text()


def build_from_description(
    goal: str,
    constraints: list[str] | None = None,
    directions: list[str] | None = None,
    setup: str | None = None,
) -> str:
    """Build a program.md from a description.

    Uses the default template, filling in the provided fields.
    """
    template = get_default_template()

    constraints_text = "\n".join(f"- {c}" for c in (constraints or ["Only edit train.py"]))
    directions_text = "\n".join(
        f"{i+1}. {d}" for i, d in enumerate(directions or ["Explore hyperparameter changes"])
    )

    result = template.replace("{goal}", goal)
    result = result.replace("{constraints}", constraints_text)
    result = result.replace("{directions}", directions_text)

    if setup:
        result = result.replace(
            "- Training script: `train.py`",
            setup,
        )

    return result
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_program_builder.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/autoresearch/program/__init__.py src/autoresearch/program/builder.py \
    src/autoresearch/program/templates/default.md tests/test_program_builder.py
git commit -m "feat: add program.md builder with templates"
```

---

## Task 8: Status, Sync & Monitor

**Files:**
- Create: `src/autoresearch/server/monitor.py`
- Create: `scripts/status.sh`
- Create: `scripts/sync.sh`

- [ ] **Step 1: Create scripts/status.sh**

```bash
#!/usr/bin/env bash
# scripts/status.sh — Check experiment status on remote server
# Usage: Run on the remote server

set -euo pipefail

INSTALL_DIR="$HOME/autoresearch"
CHECKPOINT_DIR="$INSTALL_DIR/checkpoints"

echo "=== Autoresearch Status ==="

# Agent status
if screen -list | grep -q "autoresearch-agent"; then
    echo "Agent: RUNNING"
else
    echo "Agent: STOPPED"
fi

echo ""

# GPU experiment statuses
echo "GPU | Branch            | Iter | Best val_bpb | Status  | Last update"
echo "----|-------------------|------|--------------|---------|------------"

for dir in "$CHECKPOINT_DIR"/gpu*/; do
    [ -d "$dir" ] || continue
    STATE_FILE="$dir/state.json"
    if [ -f "$STATE_FILE" ]; then
        # Parse JSON with python (available in conda env)
        python3 -c "
import json, sys
from datetime import datetime, timezone
with open('$STATE_FILE') as f:
    s = json.load(f)
updated = s.get('last_updated', '')
if updated:
    dt = datetime.fromisoformat(updated)
    diff = datetime.now(timezone.utc) - dt
    mins = int(diff.total_seconds() / 60)
    age = f'{mins}m ago'
else:
    age = 'unknown'
print(f\"{s['gpu_id']:<4}| autoresearch/gpu{s['gpu_id']:<2}| {s['iteration']:<5}| {s['best_val_bpb']:<13.6f}| {s['status']:<8}| {age}\")
"
    fi
done

echo ""

# Total experiments
TOTAL=0
BEST_BPB=999
BEST_GPU=""
for dir in "$CHECKPOINT_DIR"/gpu*/; do
    [ -d "$dir" ] || continue
    STATE_FILE="$dir/state.json"
    if [ -f "$STATE_FILE" ]; then
        python3 -c "
import json
with open('$STATE_FILE') as f:
    s = json.load(f)
print(s['iteration'], s['best_val_bpb'], s['gpu_id'])
" | while read iter bpb gpu; do
            TOTAL=$((TOTAL + iter))
        done
    fi
done

echo "Check individual logs: screen -r autoresearch-gpu<N>"
```

- [ ] **Step 2: Create scripts/sync.sh**

```bash
#!/usr/bin/env bash
# scripts/sync.sh — Sync results from remote server to local
# Usage: ./scripts/sync.sh
# Reads credentials from credentials.yaml, syncs checkpoints + results locally.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CREDS_FILE="${PROJECT_DIR}/credentials.yaml"

if [ ! -f "$CREDS_FILE" ]; then
    echo "ERROR: credentials.yaml not found" >&2
    exit 1
fi

HOST=$(grep -A5 'default:' "$CREDS_FILE" | grep 'host:' | head -1 | awk '{print $2}')
PORT=$(grep -A5 'default:' "$CREDS_FILE" | grep 'port:' | head -1 | awk '{print $2}')
USERNAME=$(grep -A5 'default:' "$CREDS_FILE" | grep 'username:' | head -1 | awk '{print $2}')
PASSWORD=$(grep -A5 'default:' "$CREDS_FILE" | grep 'password:' | head -1 | awk '{print $2}')
PORT="${PORT:-22}"

LOCAL_RESULTS="${PROJECT_DIR}/results"
mkdir -p "$LOCAL_RESULTS"

SSH_CMD="sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=accept-new -p $PORT"

echo "Syncing results from ${HOST}..."

rsync -avz --progress \
    -e "$SSH_CMD" \
    "${USERNAME}@${HOST}:~/autoresearch/checkpoints/" \
    "$LOCAL_RESULTS/checkpoints/"

echo ""
echo "Results synced to: $LOCAL_RESULTS/"
echo "Checkpoints: $LOCAL_RESULTS/checkpoints/"
```

- [ ] **Step 3: Implement monitor module**

```python
# src/autoresearch/server/monitor.py
"""Read checkpoints and format status output."""

import json
from datetime import datetime, timezone
from pathlib import Path

from autoresearch.ssh import SSHClient


def get_status(ssh: SSHClient) -> dict:
    """Get full status from remote server.

    Returns dict with keys: agent_running, gpus (list of checkpoint states).
    """
    # Check agent
    agent_result = ssh.run("screen -list | grep autoresearch-agent")
    agent_running = "autoresearch-agent" in agent_result.stdout

    # Read all checkpoints
    result = ssh.run(
        "for f in ~/autoresearch/checkpoints/gpu*/state.json; do "
        "[ -f \"$f\" ] && cat \"$f\" && echo '---SEPARATOR---'; "
        "done"
    )

    gpus = []
    if result.stdout.strip():
        chunks = result.stdout.split("---SEPARATOR---")
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk:
                try:
                    gpus.append(json.loads(chunk))
                except json.JSONDecodeError:
                    continue

    # Sort by GPU ID
    gpus.sort(key=lambda g: g.get("gpu_id", 0))

    return {
        "agent_running": agent_running,
        "gpus": gpus,
    }


def format_status(status: dict, server_host: str, agent_model: str) -> str:
    """Format status dict into a human-readable table."""
    lines = []
    agent_state = "RUNNING" if status["agent_running"] else "STOPPED"
    lines.append(f"Server: {server_host}")
    lines.append(f"Agent:  {agent_model} [{agent_state}]")
    lines.append("")
    lines.append("GPU | Branch            | Iter | Best val_bpb | Status  | Last update")
    lines.append("----|-------------------|------|--------------|---------|------------")

    total_iters = 0
    best_overall = None

    for gpu in status["gpus"]:
        gpu_id = gpu["gpu_id"]
        iteration = gpu["iteration"]
        best_bpb = gpu["best_val_bpb"]
        gpu_status = gpu["status"]

        # Calculate time since last update
        updated = gpu.get("last_updated", "")
        if updated:
            dt = datetime.fromisoformat(updated)
            diff = datetime.now(timezone.utc) - dt
            mins = int(diff.total_seconds() / 60)
            age = f"{mins}m ago"
        else:
            age = "unknown"

        lines.append(
            f"{gpu_id:<4}| autoresearch/gpu{gpu_id:<2}| {iteration:<5}| "
            f"{best_bpb:<13.6f}| {gpu_status:<8}| {age}"
        )

        total_iters += iteration
        if best_overall is None or best_bpb < best_overall[0]:
            best_overall = (best_bpb, gpu_id, gpu.get("best_commit", "?"))

    lines.append("")
    if best_overall:
        lines.append(
            f"Total experiments: {total_iters} | "
            f"Best overall: {best_overall[0]:.6f} (GPU {best_overall[1]}, commit {best_overall[2]})"
        )
    else:
        lines.append("No experiments completed yet.")

    return "\n".join(lines)
```

- [ ] **Step 4: Make scripts executable**

```bash
chmod +x scripts/status.sh scripts/sync.sh
```

- [ ] **Step 5: Commit**

```bash
git add src/autoresearch/server/monitor.py scripts/status.sh scripts/sync.sh
git commit -m "feat: add status monitoring and sync scripts"
```

---

## Task 9: CLI

**Files:**
- Create: `src/autoresearch/cli.py`
- Create: `src/autoresearch/deploy.py`

- [ ] **Step 1: Implement deploy module**

```python
# src/autoresearch/deploy.py
"""Orchestrate full deployment to remote server."""

from pathlib import Path

from autoresearch.config import load_credentials, load_project_config
from autoresearch.ssh import SSHClient
from autoresearch.server.vllm_manager import VLLMManager


def deploy(
    credentials_path: str,
    config_path: str,
    program_md_path: str,
    server: str = "default",
) -> str:
    """Deploy autoresearch to remote server and start experiments.

    Returns status message.
    """
    creds = load_credentials(credentials_path, server=server)
    config = load_project_config(config_path)
    ssh = SSHClient(**creds)

    messages = []

    # Step 1: Upload setup script and run it
    messages.append("Setting up server environment...")
    ssh.upload("scripts/setup_env.sh", "/tmp/autoresearch_setup_env.sh")
    result = ssh.run("bash /tmp/autoresearch_setup_env.sh", timeout=600)
    if result.returncode != 0:
        return f"Setup failed:\n{result.stderr}"
    messages.append("Server environment ready.")

    # Step 2: Upload project files
    messages.append("Uploading project files...")
    ssh.rsync(".", "~/autoresearch/", direction="up")
    messages.append("Files uploaded.")

    # Step 3: Upload program.md
    ssh.upload(program_md_path, "~/autoresearch/program.md")
    messages.append(f"Uploaded program.md from {program_md_path}")

    # Step 4: Start vLLM agent (multi-GPU mode)
    agent_model = config["agent_model"]
    agent_gpu = config["agent_gpu"]
    experiment_gpus = config["experiment_gpus"]

    if len(experiment_gpus) > 0:
        # Multi-GPU: dedicated agent GPU
        messages.append(f"Starting vLLM agent ({agent_model}) on GPU {agent_gpu}...")
        ssh.upload("scripts/start_agent.sh", "~/autoresearch/start_agent.sh")
        result = ssh.run(
            f"bash ~/autoresearch/start_agent.sh '{agent_model}' {agent_gpu}",
            timeout=360,
        )
        messages.append(result.stdout.strip().split("\n")[-1])
    else:
        messages.append("Single-GPU mode: agent will load/unload per cycle.")

    # Step 5: Start experiment runners
    gpu_ids = ",".join(str(g) for g in experiment_gpus) if experiment_gpus else str(agent_gpu)
    messages.append(f"Starting experiment runners on GPUs: {gpu_ids}...")
    ssh.upload("scripts/start_experiments.sh", "~/autoresearch/start_experiments.sh")
    result = ssh.run(
        f"bash ~/autoresearch/start_experiments.sh '{gpu_ids}'",
        timeout=60,
    )
    messages.append("Experiments started.")

    return "\n".join(messages)
```

- [ ] **Step 2: Implement CLI**

```python
# src/autoresearch/cli.py
"""CLI entry points for autoresearch."""

import click
from pathlib import Path


@click.group()
def main():
    """Autoresearch: Autonomous AI research on remote GPU servers."""
    pass


@main.command()
@click.option("--server", default="default", help="Server profile name from credentials.yaml")
def deploy(server):
    """Deploy autoresearch to remote server and start experiments."""
    from autoresearch.deploy import deploy as do_deploy

    project_dir = Path.cwd()
    creds_path = project_dir / "credentials.yaml"
    config_path = project_dir / ".autoresearch" / "config.yaml"
    program_path = project_dir / "program.md"

    if not creds_path.exists():
        click.echo("No credentials.yaml found. Run the /autoresearch skill to set up.")
        raise SystemExit(1)

    if not config_path.exists():
        click.echo("No project config found. Run the /autoresearch skill to configure.")
        raise SystemExit(1)

    if not program_path.exists():
        click.echo("No program.md found. Run the /autoresearch skill to create one.")
        raise SystemExit(1)

    click.echo(do_deploy(str(creds_path), str(config_path), str(program_path), server))


@main.command()
@click.option("--server", default="default", help="Server profile name")
def status(server):
    """Check experiment status on remote server."""
    from autoresearch.config import load_credentials, load_project_config
    from autoresearch.ssh import SSHClient
    from autoresearch.server.monitor import get_status, format_status

    project_dir = Path.cwd()
    creds = load_credentials(project_dir / "credentials.yaml", server=server)
    config = load_project_config(project_dir / ".autoresearch" / "config.yaml")

    ssh = SSHClient(**creds)
    raw_status = get_status(ssh)
    click.echo(format_status(raw_status, creds["host"], config["agent_model"]))


@main.command()
@click.option("--server", default="default", help="Server profile name")
def sync(server):
    """Sync results from remote server to local."""
    import subprocess

    result = subprocess.run(
        ["bash", "scripts/sync.sh"],
        capture_output=True,
        text=True,
    )
    click.echo(result.stdout)
    if result.returncode != 0:
        click.echo(result.stderr, err=True)


@main.command()
@click.argument("target", default="all")
@click.option("--server", default="default", help="Server profile name")
def stop(server, target):
    """Stop experiments. TARGET: all, agent, experiments, gpu<N>."""
    from autoresearch.config import load_credentials
    from autoresearch.ssh import SSHClient

    creds = load_credentials(Path.cwd() / "credentials.yaml", server=server)
    ssh = SSHClient(**creds)

    ssh.upload("scripts/stop.sh", "~/autoresearch/stop.sh")
    result = ssh.run(f"bash ~/autoresearch/stop.sh {target}")
    click.echo(result.stdout)


@main.command()
@click.option("--server", default="default", help="Server profile name")
def resume(server):
    """Resume experiments from last checkpoint."""
    from autoresearch.config import load_credentials, load_project_config
    from autoresearch.ssh import SSHClient

    project_dir = Path.cwd()
    creds = load_credentials(project_dir / "credentials.yaml", server=server)
    config = load_project_config(project_dir / ".autoresearch" / "config.yaml")
    ssh = SSHClient(**creds)

    # Check if agent is running, restart if needed
    agent_check = ssh.run("screen -list | grep autoresearch-agent")
    if "autoresearch-agent" not in agent_check.stdout:
        experiment_gpus = config.get("experiment_gpus", [])
        if len(experiment_gpus) > 0:
            click.echo("Restarting vLLM agent...")
            ssh.run(
                f"bash ~/autoresearch/start_agent.sh "
                f"'{config['agent_model']}' {config['agent_gpu']}",
                timeout=360,
            )

    # Restart experiment runners
    experiment_gpus = config.get("experiment_gpus", [config.get("agent_gpu", 0)])
    gpu_ids = ",".join(str(g) for g in experiment_gpus)
    click.echo(f"Resuming experiment runners on GPUs: {gpu_ids}...")
    result = ssh.run(f"bash ~/autoresearch/start_experiments.sh '{gpu_ids}'")
    click.echo(result.stdout)
```

- [ ] **Step 3: Commit**

```bash
git add src/autoresearch/cli.py src/autoresearch/deploy.py
git commit -m "feat: add CLI commands (deploy, status, sync, stop, resume)"
```

---

## Task 10: Claude Code Skill

**Files:**
- Create: `skill/SKILL.md`

- [ ] **Step 1: Create the skill definition**

```markdown
<!-- skill/SKILL.md -->
---
name: autoresearch
description: Deploy and manage autonomous AI research experiments on remote GPU servers. Interactive setup for credentials, GPU configuration, model selection, and research program creation.
---

# Autoresearch Skill

Deploy and manage autonomous AI research experiments on a remote GPU server.

## Commands

- `/autoresearch` — Full interactive setup and deploy
- `/autoresearch status` — Check experiment progress
- `/autoresearch sync` — Pull results locally
- `/autoresearch stop [all|agent|experiments|gpu<N>]` — Stop experiments
- `/autoresearch resume` — Resume from checkpoints

## Interactive Setup Flow

When the user runs `/autoresearch` (no subcommand), walk through these steps:

### Step 1: Credentials

Check for `credentials.yaml` in the project root.

If it exists, confirm the server details:
```bash
cat credentials.yaml
```

If missing, ask the user for:
1. Server hostname/IP
2. SSH username
3. SSH password
4. SSH port (default: 22)

Save to `credentials.yaml` and test the connection:
```bash
bash scripts/ssh.sh echo CONNECTION_OK
```

### Step 2: Server Discovery

SSH in and detect the environment:
```bash
bash scripts/ssh.sh 'nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader'
```

Show the user their GPU list and ask:
- Which GPU for the agent model? (default: GPU 0)
- Which GPUs for experiments? (default: all others)
- If only 1 GPU: explain single-GPU mode (load/unload cycling, slower but works)

### Step 3: Agent Model Selection

Ask the user which HuggingFace model to use as the research agent.
- Default: `google/gemma-4-27b-it`
- Validate VRAM fit: model should use <90% of agent GPU's memory

Save configuration to `.autoresearch/config.yaml`:
```yaml
agent_model: google/gemma-4-27b-it
agent_gpu: 0
experiment_gpus: [1, 2, 3, 4, 5, 6, 7]
```

### Step 4: Research Program

Ask the user how they want to define their research program:

**(A) Existing file** — "Point me to your program.md"
- Read the file, summarize it, confirm with user

**(B) Repository** — "Give me a repo URL"
- Clone/fetch the repo
- Analyze training code and any existing instructions
- Draft a program.md adapted for the autoresearch loop
- Review with user

**(C) Description** — "Describe what you want to research"
- User provides a text description
- Draft a program.md with constraints, goals, evaluation criteria
- Review with user

**(D) Idea** — "Tell me your idea"
- Ask 3-5 questions to refine scope and goals
- Draft a program.md
- Review with user

All paths produce a `program.md` in the project root.

### Step 5: Deploy

Run the deployment:
```bash
autoresearch deploy
```

This calls the deterministic scripts in order:
1. `scripts/setup_env.sh` (on server)
2. rsync project files
3. `scripts/start_agent.sh` (multi-GPU only)
4. `scripts/start_experiments.sh`

Show initial status when done.

## Status Check

When the user runs `/autoresearch status`:
```bash
autoresearch status
```

## Sync

When the user runs `/autoresearch sync`:
```bash
autoresearch sync
```

## Stop

When the user runs `/autoresearch stop`:
```bash
autoresearch stop [target]
```

## Resume

When the user runs `/autoresearch resume`:
```bash
autoresearch resume
```
```

- [ ] **Step 2: Commit**

```bash
git add skill/SKILL.md
git commit -m "feat: add Claude Code skill for interactive setup"
```

---

## Task 11: Runner Main Loop (Server-Side Entry Point)

**Files:**
- Modify: `src/autoresearch/server/runner.py` (add `__main__` block and full loop)

- [ ] **Step 1: Add the main experiment loop to runner.py**

Append to the existing `runner.py`:

```python
# --- Main experiment loop (runs on the server) ---

import argparse
import subprocess
import time


def run_experiment_loop(
    gpu_id: int,
    checkpoint_dir: str,
    project_dir: str = os.path.expanduser("~/autoresearch"),
    agent_url: str = "http://localhost:8000/v1",
    single_gpu: bool = False,
) -> None:
    """The autonomous experiment loop. Runs until killed."""
    from autoresearch.server.agent import (
        build_experiment_prompt,
        call_agent,
        parse_code_edit,
        apply_diff,
        extract_description,
    )

    train_py_path = os.path.join(project_dir, "train.py")
    program_md_path = os.path.join(project_dir, "program.md")
    results_tsv_path = os.path.join(project_dir, f"results_gpu{gpu_id}.tsv")
    checkpoint_path = os.path.join(checkpoint_dir, "state.json")
    run_log_path = os.path.join(checkpoint_dir, "run.log")

    # Load or initialize state
    if os.path.exists(checkpoint_path):
        state = read_checkpoint(checkpoint_path)
        iteration = state["iteration"]
        best_bpb = state["best_val_bpb"]
    else:
        iteration = 0
        best_bpb = float("inf")

    branch_name = f"autoresearch/gpu{gpu_id}"

    # Ensure we're on the right branch
    subprocess.run(
        ["git", "checkout", "-B", branch_name],
        cwd=project_dir,
        capture_output=True,
    )

    print(f"[GPU {gpu_id}] Starting experiment loop (iteration {iteration}, best={best_bpb})")

    while True:
        iteration += 1
        print(f"\n[GPU {gpu_id}] === Experiment {iteration} ===")

        # Read current state
        program_md = Path(program_md_path).read_text()
        train_py = Path(train_py_path).read_text()
        results_history = ""
        if os.path.exists(results_tsv_path):
            results_history = Path(results_tsv_path).read_text()

        # Ask agent for next experiment
        print(f"[GPU {gpu_id}] Asking agent for next experiment...")
        prompt = build_experiment_prompt(program_md, train_py, results_history, iteration)

        try:
            response = call_agent(prompt, base_url=agent_url)
        except Exception as e:
            print(f"[GPU {gpu_id}] Agent error: {e}")
            time.sleep(30)
            continue

        # Parse and apply edit
        edit = parse_code_edit(response)
        if edit is None:
            print(f"[GPU {gpu_id}] Could not parse agent response, retrying...")
            continue

        description = extract_description(response)
        print(f"[GPU {gpu_id}] Experiment: {description}")

        # Apply edit
        if "<<<SEARCH" in (edit if isinstance(edit, str) else ""):
            new_content = apply_diff(train_py, edit)
        else:
            new_content = edit

        Path(train_py_path).write_text(new_content)

        # Commit
        subprocess.run(
            ["git", "add", "train.py"],
            cwd=project_dir,
            capture_output=True,
        )
        commit_result = subprocess.run(
            ["git", "commit", "-m", f"experiment {iteration}: {description[:80]}"],
            cwd=project_dir,
            capture_output=True,
            text=True,
        )
        commit_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_dir,
            capture_output=True,
            text=True,
        ).stdout.strip()

        # Run training
        print(f"[GPU {gpu_id}] Training...")
        train_start = time.time()
        train_result = subprocess.run(
            ["uv", "run", "train.py"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute hard timeout
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
        )
        train_time = time.time() - train_start

        # Write log
        log_content = train_result.stdout + train_result.stderr
        Path(run_log_path).write_text(log_content)

        # Parse results
        results = parse_results(log_content)
        print(f"[GPU {gpu_id}] Results: val_bpb={results['val_bpb']}, status={results['status']}")

        if results["status"] == "crash":
            memory_gb = 0.0
            update_results_tsv(results_tsv_path, commit_hash, 0.0, 0.0, "crash", description)
            # Revert
            subprocess.run(["git", "reset", "--hard", "HEAD~1"], cwd=project_dir, capture_output=True)
        elif should_keep(results["val_bpb"], best_bpb):
            best_bpb = results["val_bpb"]
            memory_gb = results["peak_vram_mb"] / 1024
            update_results_tsv(results_tsv_path, commit_hash, best_bpb, memory_gb, "keep", description)
            print(f"[GPU {gpu_id}] KEEP — new best: {best_bpb:.6f}")
        else:
            memory_gb = results["peak_vram_mb"] / 1024
            update_results_tsv(
                results_tsv_path, commit_hash, results["val_bpb"], memory_gb, "discard", description
            )
            # Revert
            subprocess.run(["git", "reset", "--hard", "HEAD~1"], cwd=project_dir, capture_output=True)
            print(f"[GPU {gpu_id}] DISCARD — {results['val_bpb']:.6f} >= {best_bpb:.6f}")

        # Checkpoint
        write_checkpoint(
            checkpoint_path,
            gpu_id=gpu_id,
            iteration=iteration,
            best_val_bpb=best_bpb,
            current_commit=commit_hash,
            best_commit=commit_hash if results["status"] != "crash" and should_keep(results["val_bpb"], best_bpb + 1) else "?",
            status="running",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoresearch experiment runner")
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--project-dir", type=str, default=os.path.expanduser("~/autoresearch"))
    parser.add_argument("--agent-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--single-gpu", action="store_true")
    args = parser.parse_args()

    run_experiment_loop(
        gpu_id=args.gpu_id,
        checkpoint_dir=args.checkpoint_dir,
        project_dir=args.project_dir,
        agent_url=args.agent_url,
        single_gpu=args.single_gpu,
    )
```

- [ ] **Step 2: Commit**

```bash
git add src/autoresearch/server/runner.py
git commit -m "feat: add main experiment loop to runner"
```

---

## Task 12: Integration Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""End-to-end integration tests against real server."""

import pytest
from pathlib import Path

CREDS_PATH = Path(__file__).parent.parent / "credentials.yaml"
SKIP_NO_SERVER = pytest.mark.skipif(
    not CREDS_PATH.exists(),
    reason="No credentials.yaml — skipping integration tests",
)


@SKIP_NO_SERVER
def test_full_ssh_and_gpu_discovery():
    """Connect to server and discover GPUs."""
    from autoresearch.config import load_credentials
    from autoresearch.ssh import SSHClient

    creds = load_credentials(CREDS_PATH)
    ssh = SSHClient(**creds)

    # Verify connection
    result = ssh.run("echo CONNECTION_OK")
    assert "CONNECTION_OK" in result.stdout

    # Detect OS
    os_type = ssh.detect_os()
    assert os_type in ("linux", "macos", "windows")

    # Detect GPUs
    gpus = ssh.detect_gpus()
    assert len(gpus) >= 1
    assert "name" in gpus[0]
    assert "memory_mb" in gpus[0]

    print(f"Server OS: {os_type}")
    print(f"GPUs found: {len(gpus)}")
    for gpu in gpus:
        print(f"  GPU {gpu['index']}: {gpu['name']} ({gpu['memory_mb']}MB)")


@SKIP_NO_SERVER
def test_upload_and_run_status_script():
    """Upload status script and run it on server."""
    from autoresearch.config import load_credentials
    from autoresearch.ssh import SSHClient

    creds = load_credentials(CREDS_PATH)
    ssh = SSHClient(**creds)

    ssh.upload("scripts/status.sh", "/tmp/autoresearch_status.sh")
    result = ssh.run("bash /tmp/autoresearch_status.sh")
    # Should at least print the header without crashing
    assert "Autoresearch Status" in result.stdout


@SKIP_NO_SERVER
def test_config_roundtrip():
    """Save and load project config."""
    from autoresearch.config import save_project_config, load_project_config
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.yaml")
        save_project_config(
            path,
            agent_model="google/gemma-4-27b-it",
            agent_gpu=0,
            experiment_gpus=[1, 2, 3, 4, 5, 6, 7],
        )
        config = load_project_config(path)
        assert config["agent_model"] == "google/gemma-4-27b-it"
        assert len(config["experiment_gpus"]) == 7


@SKIP_NO_SERVER
def test_screen_session_lifecycle():
    """Start, list, and stop a screen session on the real server."""
    from autoresearch.config import load_credentials
    from autoresearch.ssh import SSHClient

    creds = load_credentials(CREDS_PATH)
    ssh = SSHClient(**creds)

    session = "autoresearch_integration_test"

    # Clean up any leftover session
    ssh.stop_screen(session)

    # Start
    ssh.start_screen(session, "sleep 300")
    sessions = ssh.list_screens()
    assert session in sessions

    # Stop
    ssh.stop_screen(session)
    sessions = ssh.list_screens()
    assert session not in sessions
```

- [ ] **Step 2: Run integration tests**

```bash
uv run pytest tests/test_integration.py -v --timeout=120
```

Expected: All tests PASS.

- [ ] **Step 3: Run full test suite**

```bash
uv run pytest tests/ -v --timeout=600
```

Expected: All tests PASS (server tests may be skipped if no credentials.yaml).

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: add integration tests for full deployment pipeline"
```

---

## Task 13: Update .gitignore and Final Cleanup

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Update .gitignore**

Ensure these entries are present:

```
# Results and checkpoints (synced from server)
results/
checkpoints/

# Project config (may contain server-specific settings)
.autoresearch/
```

- [ ] **Step 2: Run full test suite one last time**

```bash
uv run pytest tests/ -v --timeout=600
```

Expected: All tests PASS.

- [ ] **Step 3: Final commit**

```bash
git add .gitignore
git commit -m "chore: update gitignore for autoresearch package"
```

---

## Task 14: End-to-End Test — Multidimensional Translation

**Goal:** Validate the full package by using it to improve the `TextToAcousticModel` from the [multidimensional-translation](https://github.com/ContextLab/multidimensional-translation) repo.

**Context:** The repo implements a dual-objective translation system. The trainable component is `TextToAcousticModel` in `src/multidimensional_translation/acoustic_model.py`:
- Architecture: XLM-R-base (frozen or fine-tuned) → mean pool → Linear(768, 1024) → ReLU → Linear(1024, 1024)
- Training: MSE loss between predicted and actual XLS-R acoustic embeddings
- Evaluation metrics: chrF (translation quality), semantic_score, sound_score
- The model predicts XLS-R acoustic embeddings from text, used at decode time for fast phonetic scoring

**Files:**
- Create: `tests/e2e/program_multitrans.md` (the generated program.md for this research)

- [ ] **Step 1: Clone the multidimensional-translation repo locally for analysis**

```bash
cd /tmp
git clone https://github.com/ContextLab/multidimensional-translation.git multitrans-ref
```

- [ ] **Step 2: Analyze the repo to understand the training pipeline**

Read and understand:
- `src/multidimensional_translation/acoustic_model.py` — model architecture + training
- `src/multidimensional_translation/evaluation.py` — metrics (chrF, semantic_score, sound_score)
- `src/multidimensional_translation/audio_similarity.py` — ground-truth audio similarity
- `tests/` — existing tests and data fixtures
- `paper/` — the NeurIPS paper for research context
- `scripts/` — any training/pipeline scripts
- `setup.sh` — environment setup

Key questions to answer:
1. How is the acoustic model trained? (data pipeline, loss function, optimizer, hyperparams)
2. What is the validation metric? (MSE on held-out embeddings? Or downstream chrF/sound_score?)
3. What is the training time budget? (needs to be short enough for iterative experiments)
4. What files can the agent safely modify?

- [ ] **Step 3: Draft program.md for the acoustic model research**

Create `tests/e2e/program_multitrans.md` — a research program that instructs the agent to improve the `TextToAcousticModel`. The program.md should:

1. Define the goal: lower MSE loss (or higher sound_score) on validation data
2. Define setup: how to prepare data, what dependencies are needed
3. Define constraints: which files the agent can modify (acoustic_model.py training code), time budget per experiment
4. Define research directions the agent should explore:
   - Unfreeze XLM-R encoder layers (partial fine-tuning)
   - Try different projection head architectures (deeper, wider, skip connections)
   - Try different pooling strategies (CLS token, attention pooling, weighted layer pooling)
   - Learning rate schedules (warmup, cosine decay)
   - Data augmentation (token dropout, text perturbation)
   - Different loss functions (cosine similarity loss, contrastive loss)
   - Batch size and gradient accumulation tuning
5. Define the output format and keep/discard criteria

- [ ] **Step 4: Review program.md with user**

Present the drafted `program.md` to the user for review. Walk through:
- The research goal and primary metric
- The list of research directions
- The time budget per experiment
- Any constraints or assumptions

Wait for user approval before proceeding.

- [ ] **Step 5: Configure autoresearch for the multidimensional-translation target**

Using the `/autoresearch` skill flow:
```python
from autoresearch.config import save_project_config

save_project_config(
    ".autoresearch/config.yaml",
    agent_model="google/gemma-4-27b-it",
    agent_gpu=0,
    experiment_gpus=[1, 2, 3, 4, 5, 6, 7],
    target_repo="https://github.com/ContextLab/multidimensional-translation",
)
```

- [ ] **Step 6: Deploy to the remote server and start experiments**

```bash
autoresearch deploy
```

This will:
1. Run `setup_env.sh` on the server (installs conda env if needed)
2. Clone the multidimensional-translation repo on the server
3. Install its dependencies in the conda env
4. Upload the program.md
5. Start vLLM on GPU 0
6. Start experiment runners on GPUs 1-7

- [ ] **Step 7: Monitor initial experiments**

```bash
autoresearch status
```

Verify:
- Agent is running and responding
- At least one experiment completes successfully
- Results are being logged to results.tsv
- Checkpoints are being written

- [ ] **Step 8: Sync results and review**

```bash
autoresearch sync
```

Review:
- Were experiments meaningful? (not trivial/broken changes)
- Did any experiments improve the metric?
- Are the agent's code modifications sensible?

- [ ] **Step 9: Commit the e2e test artifacts**

```bash
git add tests/e2e/program_multitrans.md
git commit -m "test: add end-to-end test with multidimensional-translation repo"
```
