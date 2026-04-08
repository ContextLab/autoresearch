"""End-to-end integration tests against real server."""

from __future__ import annotations

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
    assert result.returncode == 0

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
def test_config_roundtrip(tmp_path):
    """Save and load project config."""
    from autoresearch.config import save_project_config, load_project_config

    path = str(tmp_path / "config.yaml")
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
    assert any(session in s for s in sessions), f"{session} not found in {sessions}"

    # Stop
    ssh.stop_screen(session)
    sessions = ssh.list_screens()
    assert not any(session in s for s in sessions), f"{session} still in {sessions}"


@SKIP_NO_SERVER
def test_monitor_format():
    """Test status formatting with fake data."""
    from autoresearch.server.monitor import format_status

    status = {
        "agent_running": True,
        "gpus": [
            {
                "gpu_id": 1,
                "iteration": 42,
                "best_val_bpb": 0.9821,
                "status": "running",
                "best_commit": "abc1234",
                "last_updated": "2026-04-05T22:30:00+00:00",
            }
        ],
    }
    output = format_status(status, "gpu-server.example.com", "google/gemma-4-27b-it")
    assert "gpu-server.example.com" in output
    assert "0.9821" in output
    assert "running" in output.lower()
