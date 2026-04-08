"""Tests for autoresearch.ssh module — real server tests, no mocks."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pytest

from autoresearch.config import load_credentials
from autoresearch.ssh import CommandResult, SSHClient

CREDS_PATH = Path(__file__).parent.parent / "credentials.yaml"
SKIP_NO_SERVER = pytest.mark.skipif(
    not CREDS_PATH.exists(),
    reason="No credentials.yaml — skipping real server tests",
)


@pytest.fixture(scope="module")
def ssh():
    """Create an SSHClient from credentials.yaml."""
    creds = load_credentials(CREDS_PATH)
    return SSHClient(
        host=creds["host"],
        username=creds["username"],
        password=creds["password"],
        port=creds.get("port", 22),
    )


# ── Remote command execution ────────────────────────────────────────────────


@SKIP_NO_SERVER
def test_ssh_run_command(ssh):
    result = ssh.run("echo hello")
    assert isinstance(result, CommandResult)
    assert "hello" in result.stdout
    assert result.returncode == 0


@SKIP_NO_SERVER
def test_ssh_run_command_failure(ssh):
    result = ssh.run("exit 42")
    assert result.returncode == 42


# ── OS and GPU detection ───────────────────────────────────────────────────


@SKIP_NO_SERVER
def test_ssh_detect_os(ssh):
    os_name = ssh.detect_os()
    assert os_name in ("linux", "macos", "windows")


@SKIP_NO_SERVER
def test_ssh_detect_gpus(ssh):
    gpus = ssh.detect_gpus()
    assert len(gpus) >= 1
    gpu = gpus[0]
    assert "index" in gpu
    assert "name" in gpu
    assert "memory_mb" in gpu
    assert isinstance(gpu["index"], int)
    assert isinstance(gpu["memory_mb"], int)


# ── File transfer ──────────────────────────────────────────────────────────


@SKIP_NO_SERVER
def test_ssh_file_transfer(ssh):
    tag = uuid.uuid4().hex[:8]
    content = f"autoresearch-test-{tag}"
    remote_path = f"/tmp/autoresearch_test_{tag}.txt"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        local_upload = f.name

    try:
        # Upload
        ssh.upload(local_upload, remote_path)

        # Verify content on remote
        result = ssh.run(f"cat {remote_path}")
        assert result.returncode == 0
        assert content in result.stdout

        # Download
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            local_download = f.name

        ssh.download(remote_path, local_download)
        assert Path(local_download).read_text() == content

    finally:
        # Cleanup remote
        ssh.run(f"rm -f {remote_path}")
        # Cleanup local
        Path(local_upload).unlink(missing_ok=True)
        if "local_download" in locals():
            Path(local_download).unlink(missing_ok=True)


# ── Screen session management ──────────────────────────────────────────────


@SKIP_NO_SERVER
def test_ssh_screen_management(ssh):
    session_name = f"artest_{uuid.uuid4().hex[:8]}"

    try:
        # Start a screen session running sleep
        ssh.start_screen(session_name, "sleep 300")

        # Give screen a moment to register
        import time
        time.sleep(1)

        # Verify it appears in list
        screens = ssh.list_screens()
        matching = [s for s in screens if session_name in s]
        assert len(matching) >= 1, f"Session {session_name} not found in {screens}"

        # Stop it
        ssh.stop_screen(session_name)
        time.sleep(1)

        # Verify it's gone
        screens = ssh.list_screens()
        matching = [s for s in screens if session_name in s]
        assert len(matching) == 0, f"Session {session_name} still present in {screens}"

    finally:
        # Best-effort cleanup
        ssh.run(f"screen -S {session_name} -X quit 2>/dev/null || true")
