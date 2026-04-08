"""Tests for shell scripts — real server tests, no mocks."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

CREDS_PATH = Path(__file__).parent.parent / "credentials.yaml"
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"

SKIP_NO_SERVER = pytest.mark.skipif(
    not CREDS_PATH.exists(),
    reason="No credentials.yaml — skipping real server tests",
)


@SKIP_NO_SERVER
def test_ssh_script_runs():
    """Run ssh.sh with a simple echo command and verify output."""
    result = subprocess.run(
        ["bash", str(SCRIPTS_DIR / "ssh.sh"), "echo", "CONNECTION_OK"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"ssh.sh failed: {result.stderr}"
    assert "CONNECTION_OK" in result.stdout


@SKIP_NO_SERVER
def test_setup_env_is_idempotent():
    """Upload setup_env.sh, run it twice, verify both succeed and second says 'already'."""
    from autoresearch.config import load_credentials
    from autoresearch.ssh import SSHClient

    creds = load_credentials(CREDS_PATH)
    ssh = SSHClient(**creds)

    remote_path = "/tmp/autoresearch_setup_env.sh"

    # Upload via SSHClient
    ssh.upload(str(SCRIPTS_DIR / "setup_env.sh"), remote_path)

    # First run
    run1 = ssh.run(f"bash {remote_path}", timeout=600)
    assert run1.returncode == 0, f"First run failed: {run1.stderr}\nstdout: {run1.stdout}"
    assert "=== Setup complete ===" in run1.stdout, f"First run did not complete: {run1.stdout}"

    # Second run — should be fast and say "already" for most steps
    run2 = ssh.run(f"bash {remote_path}", timeout=600)
    assert run2.returncode == 0, f"Second run failed: {run2.stderr}\nstdout: {run2.stdout}"
    assert "=== Setup complete ===" in run2.stdout, f"Second run did not complete: {run2.stdout}"
    assert "already" in run2.stdout.lower(), (
        f"Second run should report 'already' for idempotent steps: {run2.stdout}"
    )

    # Cleanup
    ssh.run(f"rm -f {remote_path}")
