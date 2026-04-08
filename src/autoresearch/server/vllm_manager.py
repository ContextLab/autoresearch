"""Manage vLLM lifecycle on a remote server via SSH."""

from __future__ import annotations

import time
from pathlib import Path

from autoresearch.ssh import SSHClient

# Path to the start script, relative to the project root
_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"

SESSION_NAME = "autoresearch-agent"
HEALTH_TIMEOUT = 300  # seconds to wait for vLLM to become ready
HEALTH_POLL_INTERVAL = 5  # seconds between health checks


class VLLMManager:
    """Manage a vLLM server on a remote machine.

    Args:
        ssh: An authenticated SSHClient instance.
        model: HuggingFace model name/path to serve.
        gpu_id: CUDA device index (default 0).
        port: Port for the OpenAI-compatible API (default 8000).
    """

    def __init__(
        self,
        ssh: SSHClient,
        model: str,
        gpu_id: int = 0,
        port: int = 8000,
    ) -> None:
        self.ssh = ssh
        self.model = model
        self.gpu_id = gpu_id
        self.port = port

    # ── public API ──────────────────────────────────────────────────────

    def start(self) -> str:
        """Upload and run ``start_agent.sh`` on the remote server.

        Returns:
            Status message from the remote script.
        """
        local_script = _SCRIPTS_DIR / "start_agent.sh"
        remote_script = "~/autoresearch/start_agent.sh"

        # Upload the start script and make it executable
        self.ssh.upload(local_script, remote_script)
        self.ssh.run(f"chmod +x {remote_script}")

        # Run the script (allow up to 6 minutes for startup + readiness wait)
        result = self.ssh.run(
            f"{remote_script} {self.model!r} {self.gpu_id}",
            timeout=400,
        )
        return result.stdout + result.stderr

    def stop(self) -> str:
        """Kill the vLLM screen session.

        Returns:
            Status message.
        """
        self.ssh.stop_screen(SESSION_NAME)
        return f"Stopped session {SESSION_NAME}"

    def health(self) -> bool:
        """Check the ``/health`` endpoint of the vLLM server.

        Returns:
            True if the server reports healthy, False otherwise.
        """
        result = self.ssh.run(
            f"curl -s http://localhost:{self.port}/health",
            timeout=10,
        )
        body = result.stdout.lower()
        return result.returncode == 0 and ("ok" in body or "healthy" in body)

    def is_running(self) -> bool:
        """Check whether the vLLM screen session exists.

        Returns:
            True if the session is listed, False otherwise.
        """
        result = self.ssh.run("screen -ls")
        return SESSION_NAME in result.stdout
