"""SSH client wrapping sshpass + subprocess for remote server operations."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommandResult:
    """Result of a remote command execution."""

    stdout: str
    stderr: str
    returncode: int


class SSHClient:
    """SSH client using sshpass for password-based authentication.

    Args:
        host: Remote hostname or IP.
        username: SSH username.
        password: SSH password.
        port: SSH port (default 22).
    """

    SSH_FLAGS = [
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=10",
    ]

    def __init__(self, host: str, username: str, password: str, port: int = 22) -> None:
        if not shutil.which("sshpass"):
            raise RuntimeError(
                "sshpass is not installed or not on PATH. "
                "Install it (e.g. brew install sshpass, apt install sshpass)."
            )
        self.host = host
        self.username = username
        self.password = password
        self.port = port

    # ── internal helpers ────────────────────────────────────────────────

    def _sshpass_prefix(self) -> list[str]:
        return ["sshpass", "-p", self.password]

    def _ssh_base(self) -> list[str]:
        return [
            *self._sshpass_prefix(),
            "ssh",
            *self.SSH_FLAGS,
            "-p", str(self.port),
            f"{self.username}@{self.host}",
        ]

    def _sshpass_ssh_cmd(self) -> str:
        """Return the ssh command string for use in rsync -e flag."""
        flags = " ".join(self.SSH_FLAGS)
        return f"sshpass -p {self.password} ssh {flags} -p {self.port}"

    # ── public API ──────────────────────────────────────────────────────

    def run(self, command: str, timeout: int = 120) -> CommandResult:
        """Run a command on the remote server.

        Args:
            command: Shell command string to execute remotely.
            timeout: Timeout in seconds (default 120).

        Returns:
            CommandResult with stdout, stderr, and returncode.
        """
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

    def upload(self, local_path: str | Path, remote_path: str) -> None:
        """Upload a file to the remote server via scp.

        Args:
            local_path: Local file path.
            remote_path: Remote destination path.
        """
        cmd = [
            *self._sshpass_prefix(),
            "scp",
            *self.SSH_FLAGS,
            "-P", str(self.port),
            str(local_path),
            f"{self.username}@{self.host}:{remote_path}",
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    def download(self, remote_path: str, local_path: str | Path) -> None:
        """Download a file from the remote server via scp.

        Args:
            remote_path: Remote file path.
            local_path: Local destination path.
        """
        cmd = [
            *self._sshpass_prefix(),
            "scp",
            *self.SSH_FLAGS,
            "-P", str(self.port),
            f"{self.username}@{self.host}:{remote_path}",
            str(local_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    def rsync(
        self,
        local_path: str | Path,
        remote_path: str,
        direction: str = "up",
        exclude: list[str] | None = None,
    ) -> None:
        """Rsync files between local and remote.

        Args:
            local_path: Local path.
            remote_path: Remote path.
            direction: "up" (local -> remote) or "down" (remote -> local).
            exclude: List of rsync --exclude patterns.
        """
        ssh_cmd = self._sshpass_ssh_cmd()
        remote = f"{self.username}@{self.host}:{remote_path}"

        cmd = ["rsync", "-az", "-e", ssh_cmd]
        for pattern in (exclude or []):
            cmd.extend(["--exclude", pattern])

        if direction == "up":
            cmd.extend([str(local_path), remote])
        elif direction == "down":
            cmd.extend([remote, str(local_path)])
        else:
            raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")

        subprocess.run(cmd, check=True, capture_output=True, text=True)

    def start_screen(self, session_name: str, command: str) -> None:
        """Start a detached screen session on the remote server.

        Args:
            session_name: Name for the screen session.
            command: Command to run inside the screen session.
        """
        self.run(f"screen -dmS {session_name} bash -c {command!r}")

    def stop_screen(self, session_name: str) -> None:
        """Kill a screen session on the remote server.

        Args:
            session_name: Name of the screen session to kill.
        """
        self.run(f"screen -S {session_name} -X quit")

    def list_screens(self) -> list[str]:
        """List active screen sessions on the remote server.

        Returns:
            List of screen session name strings.
        """
        result = self.run("screen -ls")
        sessions: list[str] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            # Screen lines look like: "12345.session_name\t(Detached)"
            if "." in line and ("Detached" in line or "Attached" in line):
                # Extract the part before the first tab/space with status
                name_part = line.split("\t")[0].split(" ")[0]
                sessions.append(name_part)
        return sessions

    def detect_gpus(self) -> list[dict]:
        """Detect GPUs on the remote server via nvidia-smi.

        Returns:
            List of dicts with keys: index (int), name (str), memory_mb (int).
        """
        result = self.run(
            "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits"
        )
        if result.returncode != 0:
            return []

        gpus: list[dict] = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_mb": int(parts[2]),
                })
        return gpus

    def detect_os(self) -> str:
        """Detect the operating system on the remote server.

        Returns:
            "linux", "macos", or "windows".
        """
        result = self.run("uname -s")
        uname = result.stdout.strip().lower()

        if uname == "darwin":
            return "macos"

        if uname == "linux":
            # Check for WSL
            wsl_check = self.run("cat /proc/version 2>/dev/null || true")
            if "microsoft" in wsl_check.stdout.lower():
                return "windows"
            return "linux"

        return uname
