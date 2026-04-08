"""Configuration loading and saving for autoresearch.

Credentials YAML format:
    servers:
      default:
        host: example.com
        port: 22
        username: user
        password: pass

Project config is a flat YAML dict with agent_model, agent_gpu, experiment_gpus, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_credentials(path: str | Path, server: str = "default") -> dict[str, Any]:
    """Load credentials for a named server from a YAML file.

    Args:
        path: Path to the credentials YAML file.
        server: Server profile name (default: "default").

    Returns:
        Dict with host, port, username, password.

    Raises:
        FileNotFoundError: If the credentials file does not exist.
        KeyError: If the named server is not found.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Credentials file not found: {path}")
    data = yaml.safe_load(p.read_text())
    return data["servers"][server]


def save_credentials(
    path: str | Path,
    server: str,
    host: str,
    port: int,
    username: str,
    password: str,
) -> None:
    """Save credentials for a named server to a YAML file.

    If the file already exists, the server entry is updated (other servers preserved).

    Args:
        path: Path to the credentials YAML file.
        server: Server profile name.
        host: SSH host.
        port: SSH port.
        username: SSH username.
        password: SSH password.
    """
    p = Path(path)
    if p.exists():
        data = yaml.safe_load(p.read_text()) or {}
    else:
        data = {}

    data.setdefault("servers", {})
    data["servers"][server] = {
        "host": host,
        "port": port,
        "username": username,
        "password": password,
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.dump(data, default_flow_style=False))


def load_project_config(path: str | Path) -> dict[str, Any]:
    """Load project configuration from a YAML file.

    Args:
        path: Path to the project config YAML file.

    Returns:
        Dict with project configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Project config file not found: {path}")
    return yaml.safe_load(p.read_text())


def save_project_config(
    path: str | Path,
    agent_model: str,
    agent_gpu: int,
    experiment_gpus: list[int],
    **extra: Any,
) -> None:
    """Save project configuration to a YAML file.

    Args:
        path: Path to the project config YAML file.
        agent_model: Model name for the agent.
        agent_gpu: GPU index for the agent.
        experiment_gpus: List of GPU indices for experiments.
        **extra: Additional config values to include.
    """
    data: dict[str, Any] = {
        "agent_model": agent_model,
        "agent_gpu": agent_gpu,
        "experiment_gpus": experiment_gpus,
    }
    data.update(extra)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.dump(data, default_flow_style=False))
