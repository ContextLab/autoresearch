"""Tests for autoresearch.config module — uses real temp files, no mocks."""

import os
import tempfile

import pytest
import yaml

from autoresearch.config import (
    load_credentials,
    load_project_config,
    save_credentials,
    save_project_config,
)


@pytest.fixture
def tmp_yaml(tmp_path):
    """Return a helper that writes YAML content to a temp file and returns the path."""
    def _write(data, name="test.yaml"):
        p = tmp_path / name
        p.write_text(yaml.dump(data))
        return str(p)
    return _write


# ── Credentials ──────────────────────────────────────────────────────────────


def test_load_credentials_from_yaml(tmp_yaml):
    path = tmp_yaml({
        "servers": {
            "default": {
                "host": "example.com",
                "port": 22,
                "username": "user",
                "password": "pass",
            }
        }
    })
    creds = load_credentials(path)
    assert creds["host"] == "example.com"
    assert creds["port"] == 22
    assert creds["username"] == "user"
    assert creds["password"] == "pass"


def test_load_credentials_missing_file():
    with pytest.raises(FileNotFoundError):
        load_credentials("/nonexistent/path/credentials.yaml")


def test_load_credentials_named_server(tmp_yaml):
    path = tmp_yaml({
        "servers": {
            "default": {
                "host": "default.example.com",
                "port": 22,
                "username": "defaultuser",
                "password": "defaultpass",
            },
            "gpu-box": {
                "host": "gpu.example.com",
                "port": 2222,
                "username": "gpuuser",
                "password": "gpupass",
            },
        }
    })
    creds = load_credentials(path, server="gpu-box")
    assert creds["host"] == "gpu.example.com"
    assert creds["port"] == 2222
    assert creds["username"] == "gpuuser"
    assert creds["password"] == "gpupass"


def test_save_credentials(tmp_path):
    path = str(tmp_path / "creds.yaml")
    save_credentials(path, server="default", host="h.com", port=22, username="u", password="p")

    creds = load_credentials(path)
    assert creds == {"host": "h.com", "port": 22, "username": "u", "password": "p"}


# ── Project config ───────────────────────────────────────────────────────────


def test_save_project_config(tmp_path):
    path = str(tmp_path / "project.yaml")
    save_project_config(
        path,
        agent_model="claude-opus-4",
        agent_gpu=0,
        experiment_gpus=[1, 2],
        time_budget=300,
    )
    cfg = load_project_config(path)
    assert cfg["agent_model"] == "claude-opus-4"
    assert cfg["agent_gpu"] == 0
    assert cfg["experiment_gpus"] == [1, 2]
    assert cfg["time_budget"] == 300


def test_load_project_config(tmp_yaml):
    path = tmp_yaml({
        "agent_model": "gpt-4o",
        "agent_gpu": 1,
        "experiment_gpus": [2, 3],
    })
    cfg = load_project_config(path)
    assert cfg["agent_model"] == "gpt-4o"
    assert cfg["agent_gpu"] == 1
    assert cfg["experiment_gpus"] == [2, 3]
