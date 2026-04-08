"""Orchestrate full deployment of autoresearch to a remote server."""

from __future__ import annotations

from pathlib import Path

from autoresearch.config import load_credentials, load_project_config
from autoresearch.ssh import SSHClient


def deploy(
    credentials_path: str | Path,
    config_path: str | Path,
    program_md_path: str | Path,
    server: str = "default",
) -> str:
    """Deploy autoresearch to a remote server.

    Steps:
        1. Load credentials and project config.
        2. Create SSHClient.
        3. Upload and run setup_env.sh (timeout 600s).
        4. Rsync project files to ~/autoresearch/.
        5. Upload program.md.
        6. For multi-GPU setups: start vLLM agent.
        7. Start experiment runners.

    Args:
        credentials_path: Path to credentials YAML file.
        config_path: Path to project config YAML file.
        program_md_path: Path to program.md file.
        server: Server profile name.

    Returns:
        Status message string.
    """
    messages: list[str] = []

    # 1. Load credentials and config
    creds = load_credentials(credentials_path, server)
    config = load_project_config(config_path)
    messages.append(f"Loaded credentials for server '{server}' ({creds['host']})")

    # 2. Create SSH client
    ssh = SSHClient(
        host=creds["host"],
        username=creds["username"],
        password=creds["password"],
        port=creds.get("port", 22),
    )
    messages.append("SSH client connected")

    # 3. Upload and run setup_env.sh in screen (survives SSH drops)
    import time

    scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
    setup_script = scripts_dir / "setup_env.sh"
    if setup_script.exists():
        ssh.upload(str(setup_script), "~/setup_env.sh")

        # Run in screen so it survives SSH timeouts
        # Use a marker file to detect completion
        # If previous run failed, use --clean to start fresh
        prev_failed = ssh.run("test -f ~/setup_env.failed && echo YES || echo NO")
        clean_flag = "--clean" if "YES" in prev_failed.stdout else ""
        ssh.run("rm -f ~/setup_env.done ~/setup_env.failed")
        ssh.run(
            'screen -dmS autoresearch-setup bash -c '
            f'"bash ~/setup_env.sh {clean_flag} > ~/setup_env.log 2>&1 '
            '&& touch ~/setup_env.done || touch ~/setup_env.failed"'
        )
        messages.append("Environment setup started (in screen session)")

        # Poll for completion
        poll_interval = 15  # seconds
        max_wait = 3600  # 1 hour
        elapsed = 0
        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval

            # Check for completion markers FIRST (screen may exit before next poll)
            check = ssh.run(
                "if [ -f ~/setup_env.done ]; then echo DONE; "
                "elif [ -f ~/setup_env.failed ]; then echo FAILED; "
                "elif screen -list 2>/dev/null | grep -q autoresearch-setup; then echo RUNNING; "
                "else echo GONE; fi"
            )
            status = check.stdout.strip()

            if status == "DONE":
                messages.append(f"Environment setup completed ({elapsed}s)")
                break
            elif status == "FAILED":
                log = ssh.run("tail -30 ~/setup_env.log 2>/dev/null")
                messages.append(f"Environment setup FAILED after {elapsed}s:")
                messages.append(log.stdout[-1000:] if log.stdout else "no log output")
                return "\n".join(messages)
            elif status == "GONE":
                # Screen died without markers — check one more time (race)
                time.sleep(2)
                recheck = ssh.run("test -f ~/setup_env.done && echo DONE || echo GONE")
                if "DONE" in recheck.stdout:
                    messages.append(f"Environment setup completed ({elapsed}s)")
                    break
                messages.append(f"Environment setup screen died after {elapsed}s (no marker)")
                return "\n".join(messages)

            if elapsed % 60 == 0:
                messages.append(f"  ...setup still running ({elapsed}s)")
        else:
            messages.append(f"Environment setup timed out after {max_wait}s")
            return "\n".join(messages)
    else:
        messages.append(f"Warning: setup_env.sh not found at {setup_script}")

    # 4. Rsync project files to ~/autoresearch/ (exclude secrets and unnecessary files)
    project_root = Path(__file__).resolve().parent.parent.parent
    ssh.run("mkdir -p ~/autoresearch")
    ssh.rsync(
        f"{project_root}/",
        "~/autoresearch/",
        direction="up",
        exclude=[
            "credentials.yaml",
            ".autoresearch/",
            ".git/",
            ".venv/",
            ".omc/",
            ".DS_Store",
            "__pycache__/",
            "*.pyc",
            "uv.lock",
            "results/",
            "notes/",
        ],
    )
    messages.append("Project files synced to ~/autoresearch/")

    # 4b. Install autoresearch package on server
    result = ssh.run(
        'source $HOME/miniforge3/etc/profile.d/conda.sh 2>/dev/null || eval "$(conda shell.bash hook)" && '
        'conda activate autoresearch && '
        'cd ~/autoresearch && pip install -e . --quiet 2>&1 | tail -3',
        timeout=120,
    )
    if result.returncode == 0:
        messages.append("autoresearch package installed on server")
    else:
        messages.append(f"Warning: package install failed: {result.stdout[-200:]}")

    # 5. Sync target repo if specified
    # Clone locally first (has git credentials), then rsync to server
    target_repo = config.get("target_repo")
    target_local = config.get("target_local_path")  # pre-cloned local path
    if target_repo or target_local:
        import subprocess as _sp
        import tempfile

        # Find or create local copy
        if target_local and Path(target_local).exists():
            local_target = Path(target_local)
            messages.append(f"Using existing local target: {local_target}")
        else:
            local_target = Path(tempfile.gettempdir()) / "autoresearch-target"
            if (local_target / ".git").exists():
                _sp.run(["git", "-C", str(local_target), "pull"], capture_output=True)
                messages.append(f"Target repo updated locally: {target_repo}")
            else:
                result = _sp.run(
                    ["git", "clone", "--depth", "1", target_repo, str(local_target)],
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    messages.append(f"Warning: local clone failed: {result.stderr[-200:]}")
                    messages.append("Tip: set target_local_path in config to a pre-cloned directory")
                else:
                    messages.append(f"Target repo cloned locally: {target_repo}")

        # Rsync to server (only if local clone succeeded)
        if not local_target.exists():
            messages.append("Skipping target repo sync (no local copy)")
        else:
            ssh.run("mkdir -p ~/autoresearch/target")
            ssh.rsync(
                f"{local_target}/",
                "~/autoresearch/target/",
                direction="up",
                exclude=[".git/"],
            )
            messages.append("Target repo synced to server")

            # Install target repo deps into autoresearch env (not setup.sh which creates its own env)
            result = ssh.run(
                'source $HOME/miniforge3/etc/profile.d/conda.sh 2>/dev/null || eval "$(conda shell.bash hook)" && '
                "conda activate autoresearch && "
                "cd ~/autoresearch/target && "
                "if [ -f pyproject.toml ]; then pip install -e . 2>&1 | tail -5; "
                "elif [ -f requirements.txt ]; then pip install -r requirements.txt 2>&1 | tail -5; "
                "elif [ -f setup.py ]; then pip install -e . 2>&1 | tail -5; "
                "else echo 'No deps file found'; "
                "fi",
                timeout=600,
            )
            if result.returncode == 0:
                messages.append("Target repo dependencies installed")
            else:
                messages.append(f"Warning: target deps install: {result.stdout[-300:]}")

    # 6. Upload program.md
    program_md = Path(program_md_path)
    if program_md.exists():
        ssh.upload(str(program_md), "~/autoresearch/program.md")
        messages.append("program.md uploaded")
    else:
        messages.append(f"Warning: program.md not found at {program_md}")

    # 6b. Upload run_config.yaml if it exists
    run_config_local = Path.cwd() / ".autoresearch" / "run_config.yaml"
    if run_config_local.exists():
        ssh.run("mkdir -p ~/autoresearch/.autoresearch")
        ssh.upload(str(run_config_local), "~/autoresearch/.autoresearch/run_config.yaml")
        messages.append("run_config.yaml uploaded")

    # 6c. Upload project config (for runner to read agent_model etc.)
    config_local = Path.cwd() / ".autoresearch" / "config.yaml"
    if config_local.exists():
        ssh.upload(str(config_local), "~/autoresearch/.autoresearch/config.yaml")

    # 7. For multi-GPU: start vLLM agent
    experiment_gpus = config.get("experiment_gpus", [])
    agent_gpu = config.get("agent_gpu")
    agent_model = config.get("agent_model", "google/gemma-4-26B-A4B-it")
    agent_dtype = config.get("agent_dtype", "auto")
    agent_backend = config.get("agent_backend", "transformers")
    agent_model_file = config.get("agent_model_file", "")
    if agent_gpu is not None and len(experiment_gpus) > 0:
        agent_script = scripts_dir / "start_agent.sh"
        if agent_script.exists():
            ssh.upload(str(agent_script), "~/autoresearch/start_agent.sh")
            env_vars = (
                f"AGENT_DTYPE={agent_dtype} "
                f"AGENT_BACKEND={agent_backend} "
                f"AGENT_MODEL_FILE={agent_model_file}"
            )
            result = ssh.run(
                f"{env_vars} bash ~/autoresearch/start_agent.sh '{agent_model}' {agent_gpu} --no-wait",
                timeout=120,
            )
            if result.returncode == 0:
                messages.append(f"vLLM agent launched ({agent_model} on GPU {agent_gpu})")
                messages.append("  Model loading in background — use 'autoresearch status' to check")
            else:
                messages.append(f"vLLM agent launch failed: {result.stdout[-300:]}")
        else:
            messages.append(f"Warning: start_agent.sh not found at {agent_script}")

    # 8. Start experiment runners
    gpu_ids = ",".join(str(g) for g in experiment_gpus) if experiment_gpus else str(agent_gpu or 0)
    experiments_script = scripts_dir / "start_experiments.sh"
    if experiments_script.exists():
        ssh.upload(str(experiments_script), "~/autoresearch/start_experiments.sh")
        result = ssh.run(
            f"bash ~/autoresearch/start_experiments.sh '{gpu_ids}'",
            timeout=300,
        )
        if result.returncode == 0:
            messages.append("Experiment runners started")
        else:
            messages.append(f"Experiment runners failed: {result.stderr[:200]}")
    else:
        messages.append(f"Warning: start_experiments.sh not found at {experiments_script}")

    return "\n".join(messages)
