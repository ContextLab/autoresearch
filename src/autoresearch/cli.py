"""Click CLI for autoresearch: autonomous AI research on remote GPU servers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import click

from autoresearch.config import load_credentials, load_project_config
from autoresearch.server.monitor import format_status, get_status
from autoresearch.ssh import SSHClient


def _get_paths() -> tuple[Path, Path, Path]:
    """Return standard paths for credentials, config, and program.md."""
    cwd = Path.cwd()
    credentials = cwd / "credentials.yaml"
    config = cwd / ".autoresearch" / "config.yaml"
    program_md = cwd / "program.md"
    return credentials, config, program_md


def _make_ssh(server: str) -> tuple[SSHClient, dict, dict]:
    """Create an SSHClient from credentials, returning (ssh, creds, config)."""
    credentials_path, config_path, _ = _get_paths()
    creds = load_credentials(credentials_path, server)
    config = load_project_config(config_path)
    ssh = SSHClient(
        host=creds["host"],
        username=creds["username"],
        password=creds["password"],
        port=creds.get("port", 22),
    )
    return ssh, creds, config


@click.group()
def main() -> None:
    """Autoresearch: Autonomous AI research on remote GPU servers."""


@main.command()
@click.option("--server", default="default", help="Server profile name from credentials.yaml.")
def deploy(server: str) -> None:
    """Deploy and start experiments on remote server."""
    from autoresearch.deploy import deploy as run_deploy

    credentials_path, config_path, program_md_path = _get_paths()
    try:
        result = run_deploy(credentials_path, config_path, program_md_path, server=server)
        click.echo(result)
    except Exception as e:
        raise click.ClickException(str(e))


@main.command()
@click.option("--server", default="default", help="Server profile name from credentials.yaml.")
def status(server: str) -> None:
    """Check experiment status on remote server."""
    try:
        ssh, creds, config = _make_ssh(server)
        st = get_status(ssh)
        output = format_status(st, creds["host"], config.get("agent_model", "unknown"))
        click.echo(output)
    except Exception as e:
        raise click.ClickException(str(e))


@main.command()
@click.option("--server", default="default", help="Server profile name from credentials.yaml.")
def sync(server: str) -> None:
    """Sync results from remote server to local."""
    scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
    sync_script = scripts_dir / "sync.sh"

    if not sync_script.exists():
        # Fall back to rsync directly
        try:
            ssh, creds, _config = _make_ssh(server)
            local_results = Path.cwd() / "results"
            local_results.mkdir(exist_ok=True)
            ssh.rsync(str(local_results) + "/", "~/autoresearch/results/", direction="down")
            click.echo("Results synced successfully.")
        except Exception as e:
            raise click.ClickException(str(e))
    else:
        try:
            result = subprocess.run(
                ["bash", str(sync_script), server],
                capture_output=True,
                text=True,
                check=True,
            )
            click.echo(result.stdout)
        except subprocess.CalledProcessError as e:
            raise click.ClickException(f"Sync failed: {e.stderr}")


@main.command()
@click.argument("target", default="all")
@click.option("--server", default="default", help="Server profile name from credentials.yaml.")
def stop(target: str, server: str) -> None:
    """Stop experiments. TARGET: all, agent, experiments, gpu<N>."""
    scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
    stop_script = scripts_dir / "stop.sh"

    try:
        ssh, _creds, _config = _make_ssh(server)
        if stop_script.exists():
            ssh.upload(stop_script, "~/stop.sh")
            result = ssh.run(f"bash ~/stop.sh {target}", timeout=60)
            if result.returncode == 0:
                click.echo(f"Stopped: {target}")
                if result.stdout.strip():
                    click.echo(result.stdout.strip())
            else:
                raise click.ClickException(f"Stop failed: {result.stderr[:200]}")
        else:
            raise click.ClickException(f"stop.sh not found at {stop_script}")
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(str(e))


@main.command()
@click.option("--server", default="default", help="Server profile name from credentials.yaml.")
def resume(server: str) -> None:
    """Resume experiments from checkpoints."""
    scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"

    try:
        ssh, _creds, config = _make_ssh(server)

        # Check if agent needs restarting (multi-GPU setup)
        agent_gpu = config.get("agent_gpu")
        agent_model = config.get("agent_model", "google/gemma-4-26B-A4B-it")
        agent_dtype = config.get("agent_dtype", "auto")
        agent_backend = config.get("agent_backend", "transformers")
        agent_model_file = config.get("agent_model_file", "")
        experiment_gpus = config.get("experiment_gpus", [])
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
                    click.echo("Agent restarted (loading in background)")
                else:
                    click.echo(f"Warning: Agent restart failed: {result.stdout[-200:]}")

        # Restart experiment runners
        gpu_ids = ",".join(str(g) for g in experiment_gpus) if experiment_gpus else str(agent_gpu or 0)
        experiments_script = scripts_dir / "start_experiments.sh"
        if experiments_script.exists():
            ssh.upload(str(experiments_script), "~/autoresearch/start_experiments.sh")
            result = ssh.run(
                f"bash ~/autoresearch/start_experiments.sh '{gpu_ids}'",
                timeout=60,
            )
            if result.returncode == 0:
                click.echo("Experiment runners resumed")
            else:
                raise click.ClickException(f"Resume failed: {result.stdout[-200:]}")
        else:
            raise click.ClickException(f"start_experiments.sh not found at {experiments_script}")
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(str(e))
