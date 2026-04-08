"""Monitor functions for checking autoresearch server and GPU status."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from autoresearch.ssh import SSHClient


def get_status(ssh: SSHClient) -> dict:
    """Check agent status and read GPU checkpoint states from the server.

    Args:
        ssh: Connected SSHClient instance.

    Returns:
        Dict with 'agent_running' (bool) and 'gpus' (list of checkpoint dicts).
    """
    # Check if agent screen session is running
    result = ssh.run("screen -list | grep autoresearch-agent", timeout=10)
    agent_running = "autoresearch-agent" in result.stdout

    # Read all checkpoint state.json files
    cmd = (
        'for dir in ~/autoresearch/checkpoints/gpu*/; do '
        '[ -d "$dir" ] || continue; '
        'for sf in "$dir"/state.json "$dir"/gpu*.json; do '
        '[ -f "$sf" ] || continue; '
        'cat "$sf"; '
        'echo "---SEPARATOR---"; '
        'break; done; done'
    )
    result = ssh.run(cmd, timeout=30)

    gpus: list[dict] = []
    if result.stdout.strip():
        chunks = result.stdout.split("---SEPARATOR---")
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                gpus.append(json.loads(chunk))
            except json.JSONDecodeError:
                continue

    # Sort by gpu_id if present
    gpus.sort(key=lambda g: g.get("gpu_id", 0))

    return {"agent_running": agent_running, "gpus": gpus}


def format_status(status: dict, server_host: str, agent_model: str) -> str:
    """Format status dict into a human-readable table.

    Args:
        status: Dict from get_status().
        server_host: Hostname of the server.
        agent_model: Model name used by the agent.

    Returns:
        Formatted multi-line string.
    """
    lines: list[str] = []

    # Header
    agent_state = "RUNNING" if status["agent_running"] else "STOPPED"
    lines.append(f"Server: {server_host}")
    lines.append(f"Agent:  {agent_state} (model: {agent_model})")
    lines.append("")

    gpus = status.get("gpus", [])
    if not gpus:
        lines.append("No GPU checkpoints found.")
        return "\n".join(lines)

    # Table header
    header = f"{'GPU':<6} {'Branch':<25} {'Iter':<8} {'Best val_bpb':<14} {'Status':<12} {'Last update'}"
    lines.append(header)
    lines.append("-" * len(header))

    now = datetime.now(timezone.utc)
    total_experiments = 0
    best_overall = float("inf")

    for gpu in gpus:
        gpu_id = gpu.get("gpu_id", "?")
        branch = gpu.get("branch", "?")
        iteration = gpu.get("iteration", "?")
        best_bpb = gpu.get("best_val_bpb", None)
        gpu_status = gpu.get("status", "?")
        last_updated = gpu.get("last_updated", "")

        # Calculate time since last update
        time_ago = "?"
        if last_updated:
            try:
                updated_dt = datetime.fromisoformat(last_updated)
                if updated_dt.tzinfo is None:
                    updated_dt = updated_dt.replace(tzinfo=timezone.utc)
                delta = now - updated_dt
                total_seconds = int(delta.total_seconds())
                if total_seconds < 60:
                    time_ago = f"{total_seconds}s ago"
                elif total_seconds < 3600:
                    time_ago = f"{total_seconds // 60}m ago"
                elif total_seconds < 86400:
                    time_ago = f"{total_seconds // 3600}h ago"
                else:
                    time_ago = f"{total_seconds // 86400}d ago"
            except (ValueError, TypeError):
                time_ago = last_updated

        bpb_str = f"{best_bpb:.4f}" if isinstance(best_bpb, (int, float)) else "—"
        iter_str = str(iteration)

        lines.append(
            f"{str(gpu_id):<6} {branch:<25} {iter_str:<8} {bpb_str:<14} {gpu_status:<12} {time_ago}"
        )

        # Accumulate totals
        if isinstance(iteration, (int, float)):
            total_experiments += int(iteration)
        if isinstance(best_bpb, (int, float)) and best_bpb < best_overall:
            best_overall = best_bpb

    lines.append("")
    best_str = f"{best_overall:.4f}" if best_overall < float("inf") else "—"
    lines.append(f"Total experiments: {total_experiments}  |  Best overall val_bpb: {best_str}")

    return "\n".join(lines)
