"""Core functions for the experiment runner loop."""

import argparse
import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from autoresearch.server.agent import (
    apply_diff,
    build_experiment_prompt,
    call_agent,
    extract_description,
    parse_code_edit,
)
from autoresearch.server.run_config import RunConfig

logger = logging.getLogger(__name__)


def parse_results(log_content: str) -> dict:
    """Parse val_bpb and peak_vram_mb from training log output."""
    val_match = re.search(r"^val_bpb:\s+([0-9.]+)", log_content, re.MULTILINE)
    vram_match = re.search(r"^peak_vram_mb:\s+([0-9.]+)", log_content, re.MULTILINE)

    if val_match:
        return {
            "val_bpb": float(val_match.group(1)),
            "peak_vram_mb": float(vram_match.group(1)) if vram_match else 0.0,
            "status": "ok",
        }
    return {"val_bpb": 0.0, "peak_vram_mb": 0.0, "status": "crash"}


def update_results_tsv(
    tsv_path: str | Path,
    commit: str,
    val_bpb: float,
    memory_gb: float,
    status: str,
    description: str,
) -> None:
    """Append a tab-separated row to results.tsv, creating with header if needed."""
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        tsv_path.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
    with open(tsv_path, "a") as f:
        f.write(f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n")


def write_checkpoint(
    path: str | Path,
    gpu_id: int,
    iteration: int,
    best_val_bpb: float,
    current_commit: str,
    best_commit: str,
    status: str,
) -> None:
    """Write JSON checkpoint with all fields plus last_updated timestamp."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "gpu_id": gpu_id,
        "iteration": iteration,
        "best_val_bpb": best_val_bpb,
        "current_commit": current_commit,
        "best_commit": best_commit,
        "status": status,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def read_checkpoint(path: str | Path) -> dict:
    """Read and return JSON checkpoint."""
    return json.loads(Path(path).read_text())


def should_keep(new_bpb: float, best_bpb: float) -> bool:
    """Return True if new_bpb < best_bpb (lower is better)."""
    return new_bpb < best_bpb


def _git(args: list[str], cwd: str) -> str:
    """Run a git command and return stripped stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def run_experiment_loop(
    gpu_id: int,
    checkpoint_dir: str,
    project_dir: str = os.path.expanduser("~/autoresearch"),
    agent_url: str = "http://localhost:8000/v1",
    single_gpu: bool = False,
) -> None:
    """Run the autonomous experiment loop for a single GPU.

    Reads run_config.yaml for experiment parameters (editable files,
    run command, metrics). Falls back to autoresearch defaults.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=f"[GPU {gpu_id}] %(asctime)s %(levelname)s %(message)s",
    )

    project = Path(project_dir)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"gpu{gpu_id}.json"

    # Load run configuration
    run_config_path = project / ".autoresearch" / "run_config.yaml"
    cfg = RunConfig.load(run_config_path)
    logger.info("Run config: command=%s, metric=%s (%s), files=%s",
                cfg.run_command, cfg.primary_metric, cfg.optimize_direction,
                cfg.editable_files)

    work_dir = project / cfg.work_dir
    program_md_path = project / "program.md"
    tsv_path = work_dir / f"results_gpu{gpu_id}.tsv"
    run_log_path = project / "run.log"

    branch_name = f"autoresearch/gpu{gpu_id}"

    # Load or initialize checkpoint state
    if ckpt_path.exists():
        ckpt = read_checkpoint(ckpt_path)
        iteration = ckpt["iteration"]
        best_metric = ckpt["best_val_bpb"]  # legacy field name, holds primary metric
        logger.info("Resumed from checkpoint: iteration=%d, best=%s", iteration, best_metric)
    else:
        iteration = 0
        best_metric = float("inf") if cfg.optimize_direction == "minimize" else float("-inf")
        logger.info("Starting fresh (no checkpoint found)")

    # Ensure work_dir is a git repo
    if not (work_dir / ".git").exists():
        _git(["init"], cwd=str(work_dir))
        _git(["add", "-A"], cwd=str(work_dir))
        _git(["commit", "-m", "initial", "--allow-empty"], cwd=str(work_dir))
        logger.info("Initialized git repo in %s", work_dir)

    # Create or checkout experiment branch
    try:
        _git(["checkout", branch_name], cwd=str(work_dir))
        logger.info("Checked out existing branch %s", branch_name)
    except subprocess.CalledProcessError:
        try:
            _git(["checkout", "-b", branch_name], cwd=str(work_dir))
            logger.info("Created new branch %s", branch_name)
        except subprocess.CalledProcessError:
            # Branch exists but checkout failed — force reset
            _git(["branch", "-D", branch_name], cwd=str(work_dir))
            _git(["checkout", "-b", branch_name], cwd=str(work_dir))
            logger.info("Recreated branch %s", branch_name)

    # Main experiment loop
    while True:
        iteration += 1
        logger.info("=== Iteration %d ===", iteration)

        try:
            # Read current files — send full content for accurate SEARCH/REPLACE
            logger.info("Reading program.md and editable files...")
            program_md = program_md_path.read_text() if program_md_path.exists() else ""
            editable_content = ""
            for fpath in cfg.editable_files:
                full = work_dir / fpath
                if full.exists():
                    content = full.read_text()
                    editable_content += f"\n### {fpath}\n```python\n{content}\n```\n"
            if tsv_path.exists():
                tsv_lines = tsv_path.read_text().strip().split("\n")
                # Keep header + last 10 results to avoid huge prompts
                if len(tsv_lines) > 11:
                    results_history = tsv_lines[0] + "\n" + "\n".join(tsv_lines[-10:])
                else:
                    results_history = "\n".join(tsv_lines)
            else:
                results_history = "No results yet."
            logger.info("Read %d chars of code context, building prompt...", len(editable_content))

            # Build prompt and call agent
            prompt = build_experiment_prompt(program_md, editable_content, results_history, iteration)
            logger.info("Prompt built (%d chars). Calling agent at %s ...", len(prompt), agent_url)
            try:
                response = call_agent(prompt, base_url=agent_url)
                logger.info("Agent responded (%d chars)", len(response) if response else 0)
            except Exception:
                logger.exception("Agent call failed, sleeping 30s before retry")
                time.sleep(30)
                iteration -= 1  # Don't count failed agent calls
                continue

            # Parse the code edit from the response
            code_edit = parse_code_edit(response)
            if code_edit is None:
                logger.warning("Could not parse code edit. Full response:\n%s", response)
                iteration -= 1
                continue

            description = extract_description(response)
            if not description:
                description = f"Iteration {iteration} change"

            # Apply the edit to editable files
            has_diff = "<<<" in code_edit and ("===" in code_edit or "=======" in code_edit)
            if has_diff:
                # Apply diff to all editable files
                applied = False
                for fpath in cfg.editable_files:
                    full = work_dir / fpath
                    if full.exists():
                        original = full.read_text()
                        modified = apply_diff(original, code_edit)
                        if modified != original:
                            full.write_text(modified)
                            logger.info("Applied diff to %s", fpath)
                            applied = True
                if not applied:
                    # Log the search text for debugging
                    import re as _re
                    search_blocks = _re.findall(r"<<<(?:SEARCH)?\n?(.*?)\n===", code_edit, _re.DOTALL)
                    for i, sb in enumerate(search_blocks):
                        logger.warning("SEARCH block %d (first 200 chars): %s", i, sb.strip()[:200])
                    logger.warning("No match found in any file. Agent response:\n%s", code_edit[:500])
                    iteration -= 1
                    continue
            else:
                # No SEARCH/REPLACE block — model didn't follow format, retry
                logger.warning("No SEARCH/REPLACE block in response, skipping. Got: %s", code_edit[:200])
                iteration -= 1
                continue

            # Git add all editable files and commit
            for fpath in cfg.editable_files:
                full = work_dir / fpath
                if full.exists():
                    _git(["add", fpath], cwd=str(work_dir))
            _git(["commit", "-m", f"[gpu{gpu_id}] iter {iteration}: {description[:120]}",
                  "--allow-empty"], cwd=str(work_dir))
            current_commit = _git(["rev-parse", "HEAD"], cwd=str(work_dir))[:8]
            logger.info("Committed %s: %s", current_commit, description[:80])

            # Run experiment
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logger.info("Running: %s", cfg.run_command)
            try:
                run_result = subprocess.run(
                    cfg.run_command,
                    shell=True,
                    cwd=str(work_dir),
                    capture_output=True,
                    text=True,
                    timeout=cfg.run_timeout,
                    env=env,
                )
                log_content = run_result.stdout + "\n" + run_result.stderr
            except subprocess.TimeoutExpired:
                log_content = f"TIMEOUT: exceeded {cfg.run_timeout}s limit"
                logger.warning("Run timed out")

            # Write run.log
            run_log_path.write_text(log_content)

            # Parse results using configured metrics
            results = cfg.parse_output(log_content)
            primary_value = results.get(cfg.primary_metric, 0.0)
            status = results["status"]

            if status == "crash":
                # Include last error line in description so model can learn
                error_lines = [l.strip() for l in log_content.split("\n") if l.strip() and "Error" in l]
                crash_desc = error_lines[-1][:150] if error_lines else "unknown error"
                logger.warning("Run crashed: %s", crash_desc)
                update_results_tsv(tsv_path, current_commit, primary_value, 0.0, "crash",
                                   f"{description[:80]} | ERROR: {crash_desc}")
                _git(["reset", "--hard", "HEAD~1"], cwd=str(work_dir))
            elif cfg.is_improvement(primary_value, best_metric):
                best_metric = primary_value
                logger.info("IMPROVED: %s=%.6f (new best)", cfg.primary_metric, primary_value)
                update_results_tsv(tsv_path, current_commit, primary_value, 0.0, "keep", description)
            else:
                logger.info("DISCARDED: %s=%.6f (best=%.6f), reverting",
                            cfg.primary_metric, primary_value, best_metric)
                update_results_tsv(tsv_path, current_commit, primary_value, 0.0, "discard", description)
                _git(["reset", "--hard", "HEAD~1"], cwd=str(work_dir))

            # Write checkpoint
            current_commit_full = _git(["rev-parse", "HEAD"], cwd=str(work_dir))
            best_commit = current_commit_full[:8] if best_metric != float("inf") and best_metric != float("-inf") else ""
            write_checkpoint(
                ckpt_path,
                gpu_id=gpu_id,
                iteration=iteration,
                best_val_bpb=best_metric,  # legacy field name, holds primary metric
                current_commit=current_commit_full[:8],
                best_commit=best_commit,
                status="idle",
            )

        except Exception:
            logger.exception("Unexpected error in iteration %d", iteration)
            time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the autonomous experiment loop")
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--project-dir", default=os.path.expanduser("~/autoresearch"))
    parser.add_argument("--agent-url", default="http://localhost:8000/v1")
    parser.add_argument("--single-gpu", action="store_true")
    args = parser.parse_args()
    run_experiment_loop(**vars(args))
