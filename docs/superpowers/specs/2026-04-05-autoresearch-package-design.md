# Autoresearch Package Design

## Overview

Transform the autoresearch project into a deployable Python package that runs autonomous AI research experiments on a remote GPU server. A local HuggingFace model (default: Gemma 4) acts as the fast experiment agent on the server, while Claude Code serves as the research architect — crafting research programs, deploying experiments, and monitoring progress.

## Architecture

### Hybrid Agent Model

- **Claude Code (local):** Research strategist. Crafts `program.md`, deploys to server, monitors progress via skills.
- **HF model on GPU server (remote):** Fast experiment agent. Runs the modify-train-evaluate-decide loop autonomously using vLLM.

### GPU Allocation (Adaptive)

**Multi-GPU (2+ GPUs):**
```
GPU 0:       vLLM server (agent LLM, always loaded)
GPU 1-N:     Experiment runners (each in its own screen session)
```

**Single-GPU:**
```
GPU 0:       Time-shared — load agent → generate → unload → train → repeat
```

- Mode auto-detected during server discovery, user can override
- User can configure which GPU is the agent and which run experiments
- Default agent model: `google/gemma-4-27b-it`, user-configurable

### Server Requirements

- Any SSH-accessible server with 1+ NVIDIA GPUs
- Supported OS: Ubuntu/Linux, macOS, Windows (WSL)
- No assumptions about pre-installed software — setup scripts detect OS and install conda, uv, and all dependencies as needed
- All setup scripts are idempotent (safe to re-run)
- Connection: SSH with password auth via sshpass

### Single-GPU Mode

When only 1 GPU is available, the system uses the load-unload pattern:
- Load agent model → generate code edit → unload agent model → free VRAM → run training → repeat
- Slower (~30-60s overhead per cycle for model loading) but functional
- Automatically detected during server discovery

### Multi-GPU Mode

When 2+ GPUs are available:
- 1 GPU dedicated to vLLM (agent stays loaded, fast inference)
- Remaining GPUs run parallel experiment loops
- User can override GPU assignments

## Package Structure

```
autoresearch/
  pyproject.toml
  credentials.yaml          # .gitignored, contains host/username/password
  src/
    autoresearch/
      __init__.py
      cli.py                # CLI entry points
      config.py             # Credentials + project config loading
      ssh.py                # SSH connection, file transfer, screen management
      deploy.py             # Package deployment to remote server
      server/
        agent.py            # Agent harness: prompts vLLM, parses edits, applies diffs
        runner.py           # Experiment runner: train -> evaluate -> keep/discard loop
        vllm_server.py      # vLLM lifecycle management
        monitor.py          # Status reporting, results aggregation
      program/
        builder.py          # program.md construction from various inputs
        templates/          # Built-in program.md templates
  scripts/
    ssh.sh                  # SSH wrapper using sshpass + credentials.yaml
    setup_env.sh            # Creates conda env, installs deps, prepares data
    start_agent.sh          # Starts vLLM on agent GPU in screen session
    start_experiments.sh    # Starts experiment runners in screen sessions
    status.sh               # Reads checkpoints, reports status
    sync.sh                 # rsync checkpoints + results locally
    stop.sh                 # Kills screen sessions, stops vLLM
  skill/
    SKILL.md                # Claude Code skill definition
  tests/
```

## Design Principle: Deterministic Scripts

Anything that happens the same way every time is implemented as a tested shell script, not left to Claude to improvise:

| Operation | Script | What it does |
|-|-|-|
| SSH connection | `scripts/ssh.sh` | sshpass wrapper, reads credentials.yaml |
| Server env setup | `scripts/setup_env.sh` | conda env creation, pip install, data prep |
| vLLM launch | `scripts/start_agent.sh` | Starts vLLM on specified GPU in screen |
| Experiment launch | `scripts/start_experiments.sh` | Starts runners in per-GPU screen sessions |
| Status check | `scripts/status.sh` | Reads checkpoint state.json files |
| Sync results | `scripts/sync.sh` | rsync checkpoints + results locally |
| Teardown | `scripts/stop.sh` | Kills screens, stops vLLM |

Claude Code skills are thin wrappers that call these scripts and format output.

## Interactive Setup Flow (Claude Code Skill)

The `/autoresearch` skill walks through deterministic steps:

### Step 1: Credentials

- Check for `credentials.yaml` in project root
- If missing: prompt for host, username, password, SSH port
- Test connection via `scripts/ssh.sh`
- Save to `credentials.yaml` (gitignored)
- Host, username, and password are all treated as secrets

### Step 2: Server Discovery

- SSH in, run `nvidia-smi` to detect GPUs
- Show GPU list with VRAM
- User picks agent GPU and experiment GPUs (default: GPU 0 = agent, rest = experiments)

### Step 3: Agent Model Selection

- Default: `google/gemma-4-27b-it`
- User can specify any HF model
- Validate it fits on the agent GPU's VRAM
- Confirm choice

### Step 4: Research Program (program.md)

User chooses one of four input paths:

**(A) Existing file** — User provides path to a `program.md`. Claude reads, summarizes, user confirms.

**(B) Repository** — User provides a repo URL. Claude analyzes the training code and drafts a `program.md` adapted for the autoresearch loop.

**(C) Description** — User provides a text description. Claude drafts a `program.md` with constraints, goals, evaluation criteria.

**(D) Idea iteration** — User provides a vague idea. Claude asks 3-5 questions to refine, then drafts `program.md`.

All paths produce a `program.md` following this standard structure:

```markdown
# Research Program

## Goal
<what we're optimizing for>

## Setup
<data, tokenizer, base model, evaluation metric>

## Constraints
<what's off-limits, time budget, VRAM limits>

## Experimentation
<what the agent can modify, keep/discard criteria>

## Research Directions
<suggested areas to explore, ordered by priority>

## Output Format
<how to log results>
```

### Step 5: Deploy & Launch

- rsync package + program.md + template files to server
- Run `scripts/setup_env.sh` (idempotent)
- Run `scripts/start_agent.sh`
- Run `scripts/start_experiments.sh`
- Confirm all running, show initial status

## Server-Side Components

### vLLM Agent Server

- Runs in screen session `autoresearch-agent` on GPU 0
- Serves OpenAI-compatible API at `localhost:8000`
- Default model: `google/gemma-4-27b-it` (configurable)
- Managed by `src/autoresearch/server/vllm_server.py`

### Experiment Runner

Each experiment GPU runs an independent experiment loop in screen session `autoresearch-gpu{N}`. In single-GPU mode, one runner operates on GPU 0 with load-unload agent cycling:

```
loop:
  1. Read program.md for current research direction
  2. Read current train.py + results.tsv for context
  3. POST to localhost:8000 — ask agent model for next experiment
  4. Parse response, apply diff to train.py
  5. git commit on per-GPU branch (autoresearch/gpu{N})
  6. CUDA_VISIBLE_DEVICES={N} uv run train.py > run.log 2>&1
  7. Parse results (val_bpb, peak_vram_mb)
  8. If improved: keep commit, update results.tsv
     If worse: git reset to previous best
  9. Write checkpoint to ~/autoresearch/checkpoints/gpu{N}/state.json
  10. goto loop
```

### Per-GPU Isolation

- Each GPU operates on its own git branch (`autoresearch/gpu{N}`)
- Each has its own `results.tsv`
- No merge conflicts between concurrent experiments
- Claude can later review all branches and cherry-pick best findings

### Checkpoint Format

```json
{
  "gpu_id": 1,
  "iteration": 42,
  "best_val_bpb": 0.9821,
  "current_commit": "a1b2c3d",
  "best_commit": "f9e8d7c",
  "status": "running",
  "last_updated": "2026-04-05T22:30:00Z"
}
```

## Status, Sync & Lifecycle

### Status (`/autoresearch status`)

Calls `scripts/status.sh`, which SSHes in and reads checkpoint state.json files. Output:

```
Server: gpu-server.example.com
Agent:  google/gemma-4-27b-it on GPU 0 [running]

GPU | Branch            | Iter | Best val_bpb | Status  | Last update
----|-------------------|------|--------------|---------|------------
1   | autoresearch/gpu1 | 42   | 0.9821       | running | 2m ago
2   | autoresearch/gpu2 | 38   | 0.9847       | running | 4m ago
...

Total experiments: 272 | Best overall: 0.9789 (GPU 7, commit e3f4a5b)
```

### Sync (`/autoresearch sync`)

- rsync all `checkpoints/`, `results.tsv`, and `run.log` files locally into `results/` (gitignored)
- Optionally `git fetch` remote branches for local code inspection

### Stop (`/autoresearch stop`)

- Options: stop one GPU, stop all experiments, stop everything (including vLLM)
- Kills relevant screen sessions
- Final sync before stopping

### Resume (`/autoresearch resume`)

- Reads checkpoint state, restarts screen sessions from last checkpoint
- vLLM health check — restart if needed

## Credentials

File: `credentials.yaml` (project root, gitignored)

```yaml
servers:
  default:
    host: gpu-server.example.com
    port: 22
    username: myuser
    password: <secret>
```

All fields (host, username, password) are treated as secrets. The file is listed in `.gitignore`.

## Server Environment Setup

`scripts/setup_env.sh` is fully idempotent and handles any server state:

1. **Detect OS** (Linux/macOS/Windows-WSL) and architecture (x86_64/arm64)
2. **Install conda** if not present (Miniforge — works on all platforms, no license issues)
3. **Create conda env** `autoresearch` with Python 3.11 (skip if exists)
4. **Install uv** in the conda env (if not present)
5. **Install vLLM, PyTorch, transformers** via pip in the conda env
6. **rsync the autoresearch package** from local to server
7. **Run `uv sync`** for training dependencies
8. **Run `uv run prepare.py`** for data/tokenizer prep (if not cached)

All subsequent scripts activate the conda env before running any commands. The script checks each step's preconditions and skips already-completed steps, so re-running is fast and safe.

## Expected Performance

**Multi-GPU (e.g., 8 GPUs):**
- 1 agent GPU + 7 experiment GPUs
- ~5 minutes per experiment + ~30s agent thinking overhead
- ~12 experiments/hour/GPU = ~84 experiments/hour total
- ~580 experiments in an 8-hour overnight run

**Single-GPU:**
- ~5 minutes per experiment + ~60s agent load/unload + ~30s thinking
- ~9 experiments/hour
- ~72 experiments in an 8-hour overnight run
