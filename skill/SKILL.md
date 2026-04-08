---
name: autoresearch
description: Deploy and manage autonomous AI research experiments on remote GPU servers. Interactive setup for credentials, GPU configuration, model selection, and research program creation.
---

# Autoresearch Skill

Deploy and manage autonomous AI research experiments on a remote GPU server.

## Commands

- `/autoresearch` — Full interactive setup and deploy
- `/autoresearch status` — Check experiment progress
- `/autoresearch sync` — Pull results locally
- `/autoresearch stop [all|agent|experiments|gpu<N>]` — Stop experiments
- `/autoresearch resume` — Resume from checkpoints

## Interactive Setup Flow

When the user runs `/autoresearch` (no subcommand), walk through these steps IN ORDER:

### Step 1: Credentials

Check for `credentials.yaml` in the project root.

If it exists, load and confirm the server details with the user.

If missing, ask the user for:
1. Server hostname/IP
2. SSH username
3. SSH password
4. SSH port (default: 22)

Save to `credentials.yaml` (this file is gitignored).
Test the connection:
```bash
bash scripts/ssh.sh echo CONNECTION_OK
```

### Step 2: Server Discovery

SSH in and detect the environment:
```bash
bash scripts/ssh.sh 'nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader'
```

Show the user their GPU list and ask:
- Which GPU for the agent model? (default: GPU 0)
- Which GPUs for experiments? (default: all others)
- If only 1 GPU: explain single-GPU mode (load/unload cycling, slower but works)

### Step 3: Agent Model Selection

Ask the user which HuggingFace model to use as the research agent.
- Default: `google/gemma-4-27b-it`
- Show estimated VRAM usage for the model
- User can specify any HF model ID

Save configuration to `.autoresearch/config.yaml`:
```yaml
agent_model: google/gemma-4-27b-it
agent_gpu: 0
experiment_gpus: [1, 2, 3, 4, 5, 6, 7]
```

### Step 4: Research Program

Ask the user how they want to define their research program:

**(A) Existing file** — "Point me to your program.md"
- Read the file, summarize it, confirm with user

**(B) Repository** — "Give me a repo URL"
- Clone/fetch the repo
- Analyze training code and any existing instructions
- Draft a program.md adapted for the autoresearch loop
- Review with user

**(C) Description** — "Describe what you want to research"
- User provides a text description
- Draft a program.md with constraints, goals, evaluation criteria
- Review with user

**(D) Idea** — "Tell me your idea and I'll help refine it"
- Ask 3-5 questions to refine scope and goals
- Draft a program.md
- Review with user

All paths produce a `program.md` in the project root.

### Step 5: Deploy

Run the full deployment:
```bash
autoresearch deploy
```

Show initial status when done.

## Subcommands

### `/autoresearch status`
```bash
autoresearch status
```
Shows GPU status table with iteration counts, best val_bpb, and timing.

### `/autoresearch sync`
```bash
autoresearch sync
```
Pulls checkpoints and results locally via rsync.

### `/autoresearch stop [target]`
```bash
autoresearch stop [all|agent|experiments|gpu<N>]
```
Stops specified sessions on the server.

### `/autoresearch resume`
```bash
autoresearch resume
```
Restarts experiments from last checkpoint.
