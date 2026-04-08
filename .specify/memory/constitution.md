<!--
Sync Impact Report
==================
Version change: 0.0.0 → 1.0.0 (initial ratification)
Added principles:
  - I. Correctness
  - II. Compound Engineering
  - III. User Friendliness
  - IV. Robustness
  - V. Remote-Friendly
Added sections:
  - Technology Stack
  - Development Workflow
Removed sections: none (initial version)
Templates requiring updates:
  ✅ .specify/templates/plan-template.md (no changes needed)
  ✅ .specify/templates/spec-template.md (no changes needed)
  ✅ .specify/templates/tasks-template.md (no changes needed)
Follow-up TODOs: none
-->

# Autoresearch Constitution

## Core Principles

### I. Correctness

Never assume — verify by actually running code. Every claim of
correctness MUST be backed by executed tests or a production example.

- All new features MUST be validated against a real-world use case
  before being marked complete.
- Tests MUST call real functions, real APIs, real servers. Mock
  objects are forbidden.
- External libraries and resources MUST be verified via actual
  invocation, not assumed to work based on documentation alone.
- If a test passes but the feature does not work in practice,
  the test is wrong — fix the test, not the definition of "works."

### II. Compound Engineering

When something does not work as expected: fix it, note it, and
improve the system so the same failure cannot recur.

- Every debugging session MUST produce a durable artifact: a
  fixed script, an improved error message, an updated note, or a
  new test that catches the regression.
- Session notes MUST be updated continuously so progress survives
  context loss.
- Commit frequently with descriptive messages to maintain a
  comprehensive change history.
- Before closing a task, verify that the fix is general — not just
  a patch for one instance of the problem.

### III. User Friendliness

All user-facing functions MUST be intuitive, simple,
well-documented, and behave as advertised.

- Fail fast: if an operation will not succeed, detect and report
  the reason before doing expensive work.
- Error messages MUST be actionable — state what went wrong, why,
  and what the user can do about it.
- CLI commands MUST have `--help` text. Python functions MUST have
  docstrings describing parameters and return values.
- Default values MUST be sensible. Optional parameters MUST NOT be
  required for the common case.

### IV. Robustness

All install and setup scripts MUST be idempotent and
multi-platform.

- Supported platforms: macOS, Ubuntu/Linux, Windows (WSL).
- Running a setup script twice MUST produce the same result as
  running it once — no errors, no duplicated state.
- Scripts MUST detect the current OS and architecture and adapt
  accordingly.
- Every step MUST check its preconditions and skip if already
  satisfied, printing a clear "already done" message.

### V. Remote-Friendly

This project MUST support GPU clusters. All setup MUST be
performed via remote-running scripts with zero manual
intervention by the user.

- All required packages MUST be installed, configured, and
  validated automatically if not already present.
- Python functionality MUST run inside conda environments (or
  equivalent) — never depend on system Python or pre-existing
  system library versions.
- Conda itself MUST be installed automatically (via Miniforge)
  if not present on the target machine.
- SSH operations MUST use deterministic scripts that read
  credentials from a config file — no interactive prompts during
  automated workflows.
- Screen sessions MUST be used for long-running server processes
  so the user can disconnect without interrupting work.

## Technology Stack

- **Language:** Python 3.10+ (3.11 preferred in conda envs)
- **Package management:** uv (local), conda + pip (remote servers)
- **CLI framework:** Click
- **Configuration:** YAML (credentials, project config)
- **Remote execution:** SSH via sshpass + subprocess
- **Agent serving:** vLLM with OpenAI-compatible API
- **Session management:** GNU screen
- **Build system:** Hatchling (src layout)
- **Testing:** pytest with real-world tests (no mocks)

## Development Workflow

1. **Write tests first** that exercise real functionality.
2. **Implement** the minimum code to make tests pass.
3. **Run on a production example** before marking complete.
4. **Commit** with a descriptive message after each logical unit.
5. **Note** any issues encountered and how they were resolved.
6. **Push** only after all tests pass (including linters and docs).
7. **Review** — check for passwords, keys, and personal info
   before committing.

## Governance

This constitution supersedes ad-hoc practices. All code changes
MUST comply with these principles. Amendments require:

1. A written proposal describing the change and rationale.
2. Update to this document with version increment.
3. Propagation check across dependent templates.

Versioning follows semantic versioning:
- MAJOR: principle removal or backward-incompatible redefinition.
- MINOR: new principle or materially expanded guidance.
- PATCH: clarification, wording, or typo fix.

Runtime development guidance lives in `CLAUDE.md` at the project
root.

**Version**: 1.0.0 | **Ratified**: 2026-04-06 | **Last Amended**: 2026-04-06
