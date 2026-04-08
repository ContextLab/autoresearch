"""Run configuration for the experiment loop.

Defines what files to edit, how to run experiments, and how to parse results.
Loaded from .autoresearch/run_config.yaml on the server.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RunConfig:
    """Configuration for the experiment runner."""

    # Files the agent can modify (relative to project_dir)
    editable_files: list[str] = field(default_factory=lambda: ["train.py"])

    # Command to run an experiment (executed in project_dir)
    run_command: str = "uv run train.py"

    # Timeout for each experiment run (seconds)
    run_timeout: int = 600

    # Metric extraction: regex patterns applied to run output
    # Each pattern should have one capture group for the value
    metrics: dict[str, str] = field(default_factory=lambda: {
        "val_bpb": r"^val_bpb:\s+([0-9.]+)",
        "peak_vram_mb": r"^peak_vram_mb:\s+([0-9.]+)",
    })

    # Which metric to optimize
    primary_metric: str = "val_bpb"

    # Direction: "minimize" or "maximize"
    optimize_direction: str = "minimize"

    # Working directory for experiments (relative to project_dir, or "." for project_dir itself)
    work_dir: str = "."

    def parse_output(self, log_content: str) -> dict[str, Any]:
        """Parse experiment output using configured metric patterns.

        Returns dict with metric values and "status" ("ok" or "crash").
        """
        result: dict[str, Any] = {"status": "crash"}
        found_primary = False

        for name, pattern in self.metrics.items():
            match = re.search(pattern, log_content, re.MULTILINE)
            if match:
                result[name] = float(match.group(1))
                if name == self.primary_metric:
                    found_primary = True
            else:
                result[name] = 0.0

        if found_primary:
            result["status"] = "ok"

        return result

    def is_improvement(self, new_value: float, best_value: float) -> bool:
        """Check if new metric value is an improvement over best."""
        if self.optimize_direction == "minimize":
            return new_value < best_value
        else:
            return new_value > best_value

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        """Load run config from YAML file."""
        p = Path(path)
        if not p.exists():
            return cls()  # defaults

        with open(p) as f:
            data = yaml.safe_load(f) or {}

        defaults = cls()
        return cls(
            editable_files=data.get("editable_files", defaults.editable_files),
            run_command=data.get("run_command", defaults.run_command),
            run_timeout=data.get("run_timeout", defaults.run_timeout),
            metrics=data.get("metrics", defaults.metrics),
            primary_metric=data.get("primary_metric", defaults.primary_metric),
            optimize_direction=data.get("optimize_direction", defaults.optimize_direction),
            work_dir=data.get("work_dir", defaults.work_dir),
        )

    def save(self, path: str | Path) -> None:
        """Save run config to YAML file."""
        data = {
            "editable_files": self.editable_files,
            "run_command": self.run_command,
            "run_timeout": self.run_timeout,
            "metrics": self.metrics,
            "primary_metric": self.primary_metric,
            "optimize_direction": self.optimize_direction,
            "work_dir": self.work_dir,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
