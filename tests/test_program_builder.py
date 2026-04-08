"""Tests for program.md builder."""

import tempfile
from pathlib import Path

from autoresearch.program.builder import (
    build_from_description,
    get_default_template,
    load_program,
)


def test_build_from_description():
    """Verify output has Goal, Constraints, Research Directions with provided content."""
    result = build_from_description(
        goal="Minimize val_bpb on FineWeb-Edu",
        constraints=["Only edit train.py", "No new dependencies"],
        directions=["Try larger batch sizes", "Adjust learning rate schedule"],
    )
    assert "## Goal" in result
    assert "Minimize val_bpb on FineWeb-Edu" in result
    assert "## Constraints" in result
    assert "- Only edit train.py" in result
    assert "- No new dependencies" in result
    assert "## Research Directions" in result
    assert "1. Try larger batch sizes" in result
    assert "2. Adjust learning rate schedule" in result


def test_build_from_existing_file():
    """Write a temp program.md, load it, verify content."""
    content = "# My Program\n\nThis is a test program.\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False
    ) as f:
        f.write(content)
        f.flush()
        loaded = load_program(f.name)
    assert loaded == content


def test_default_template_is_valid():
    """Verify template has all required sections."""
    template = get_default_template()
    required_sections = [
        "## Goal",
        "## Setup",
        "## Constraints",
        "## Experimentation",
        "## Research Directions",
        "## Output Format",
    ]
    for section in required_sections:
        assert section in template, f"Missing section: {section}"


def test_build_from_description_uses_template():
    """Verify built program has all standard sections from the template."""
    result = build_from_description(goal="Test goal")
    required_sections = [
        "## Goal",
        "## Setup",
        "## Constraints",
        "## Experimentation",
        "## Research Directions",
        "## Output Format",
    ]
    for section in required_sections:
        assert section in result, f"Missing section: {section}"
    # Verify defaults were applied
    assert "- Only edit train.py" in result
    assert "1. Explore hyperparameter changes" in result
