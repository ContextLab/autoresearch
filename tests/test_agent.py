"""Tests for the agent harness (no server required)."""

from autoresearch.server.agent import (
    apply_diff,
    build_experiment_prompt,
    extract_description,
    parse_code_edit,
)


def test_build_experiment_prompt():
    prompt = build_experiment_prompt(
        program_md="Do research",
        code_context="### train.py\n```python\nimport torch\n```",
        results_history="baseline: 1.05",
        iteration=3,
    )
    assert "Do research" in prompt
    assert "import torch" in prompt
    assert "baseline: 1.05" in prompt
    assert "Iteration 3" in prompt


def test_parse_code_edit_fenced():
    response = (
        "Here is the updated code:\n\n"
        "```python\nimport torch\nprint('hello')\n```\n\n"
        "This should improve results."
    )
    result = parse_code_edit(response)
    assert result == "import torch\nprint('hello')\n"


def test_parse_code_edit_diff_format():
    response = (
        "I will change the learning rate.\n\n"
        "<<<SEARCH\nLR = 0.001\n===\nLR = 0.0005\n>>>REPLACE\n"
    )
    result = parse_code_edit(response)
    # Should return full response since it contains SEARCH/REPLACE
    assert result is not None
    assert "<<<SEARCH" in result
    assert ">>>REPLACE" in result


def test_apply_diff_to_file():
    original = "DEPTH = 12\nLR = 0.001\nBATCH = 64\n"
    diff_text = "<<<SEARCH\nLR = 0.001\n===\nLR = 0.0005\n>>>REPLACE"
    result = apply_diff(original, diff_text)
    assert "LR = 0.0005" in result
    assert "LR = 0.001" not in result
    # Unchanged parts preserved
    assert "DEPTH = 12" in result
    assert "BATCH = 64" in result


def test_extract_experiment_description():
    response = (
        "# Header\n"
        "Short\n"
        "I increased the learning rate from 0.001 to 0.002 to speed up convergence.\n\n"
        "```python\nLR = 0.002\n```"
    )
    desc = extract_description(response)
    assert len(desc) <= 200
    assert "learning rate" in desc

    # Test truncation with a very long line
    long_response = "A" * 300 + "\n```python\ncode\n```"
    desc2 = extract_description(long_response)
    assert len(desc2) == 200
