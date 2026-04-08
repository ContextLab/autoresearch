"""Agent harness: prompt construction, response parsing, and LLM interaction."""

from __future__ import annotations

import re

import openai


def build_experiment_prompt(
    program_md: str,
    code_context: str,
    results_history: str,
    iteration: int = 0,
) -> str:
    """Construct the full prompt sent to the agent LLM.

    Args:
        program_md: Research program instructions.
        code_context: Current code (may be multiple files, each with a ### header).
        results_history: Past experiment results (TSV or text).
        iteration: Current iteration number.
    """
    return (
        "You are an autonomous code modification agent. You MUST output a code change.\n\n"
        f"## Iteration {iteration}\n\n"
        f"## Goal\n\n{program_md}\n\n"
        f"## Current Code\n\n{code_context}\n\n"
        f"## Past Results\n\n{results_history}\n\n"
        "## INSTRUCTIONS\n\n"
        "Pick ONE file and make ONE small change using SEARCH/REPLACE format:\n\n"
        "File: <filename>\n"
        "Change: <one sentence>\n\n"
        "<<<SEARCH\n"
        "<exact lines from the original code to find>\n"
        "===\n"
        "<replacement lines>\n"
        ">>>REPLACE\n\n"
        "IMPORTANT: Use <<<SEARCH/===/ >>>REPLACE blocks. "
        "The SEARCH section must match existing code exactly. "
        "Keep changes small and focused.\n"
    )


def parse_code_edit(response: str) -> str | None:
    """Parse agent response to extract code edits.

    Tries multiple formats in order:
    1. SEARCH/REPLACE blocks
    2. ```python fenced code blocks
    3. Any ``` fenced code blocks
    Returns None if unparseable.
    """
    # Try SEARCH/REPLACE format first (preferred for multi-file edits)
    # Accept both <<<SEARCH/>>>REPLACE and bare <<</===/>>>
    if ("<<<" in response and ("===" in response or "=======" in response)):
        return response

    # Try fenced python code blocks — return the longest match
    matches = re.findall(r"```python\n(.*?)```", response, re.DOTALL)
    if matches:
        return max(matches, key=len)

    # Try any fenced code block (model may omit language tag)
    matches = re.findall(r"```\w*\n(.*?)```", response, re.DOTALL)
    if matches:
        return max(matches, key=len)

    # Try bare ``` blocks
    matches = re.findall(r"```(.*?)```", response, re.DOTALL)
    if matches:
        longest = max(matches, key=len).strip()
        if len(longest) > 20:
            return longest

    # Handle truncated response: unclosed code block (max_tokens hit)
    unclosed = re.search(r"```(?:python)?\n(.+)", response, re.DOTALL)
    if unclosed:
        code = unclosed.group(1).strip()
        if len(code) > 20:
            return code

    return None


def apply_diff(original: str, diff_text: str) -> str:
    """Apply SEARCH/REPLACE blocks to original content.

    Accepts both <<<SEARCH/>>>REPLACE and bare <<</>>>.
    """
    # Try full format first, then bare format
    patterns = [
        re.compile(r"<<<SEARCH\n(.*?)\n={3,}\n(.*?)\n>>>REPLACE", re.DOTALL),
        re.compile(r"<<<\n?(.*?)\n={3,}\n(.*?)\n>>>", re.DOTALL),
        # Handle truncated response (no closing >>>)
        re.compile(r"<<<(?:SEARCH)?\n(.*?)\n={3,}\n(.*)", re.DOTALL),
    ]
    result = original
    for pattern in patterns:
        matches = list(pattern.finditer(diff_text))
        if matches:
            for match in matches:
                old = match.group(1).strip()
                new = match.group(2).strip()
                if old in result:
                    result = result.replace(old, new, 1)
                else:
                    # Fuzzy match: try matching first meaningful line
                    old_lines = [l for l in old.split("\n") if l.strip()]
                    if old_lines:
                        first_line = old_lines[0].strip()
                        if first_line in result and len(first_line) > 10:
                            # Find the block starting at first_line
                            idx = result.index(first_line)
                            # Replace from first_line through len(old) chars
                            end_idx = idx + len(old)
                            if end_idx > len(result):
                                end_idx = len(result)
                            result = result[:idx] + new + result[end_idx:]
            break
    return result


def extract_description(response: str) -> str:
    """Extract a concise description from the agent response.

    Takes text before the first ``` block, returns the first meaningful line
    (>10 chars, not starting with #), truncated to 200 chars.
    """
    before_code = response.split("```")[0]
    for line in before_code.splitlines():
        line = line.strip()
        if len(line) > 10 and not line.startswith("#"):
            return line[:200]
    return ""


def call_agent(
    prompt: str,
    base_url: str = "http://localhost:8000/v1",
    model: str = "default",
    max_tokens: int = 2048,
    temperature: float = 0.7,
    timeout: int = 600,
) -> str:
    """Call the model server via OpenAI-compatible API.

    Returns the response text.
    """
    client = openai.OpenAI(
        base_url=base_url,
        api_key="not-needed",
        timeout=timeout,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content
