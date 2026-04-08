"""Tests for the experiment runner functions."""

import json
from pathlib import Path

from autoresearch.server.runner import (
    parse_results,
    read_checkpoint,
    should_keep,
    update_results_tsv,
    write_checkpoint,
)


def test_parse_training_results():
    """Parse log with val_bpb and peak_vram_mb lines."""
    log = (
        "step 100 | loss 3.456\n"
        "val_bpb: 2.1234\n"
        "peak_vram_mb: 4096.5\n"
        "done\n"
    )
    result = parse_results(log)
    assert result["val_bpb"] == 2.1234
    assert result["peak_vram_mb"] == 4096.5
    assert result["status"] == "ok"


def test_parse_training_results_crash():
    """Parse log without val_bpb (traceback / crash)."""
    log = (
        "step 50 | loss 3.9\n"
        "Traceback (most recent call last):\n"
        "  File 'train.py', line 99\n"
        "RuntimeError: CUDA out of memory\n"
    )
    result = parse_results(log)
    assert result["val_bpb"] == 0.0
    assert result["peak_vram_mb"] == 0.0
    assert result["status"] == "crash"


def test_update_results_tsv(tmp_path):
    """Write two rows and verify header + content."""
    tsv = tmp_path / "results.tsv"
    update_results_tsv(tsv, "abc1234", 2.1234, 4.1, "ok", "baseline run")
    update_results_tsv(tsv, "def5678", 2.0500, 4.2, "ok", "tweak lr")

    lines = tsv.read_text().splitlines()
    assert len(lines) == 3  # header + 2 rows
    assert lines[0] == "commit\tval_bpb\tmemory_gb\tstatus\tdescription"
    cols1 = lines[1].split("\t")
    assert cols1[0] == "abc1234"
    assert cols1[1] == "2.123400"
    assert cols1[2] == "4.1"
    assert cols1[3] == "ok"
    assert cols1[4] == "baseline run"
    cols2 = lines[2].split("\t")
    assert cols2[0] == "def5678"
    assert cols2[1] == "2.050000"


def test_checkpoint_write_and_read(tmp_path):
    """Roundtrip checkpoint JSON."""
    ckpt_path = tmp_path / "sub" / "checkpoint.json"
    write_checkpoint(
        ckpt_path,
        gpu_id=0,
        iteration=5,
        best_val_bpb=2.05,
        current_commit="abc1234",
        best_commit="abc1234",
        status="running",
    )
    data = read_checkpoint(ckpt_path)
    assert data["gpu_id"] == 0
    assert data["iteration"] == 5
    assert data["best_val_bpb"] == 2.05
    assert data["current_commit"] == "abc1234"
    assert data["best_commit"] == "abc1234"
    assert data["status"] == "running"
    assert "last_updated" in data


def test_should_keep_result():
    """Improved=True, worse=False, equal=False."""
    assert should_keep(2.05, 2.10) is True   # improved
    assert should_keep(2.15, 2.10) is False  # worse
    assert should_keep(2.10, 2.10) is False  # equal
