"""smollm2_eval 테스트."""

from __future__ import annotations

from pathlib import Path

from ml_playground import smollm2_eval


def test_fallback_eval_samples_limit() -> None:
    samples = smollm2_eval.fallback_eval_samples(3)
    assert len(samples) == 3


def test_read_eval_lines_fallback(tmp_path: Path) -> None:
    path = tmp_path / "missing.txt"
    lines = smollm2_eval.read_eval_lines(path, 2)
    assert len(lines) == 2


def test_read_eval_lines_from_file(tmp_path: Path) -> None:
    path = tmp_path / "eval.txt"
    path.write_text("\n안녕\n\n반가워\n", encoding="utf-8")
    lines = smollm2_eval.read_eval_lines(path, 10)
    assert lines == ["안녕", "반가워"]
