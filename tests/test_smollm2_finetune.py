"""smollm2_finetune 테스트."""

from __future__ import annotations

from typing import cast

from ml_playground import smollm2_finetune as finetune


def test_sanitize_text() -> None:
    text = "  안녕\n하세요  "
    assert finetune.sanitize_text(text) == "안녕 하세요"


def test_build_training_lines_instruction_output() -> None:
    rows = [
        {"instruction": "오늘 어때?", "output": "좋아."},
        {"instruction": "배고파", "output": "뭐 먹을래?"},
    ]
    lines = finetune.build_training_lines(cast(list[dict[str, object]], rows), 0)
    assert lines[0].startswith("질문:")
    assert "답변:" in lines[0]


def test_build_training_lines_text_fallback() -> None:
    rows = [
        {"text": "안녕하세요"},
        {"text": "반갑습니다"},
    ]
    lines = finetune.build_training_lines(cast(list[dict[str, object]], rows), 1)
    assert lines == ["안녕하세요"]


def test_read_lines(tmp_path) -> None:
    path = tmp_path / "train.txt"
    path.write_text("\n안녕\n\n반갑\n", encoding="utf-8")
    lines = finetune.read_lines(path)
    assert lines == ["안녕", "반갑"]


def test_prepare_dataset_writes_file(monkeypatch, tmp_path) -> None:
    fake_rows = [
        {"instruction": "안녕", "output": "반가워"},
        {"instruction": "점심 뭐 먹지?", "output": "아무거나"},
    ]

    def fake_load_hf_dataset(dataset_name: str, split_name: str) -> list[dict[str, object]]:
        _ = dataset_name
        _ = split_name
        return cast(list[dict[str, object]], fake_rows)

    monkeypatch.setattr(finetune, "load_hf_dataset", fake_load_hf_dataset)
    train_path = tmp_path / "out.txt"

    finetune.prepare_dataset(train_path, "dummy", "train", 0)
    content = train_path.read_text(encoding="utf-8")
    assert "질문:" in content
    assert "답변:" in content
