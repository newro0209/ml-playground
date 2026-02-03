"""smollm2_finetune 테스트."""

from __future__ import annotations

from typing import cast

import pytest
import torch

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
    assert "입력:" not in lines[0]


def test_build_training_lines_text_fallback() -> None:
    rows = [
        {"text": "안녕하세요"},
        {"text": "반갑습니다"},
    ]
    lines = finetune.build_training_lines(cast(list[dict[str, object]], rows), 1)
    assert lines == ["안녕하세요"]


def test_build_training_lines_instruction_response() -> None:
    rows = [
        {"instruction": "자기소개 해줘", "response": "안녕하세요. 도움을 드릴게요."},
        {"instruction": "오늘 날씨 어때?", "response": "맑다고 해요."},
    ]
    lines = finetune.build_training_lines(cast(list[dict[str, object]], rows), 0)
    assert lines[0].startswith("질문:")
    assert "답변:" in lines[0]


def test_build_training_lines_instruction_with_input() -> None:
    rows = [
        {
            "instruction": "요약해줘",
            "input": "이 문장은 요약 대상입니다.",
            "response": "요약된 문장입니다.",
        }
    ]
    lines = finetune.build_training_lines(cast(list[dict[str, object]], rows), 0)
    assert "입력:" in lines[0]


def test_build_training_lines_question_answer() -> None:
    rows = [
        {
            "custom_id": "request-0",
            "question": "파이썬 스네이크 게임 만들어줘",
            "answer": "기본 게임 루프부터 설명할게요.",
        }
    ]
    lines = finetune.build_training_lines(cast(list[dict[str, object]], rows), 0)
    assert lines[0].startswith("질문:")
    assert "답변:" in lines[0]


def test_extract_candidate_tokens_limit() -> None:
    lines = ["안녕 반가워", "안녕 다시"]
    tokens = finetune.extract_candidate_tokens(lines, 2)
    assert tokens == ["안녕", "반가워"]


def test_contains_korean() -> None:
    assert finetune.contains_korean("한글토큰") is True
    assert finetune.contains_korean("english") is False


def test_build_swapped_vocab_skips_special_tokens() -> None:
    id_to_token = ["<pad>", "tokA", "123", "##"]
    candidate_tokens = ["한글1", "한글2"]
    new_tokens, swap_ids = finetune.build_swapped_vocab(
        id_to_token=id_to_token,
        candidate_tokens=candidate_tokens,
        special_tokens={"<pad>"},
    )
    assert new_tokens[0] == "<pad>"
    assert swap_ids


def test_apply_swapped_vocab_with_backend_json() -> None:
    class DummyBackendTokenizer:
        def get_vocab(self) -> dict[str, int]:
            return {"a": 0, "b": 1, "ab": 2}

        def to_str(self) -> str:
            return (
                "{"
                "\"model\": {"
                "\"vocab\": {\"a\": 0, \"b\": 1, \"ab\": 2},"
                "\"merges\": [[\"a\", \"b\"]]"
                "}"
                "}"
            )

    class DummyTokenizer:
        def __init__(self) -> None:
            self.backend_tokenizer = DummyBackendTokenizer()

    tokenizer = DummyTokenizer()
    finetune.apply_swapped_vocab(tokenizer, ["a", "b", "ab"])
    assert tokenizer.backend_tokenizer.model is not None


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


def test_ask_resume_yes() -> None:
    def fake_input(_: str) -> str:
        return "y"

    assert finetune.ask_resume(fake_input) is True


def test_ask_resume_no() -> None:
    def fake_input(_: str) -> str:
        return "n"

    assert finetune.ask_resume(fake_input) is False


def test_suggest_batch_size_cuda_vram_32(monkeypatch) -> None:
    monkeypatch.setattr(finetune, "resolve_total_vram_gb", lambda: 32.0)
    device = torch.device("cuda")
    assert finetune.suggest_batch_size(device) == 32


def test_suggest_batch_size_cuda_vram_boundary(monkeypatch) -> None:
    monkeypatch.setattr(finetune, "resolve_total_vram_gb", lambda: 24.0)
    device = torch.device("cuda")
    assert finetune.suggest_batch_size(device) == 16


def test_suggest_batch_size_cpu_ram_64(monkeypatch) -> None:
    monkeypatch.setattr(finetune, "resolve_total_ram_gb", lambda: 64.0)
    device = torch.device("cpu")
    assert finetune.suggest_batch_size(device) == 32


def test_build_training_lines_raises_on_empty() -> None:
    with pytest.raises(ValueError):
        finetune.build_training_lines([], 0)
