"""smollm2_demo 테스트."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch

import ml_playground.smollm2_demo as demo


class DummyTokens(dict):
    """토크나이저 출력 흉내."""


class DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token = ""

    def __call__(self, prompt: str, return_tensors: str) -> DummyTokens:
        _ = return_tensors
        tokens: DummyTokens = DummyTokens()
        tokens["input_ids"] = torch.tensor([[1, 2, 3]])
        tokens["attention_mask"] = torch.tensor([[1, 1, 1]])
        return tokens

    def decode(self, ids: list[int], skip_special_tokens: bool) -> str:
        _ = ids
        _ = skip_special_tokens
        return "테스트 출력"


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        return_dict_in_generate: bool,
    ) -> torch.Tensor:
        _ = input_ids
        _ = attention_mask
        _ = max_new_tokens
        _ = do_sample
        _ = temperature
        _ = top_p
        _ = return_dict_in_generate
        return torch.tensor([[1, 2, 3, 4]])


def test_resolve_device_cpu() -> None:
    device = demo.resolve_device("cpu")
    assert device.type == "cpu"


def test_main_runs(monkeypatch, capsys) -> None:
    class FakeArgs:
        def __init__(self) -> None:
            self.prompt = "테스트"
            self.prompts_file = ""
            self.checkpoint = "dummy"
            self.max_new_tokens = 4
            self.device = "cpu"

    def fake_parse_args() -> FakeArgs:
        return FakeArgs()

    def fake_from_pretrained(_: str) -> DummyTokenizer:
        return DummyTokenizer()

    def fake_model_from_pretrained(_: str) -> DummyModel:
        return DummyModel()

    monkeypatch.setattr(demo, "parse_args", fake_parse_args)
    monkeypatch.setattr(demo.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(demo.AutoModelForCausalLM, "from_pretrained", fake_model_from_pretrained)

    demo.main()
    captured = cast(str, capsys.readouterr().out)
    assert "테스트 출력" in captured


def test_read_prompts(tmp_path: Path) -> None:
    path = tmp_path / "prompts.txt"
    path.write_text("안녕\n\n반가워\n", encoding="utf-8")
    prompts = demo.read_prompts(path)
    assert prompts == ["안녕", "반가워"]


def test_resolve_prompts_from_file(tmp_path: Path) -> None:
    path = tmp_path / "prompts.txt"
    path.write_text("첫 번째\n두 번째\n", encoding="utf-8")
    prompts = demo.resolve_prompts("무시", path.as_posix())
    assert prompts == ["첫 번째", "두 번째"]


def test_resolve_prompts_single() -> None:
    prompts = demo.resolve_prompts("단일", "")
    assert prompts == ["단일"]
