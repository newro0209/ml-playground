"""run_*.sh 실행 커맨드 연동 테스트."""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

import torch

from ml_playground import smollm2_demo
from ml_playground import smollm2_eval
from ml_playground import smollm2_finetune
from ml_playground import smollm2_gguf


def extract_script_args(
    script_path: Path,
    module_name: str,
    occurrence: int = 0,
) -> list[str]:
    # 1) 스크립트 내용을 읽어 모듈 실행 라인을 찾습니다.
    # 2) 이어지는 인자 라인을 합쳐 하나의 명령 문자열을 만듭니다.
    # 3) shlex로 분리해 인자 리스트만 반환합니다.
    text = script_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    start_index = -1
    found = -1
    for index, line in enumerate(lines):
        if f"python -m {module_name}" in line:
            found += 1
            if found == occurrence:
                start_index = index
                break
    if start_index < 0:
        raise ValueError("스크립트에서 모듈 실행 라인을 찾지 못했습니다.")
    head = lines[start_index]
    tail = head.split(f"python -m {module_name}", 1)[1]
    collected: list[str] = [tail]
    for line in lines[start_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            break
        if not stripped.startswith("--"):
            break
        collected.append(stripped)
    joined = " ".join(part.rstrip("\\") for part in collected)
    args = shlex.split(joined)
    expanded: list[str] = []
    for token in args:
        expanded.append(os.path.expandvars(token))
    return expanded


def test_extract_script_args_missing_module(tmp_path: Path) -> None:
    script = tmp_path / "run_missing.sh"
    script.write_text("echo nope\n", encoding="utf-8")
    try:
        extract_script_args(script, "ml_playground.nope")
    except ValueError as exc:
        assert "모듈 실행 라인" in str(exc)
        return
    raise AssertionError("ValueError가 발생해야 합니다.")


def test_run_smollm2_finetune_script(monkeypatch, tmp_path: Path) -> None:
    # 1) 스크립트에서 인자를 추출합니다.
    # 2) 학습 루프가 가볍게 끝나도록 핵심 의존성을 더미로 교체합니다.
    # 3) 동일 인자로 main을 호출해 예외 없이 종료되는지 확인합니다.
    args = extract_script_args(
        Path("scripts/run_smollm2_finetune.sh"), "ml_playground.smollm2_finetune"
    )

    class DummyTokenizer:
        def __init__(self) -> None:
            self.pad_token_id = 0
            self.eos_token = ""

        def __call__(
            self,
            text: str,
            return_tensors: str,
            truncation: bool,
            max_length: int,
            padding: str,
        ) -> dict[str, torch.Tensor]:
            _ = text
            _ = return_tensors
            _ = truncation
            _ = max_length
            _ = padding
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

        def save_pretrained(self, _: str) -> None:
            return None

    class DummyOutput:
        def __init__(self, loss: torch.Tensor) -> None:
            self.loss = loss

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: torch.Tensor,
        ) -> DummyOutput:
            _ = input_ids
            _ = attention_mask
            _ = labels
            return DummyOutput(self.dummy.sum())

        def save_pretrained(self, _: str) -> None:
            return None

    monkeypatch.setattr(
        smollm2_finetune.AutoTokenizer,
        "from_pretrained",
        lambda _: DummyTokenizer(),
    )
    monkeypatch.setattr(
        smollm2_finetune.AutoModelForCausalLM,
        "from_pretrained",
        lambda _: DummyModel(),
    )
    monkeypatch.setattr(smollm2_finetune, "read_lines", lambda _: ["테스트 문장"])
    monkeypatch.setattr(smollm2_finetune, "load_hf_dataset", lambda *_: [])
    monkeypatch.setattr(smollm2_finetune, "save_checkpoint", lambda *_: None)
    monkeypatch.setattr(smollm2_finetune, "save_trainer_state", lambda *_: None)
    def fake_swap(
        tokenizer: object,
        model: object,
        lines: list[str],
        max_tokens: int,
    ) -> list[int]:
        _ = tokenizer
        _ = model
        _ = lines
        _ = max_tokens
        return []

    monkeypatch.setattr(smollm2_finetune, "swap_tokenizer_with_dataset", fake_swap)
    monkeypatch.setattr(smollm2_finetune, "resolve_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(smollm2_finetune, "has_resume_checkpoint", lambda _: False)

    checkpoint_dir = Path("checkpoints/smollm2-ko-instruct")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_file = Path("data/processed/ko_instruct_train.txt")
    train_file.parent.mkdir(parents=True, exist_ok=True)
    train_file.write_text("테스트\n", encoding="utf-8")

    monkeypatch.setenv("PYTHONPATH", "./src")
    monkeypatch.setattr(sys, "argv", ["smollm2_finetune.py", *args])
    smollm2_finetune.main()


def test_run_smollm2_demo_script(monkeypatch) -> None:
    # 1) 스크립트 인자를 읽습니다.
    # 2) 모델/토크나이저를 더미로 교체합니다.
    # 3) 동일 인자로 main이 동작하는지 확인합니다.
    monkeypatch.setenv(
        "BASE_CHECKPOINT", "HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    monkeypatch.setenv("PROMPTS_FILE", "data/prompts/ko_instruct_eval.txt")
    args = extract_script_args(
        Path("scripts/run_smollm2_demo.sh"),
        "ml_playground.smollm2_demo",
        occurrence=0,
    )

    class DummyTokenizer:
        def __call__(self, prompt: str, return_tensors: str) -> dict[str, torch.Tensor]:
            _ = prompt
            _ = return_tensors
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

        def decode(self, ids: list[int], skip_special_tokens: bool) -> str:
            _ = ids
            _ = skip_special_tokens
            return "데모 출력"

    class DummyModel(torch.nn.Module):
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
            return torch.tensor([[1, 2, 3]])

    monkeypatch.setattr(
        smollm2_demo.AutoTokenizer,
        "from_pretrained",
        lambda _: DummyTokenizer(),
    )
    monkeypatch.setattr(
        smollm2_demo.AutoModelForCausalLM,
        "from_pretrained",
        lambda _: DummyModel(),
    )
    monkeypatch.setattr(smollm2_demo, "resolve_prompts", lambda *_: ["테스트"])
    monkeypatch.setattr(sys, "argv", ["smollm2_demo.py", *args])
    smollm2_demo.main()


def test_run_smollm2_demo_script_finetuned(monkeypatch) -> None:
    # 1) 스크립트의 두 번째 실행 인자를 읽습니다.
    # 2) 모델/토크나이저를 더미로 교체합니다.
    # 3) 동일 인자로 main이 동작하는지 확인합니다.
    monkeypatch.setenv("FINETUNED_CHECKPOINT", "checkpoints/smollm2-ko-instruct")
    monkeypatch.setenv("PROMPTS_FILE", "data/prompts/ko_instruct_eval.txt")
    args = extract_script_args(
        Path("scripts/run_smollm2_demo.sh"),
        "ml_playground.smollm2_demo",
        occurrence=1,
    )

    class DummyTokenizer:
        def __call__(self, prompt: str, return_tensors: str) -> dict[str, torch.Tensor]:
            _ = prompt
            _ = return_tensors
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

        def decode(self, ids: list[int], skip_special_tokens: bool) -> str:
            _ = ids
            _ = skip_special_tokens
            return "데모 출력"

    class DummyModel(torch.nn.Module):
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
            return torch.tensor([[1, 2, 3]])

    monkeypatch.setattr(
        smollm2_demo.AutoTokenizer,
        "from_pretrained",
        lambda _: DummyTokenizer(),
    )
    monkeypatch.setattr(
        smollm2_demo.AutoModelForCausalLM,
        "from_pretrained",
        lambda _: DummyModel(),
    )
    monkeypatch.setattr(smollm2_demo, "resolve_prompts", lambda *_: ["테스트"])
    monkeypatch.setattr(sys, "argv", ["smollm2_demo.py", *args])
    smollm2_demo.main()


def test_run_smollm2_eval_script(monkeypatch) -> None:
    # 1) 스크립트 인자를 읽습니다.
    # 2) 평가 계산을 더미 값으로 대체합니다.
    # 3) 동일 인자로 main이 동작하는지 확인합니다.
    args = extract_script_args(
        Path("scripts/run_smollm2_eval.sh"), "ml_playground.smollm2_eval"
    )

    class DummyTokenizer:
        pass

    class DummyModel(torch.nn.Module):
        pass

    monkeypatch.setattr(
        smollm2_eval.AutoTokenizer,
        "from_pretrained",
        lambda _: DummyTokenizer(),
    )
    monkeypatch.setattr(
        smollm2_eval.AutoModelForCausalLM,
        "from_pretrained",
        lambda _: DummyModel(),
    )
    monkeypatch.setattr(smollm2_eval, "compute_average_loss", lambda *_: 1.0)
    monkeypatch.setattr(sys, "argv", ["smollm2_eval.py", *args])
    smollm2_eval.main()


def test_run_smollm2_gguf_script(monkeypatch, tmp_path: Path) -> None:
    # 1) 스크립트 인자를 읽습니다.
    # 2) 필요한 디렉터리와 변환 스크립트를 준비합니다.
    # 3) 동일 인자로 main이 동작하는지 확인합니다.
    llama_dir = tmp_path / "llama.cpp"
    llama_dir.mkdir(parents=True, exist_ok=True)
    (llama_dir / "convert_hf_to_gguf.py").write_text("# dummy", encoding="utf-8")
    checkpoint_dir = Path("checkpoints/smollm2-ko-instruct")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("LLAMA_CPP_DIR", llama_dir.as_posix())
    args = extract_script_args(
        Path("scripts/run_smollm2_gguf.sh"), "ml_playground.smollm2_gguf"
    )
    monkeypatch.setattr(smollm2_gguf, "run_convert", lambda *_: None)
    monkeypatch.setattr(sys, "argv", ["smollm2_gguf.py", *args])
    smollm2_gguf.main()
