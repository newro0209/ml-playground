"""SmolLM2 한국어 미세조정 성능 비교 스크립트."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmolLM2 한국어 성능 비교")
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M",
        help="원본 체크포인트",
    )
    parser.add_argument(
        "--finetuned-checkpoint",
        type=str,
        default="checkpoints/smollm2-ko",
        help="미세조정 체크포인트",
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        default="data/processed/korean_eval.txt",
        help="평가 텍스트 파일(한 줄=한 샘플)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="평가에 사용할 최대 샘플 수",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="평가 시 최대 시퀀스 길이",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="장치 선택(auto|cpu|cuda)",
    )
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def read_eval_lines(path: Path, max_samples: int) -> list[str]:
    if not path.exists():
        return fallback_eval_samples(max_samples)
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line:
                lines.append(line)
            if 0 < max_samples <= len(lines):
                break
    if not lines:
        return fallback_eval_samples(max_samples)
    return lines


def fallback_eval_samples(max_samples: int) -> list[str]:
    samples = [
        "안녕하세요? 오늘 기분이 어때요?",
        "친구와 다퉜는데 어떻게 하면 좋을까요?",
        "오늘 날씨가 좋아서 산책하고 싶어요.",
        "새로운 프로젝트를 시작하는 게 부담돼요.",
        "요즘 집중이 잘 안 되는데 조언해 주세요.",
        "일이 많아서 스트레스를 받고 있어요.",
        "맛있는 저녁 메뉴를 추천해 주세요.",
        "운동을 꾸준히 하려면 어떻게 해야 하나요?",
        "주말에 쉬고 싶은데 죄책감이 들어요.",
        "새로운 기술을 배우려면 어디서부터 시작해야 할까요?",
    ]
    if max_samples > 0:
        return samples[:max_samples]
    return samples


def compute_average_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        input_ids = cast(torch.Tensor, encoded["input_ids"]).to(device)
        attention_mask = cast(torch.Tensor, encoded.get("attention_mask", None))
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
        loss = cast(torch.Tensor, outputs.loss)
        total_loss += loss.item()
        total_count += 1
    if total_count == 0:
        raise ValueError("평가할 텍스트가 없습니다.")
    return total_loss / total_count


def report_results(base_loss: float, finetuned_loss: float) -> None:
    base_ppl = float(torch.exp(torch.tensor(base_loss)))
    finetuned_ppl = float(torch.exp(torch.tensor(finetuned_loss)))
    improvement = base_ppl - finetuned_ppl
    improvement_pct = (improvement / base_ppl) * 100.0
    print(f"원본 평균 손실: {base_loss:.4f}")
    print(f"미세조정 평균 손실: {finetuned_loss:.4f}")
    print(f"원본 퍼플렉시티: {base_ppl:.2f}")
    print(f"미세조정 퍼플렉시티: {finetuned_ppl:.2f}")
    print(f"퍼플렉시티 개선: {improvement:.2f} ({improvement_pct:.2f}%)")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    eval_lines = read_eval_lines(Path(args.eval_file), args.max_samples)

    base_tokenizer = cast(
        PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(args.base_checkpoint)
    )
    base_model = cast(
        PreTrainedModel, AutoModelForCausalLM.from_pretrained(args.base_checkpoint)
    )

    finetuned_tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(args.finetuned_checkpoint),
    )
    finetuned_model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(args.finetuned_checkpoint),
    )

    torch.nn.Module.to(base_model, device)
    torch.nn.Module.to(finetuned_model, device)

    base_loss = compute_average_loss(
        base_model, base_tokenizer, eval_lines, args.max_length, device
    )
    finetuned_loss = compute_average_loss(
        finetuned_model, finetuned_tokenizer, eval_lines, args.max_length, device
    )

    report_results(base_loss, finetuned_loss)


if __name__ == "__main__":
    main()
