"""SmolLM2 데모 실행 스크립트."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ml_playground.smollm2.common import resolve_device


def parse_args() -> argparse.Namespace:
    # 데모 실행에 필요한 인자를 정의하고 기본값을 명시합니다.
    parser = argparse.ArgumentParser(description="SmolLM2 데모")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Gravity is",
        help="생성에 사용할 프롬프트",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="",
        help="여러 프롬프트를 담은 텍스트 파일 경로(한 줄=한 프롬프트)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
        help="로드할 체크포인트 경로 또는 허브 이름",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="생성할 최대 토큰 수",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="장치 선택(auto|cpu|cuda)",
    )
    return parser.parse_args()


def read_prompts(path: Path) -> list[str]:
    # 프롬프트 파일을 읽어 한 줄씩 정리합니다.
    # 빈 줄은 제거하여 불필요한 생성 호출을 줄입니다.
    if not path.exists():
        raise FileNotFoundError(f"프롬프트 파일이 없습니다: {path}")
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line:
                prompts.append(line)
    if not prompts:
        raise ValueError("프롬프트 파일에 유효한 내용이 없습니다.")
    return prompts


def resolve_prompts(single_prompt: str, prompts_file: str) -> list[str]:
    # 파일 경로가 있으면 그 내용을 우선합니다.
    # 없으면 단일 프롬프트를 리스트로 감싸 반환합니다.
    if prompts_file:
        return read_prompts(Path(prompts_file))
    return [single_prompt]


def main() -> None:
    # 1) 인자 파싱으로 실행 설정을 확정합니다.
    args = parse_args()
    # 2) 체크포인트와 장치를 결정합니다.
    checkpoint = args.checkpoint
    device = resolve_device(args.device)

    # 3) 토크나이저와 모델을 로드합니다.
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    torch.nn.Module.to(model, device)

    # 4) 프롬프트 목록을 준비하고 순차적으로 생성합니다.
    prompts = resolve_prompts(args.prompt, args.prompts_file)
    for index, prompt in enumerate(prompts, start=1):
        print(f"=== 프롬프트 {index}/{len(prompts)} ===")
        print(prompt)

        tokens = tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                return_dict_in_generate=False,
            )
        output_ids = cast(torch.LongTensor, outputs)[0]
        print(tokenizer.decode(output_ids.tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
