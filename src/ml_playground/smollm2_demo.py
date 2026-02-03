"""SmolLM2 135M Base 간단 실행 데모."""

from __future__ import annotations

import argparse
from typing import cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmolLM2 135M Base 데모")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Gravity is",
        help="생성에 사용할 프롬프트",
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


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def main() -> None:
    args = parse_args()

    checkpoint = "HuggingFaceTB/SmolLM2-135M"
    device = resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    torch.nn.Module.to(model, device)

    tokens = tokenizer(args.prompt, return_tensors="pt")
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
