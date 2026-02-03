"""SmolLM2 135M 한국어 미세조정 스크립트."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class TextLineDataset(Dataset):
    """텍스트 라인 기반 데이터셋."""

    def __init__(self, lines: list[str], tokenizer: AutoTokenizer, max_length: int) -> None:
        self._lines = lines
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._lines)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        text = self._lines[index]
        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
            padding="max_length",
        )
        input_ids = cast(torch.Tensor, encoded["input_ids"]).squeeze(0)
        attention_mask = cast(torch.Tensor, encoded["attention_mask"]).squeeze(0)
        labels = input_ids.clone()
        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmolLM2 한국어 미세조정")
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/processed/korean_train.txt",
        help="학습에 사용할 텍스트 파일(한 줄=한 샘플)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M",
        help="기본 체크포인트",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/smollm2-ko",
        help="모델 저장 경로",
    )
    parser.add_argument("--epochs", type=int, default=1, help="학습 에폭 수")
    parser.add_argument("--batch-size", type=int, default=2, help="배치 크기")
    parser.add_argument("--lr", type=float, default=5e-5, help="학습률")
    parser.add_argument("--max-length", type=int, default=256, help="최대 시퀀스 길이")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ohilikeit/empathetic_dialogues_mutli_turn_ko",
        help="자동 다운로드에 사용할 데이터셋",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="데이터셋 분할 이름",
    )
    parser.add_argument(
        "--dataset-max-samples",
        type=int,
        default=0,
        help="사용할 최대 샘플 수(0이면 전체 사용)",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="학습 파일이 있어도 데이터셋을 다시 내려받아 덮어쓰기",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="장치 선택(auto|cpu|cuda)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=20,
        help="로그 출력 스텝 간격",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="체크포인트 저장 스텝 간격(0이면 에폭 종료 시만 저장)",
    )
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"학습 파일이 없습니다: {path}")
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line:
                lines.append(line)
    if not lines:
        raise ValueError("학습 파일에 유효한 텍스트 라인이 없습니다.")
    return lines


def sanitize_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split()).strip()


def build_training_lines(
    dataset_rows: list[dict[str, object]],
    max_samples: int,
) -> list[str]:
    lines: list[str] = []
    limit = max_samples if max_samples > 0 else len(dataset_rows)
    for row in dataset_rows[:limit]:
        instruction = row.get("instruction")
        output = row.get("output")
        if isinstance(instruction, str) and isinstance(output, str):
            combined = f"질문: {instruction} 답변: {output}"
            line = sanitize_text(combined)
            if line:
                lines.append(line)
            continue
        text = row.get("text")
        if isinstance(text, str):
            line = sanitize_text(text)
            if line:
                lines.append(line)
    if not lines:
        raise ValueError("데이터셋에서 유효한 텍스트를 만들 수 없습니다.")
    return lines


def prepare_dataset(
    train_path: Path,
    dataset_name: str,
    split_name: str,
    max_samples: int,
) -> None:
    print(f"데이터셋 로딩: {dataset_name} ({split_name})")
    dataset = load_hf_dataset(dataset_name, split_name)
    raw_rows = cast(list[dict[str, object]], list(dataset))
    lines = build_training_lines(raw_rows, max_samples)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with train_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def load_hf_dataset(dataset_name: str, split_name: str) -> list[dict[str, object]]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "datasets 패키지가 필요합니다. `pip install datasets`로 설치하세요."
        ) from exc
    dataset = load_dataset(path=dataset_name, split=split_name)
    return cast(list[dict[str, object]], list(dataset))


def save_checkpoint(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir.as_posix())
    tokenizer.save_pretrained(output_dir.as_posix())


def main() -> None:
    args = parse_args()

    train_path = Path(args.train_file)
    output_dir = Path(args.output_dir)
    device = resolve_device(args.device)

    torch.manual_seed(42)

    needs_prepare = args.force_prepare or not train_path.exists() or train_path.stat().st_size == 0
    if needs_prepare:
        print("학습 파일이 없어 데이터셋을 다운로드합니다.")
        prepare_dataset(
            train_path=train_path,
            dataset_name=args.dataset,
            split_name=args.dataset_split,
            max_samples=args.dataset_max_samples,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("pad_token_id 설정을 위해 eos_token이 필요합니다.")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    torch.nn.Module.to(model, device)

    lines = read_lines(train_path)
    dataset = TextLineDataset(lines, tokenizer, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            input_ids = cast(torch.Tensor, batch["input_ids"]).to(device)
            attention_mask = cast(torch.Tensor, batch["attention_mask"]).to(device)
            labels = cast(torch.Tensor, batch["labels"]).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = cast(torch.Tensor, outputs.loss)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if global_step % args.log_every == 0:
                print(
                    f"에폭 {epoch + 1}/{args.epochs} | 스텝 {step} | "
                    f"손실 {loss.item():.4f}"
                )
            if args.save_every > 0 and global_step % args.save_every == 0 and global_step > 0:
                save_checkpoint(model, tokenizer, output_dir)

            global_step += 1

        save_checkpoint(model, tokenizer, output_dir)

    print(f"학습 완료. 저장 위치: {output_dir}")


if __name__ == "__main__":
    main()
