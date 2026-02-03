"""SmolLM2 135M 한국어 미세조정 스크립트."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ml_playground.smollm2_common import resolve_device


class TextLineDataset(Dataset):
    """텍스트 라인 기반 데이터셋."""

    def __init__(
        self, lines: list[str], tokenizer: PreTrainedTokenizerBase, max_length: int
    ) -> None:
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
    # 커맨드라인 인자를 정의해 실행 흐름을 명확히 통제합니다.
    # 기본값은 Instruct 미세조정에 맞춘 설정으로 구성합니다.
    parser = argparse.ArgumentParser(description="SmolLM2 한국어 미세조정")
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/processed/ko_instruct_train.txt",
        help="학습에 사용할 텍스트 파일(한 줄=한 샘플)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
        help="기본 체크포인트",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/smollm2-ko-instruct",
        help="모델 저장 경로",
    )
    parser.add_argument("--epochs", type=int, default=1, help="학습 에폭 수")
    parser.add_argument(
        "--batch-size", type=int, default=0, help="배치 크기(0이면 자동)"
    )
    parser.add_argument(
        "--auto-batch-size",
        action="store_true",
        help="RAM/VRAM을 기준으로 배치 크기를 자동으로 계산",
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="학습률")
    parser.add_argument("--max-length", type=int, default=256, help="최대 시퀀스 길이")
    parser.add_argument(
        "--dataset",
        type=str,
        default="beomi/KoAlpaca-RealQA",
        help="자동 다운로드에 사용할 데이터셋(접근 동의가 필요한 데이터셋 포함)",
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
    parser.add_argument(
        "--resume-mode",
        type=str,
        default="ask",
        choices=["ask", "always", "never"],
        help="중단 이후 재개 여부(ask|always|never)",
    )
    parser.add_argument(
        "--tokenizer-swap",
        action="store_true",
        help="데이터셋 토큰으로 토크나이저 일부를 교체",
    )
    parser.add_argument(
        "--tokenizer-swap-min-pct",
        type=float,
        default=0.5,
        help="교체 시작 비율(예: 0.5)",
    )
    parser.add_argument(
        "--tokenizer-swap-max-pct",
        type=float,
        default=0.7,
        help="교체 종료 비율(예: 0.7)",
    )
    parser.add_argument(
        "--tokenizer-swap-max-tokens",
        type=int,
        default=2000,
        help="교체에 사용할 최대 토큰 수",
    )
    return parser.parse_args()


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


def extract_candidate_tokens(
    lines: list[str],
    max_tokens: int,
) -> list[str]:
    # 데이터셋 텍스트에서 후보 토큰을 추출합니다.
    # 공백 기반으로 분리하되 불필요한 기호를 제거해 간단한 후보 집합을 만듭니다.
    tokens: list[str] = []
    seen: set[str] = set()
    for line in lines:
        for raw_token in line.replace("\n", " ").split():
            token = raw_token.strip()
            if not token:
                continue
            if token in seen:
                continue
            seen.add(token)
            tokens.append(token)
            if 0 < max_tokens <= len(tokens):
                return tokens
    return tokens


def resolve_swap_range(
    vocab_size: int,
    min_pct: float,
    max_pct: float,
) -> tuple[int, int]:
    # 토크나이저의 교체 구간을 비율로 계산합니다.
    # 잘못된 비율 값은 명시적으로 실패시켜 오류를 드러냅니다.
    if vocab_size <= 0:
        raise ValueError("vocab_size는 1 이상이어야 합니다.")
    if not (0.0 <= min_pct < max_pct <= 1.0):
        raise ValueError("교체 비율은 0.0 <= min < max <= 1.0 이어야 합니다.")
    start = int(vocab_size * min_pct)
    end = int(vocab_size * max_pct)
    if start >= end:
        raise ValueError("교체 구간이 비어 있습니다.")
    return start, end


def build_swapped_vocab(
    id_to_token: list[str],
    candidate_tokens: list[str],
    start: int,
    end: int,
    special_tokens: set[str],
) -> tuple[list[str], list[int]]:
    # 지정된 구간을 후보 토큰으로 교체한 새로운 vocab 리스트를 만듭니다.
    # 특수 토큰은 교체하지 않도록 보호합니다.
    if start < 0 or end > len(id_to_token):
        raise ValueError("교체 범위가 vocab 크기를 벗어났습니다.")
    if start >= end:
        raise ValueError("교체 범위가 비어 있습니다.")
    new_tokens = list(id_to_token)
    swapped_ids: list[int] = []
    candidate_index = 0
    for vocab_id in range(start, end):
        current_token = id_to_token[vocab_id]
        if current_token in special_tokens:
            continue
        while candidate_index < len(candidate_tokens):
            candidate = candidate_tokens[candidate_index]
            candidate_index += 1
            if candidate in special_tokens:
                continue
            if candidate in new_tokens:
                continue
            new_tokens[vocab_id] = candidate
            swapped_ids.append(vocab_id)
            break
        if candidate_index >= len(candidate_tokens):
            break
    if not swapped_ids:
        raise ValueError("교체할 토큰이 충분하지 않습니다.")
    return new_tokens, swapped_ids


def apply_swapped_vocab(
    tokenizer: PreTrainedTokenizerBase,
    new_id_to_token: list[str],
) -> None:
    # Fast 토크나이저의 vocab을 직접 교체합니다.
    # 내부 모델 접근이 불가능하면 명확한 오류를 발생시킵니다.
    if not hasattr(tokenizer, "backend_tokenizer"):
        raise ValueError("Fast 토크나이저가 아니어서 vocab 교체를 지원하지 않습니다.")
    backend_tokenizer = tokenizer.backend_tokenizer
    model = getattr(backend_tokenizer, "model", None)
    if model is None:
        raise ValueError("토크나이저 모델에서 vocab 정보를 가져올 수 없습니다.")
    # 일부 tokenizers 버전은 get_vocab/get_merges 대신 속성 접근만 제공합니다.
    vocab = model.get_vocab() if hasattr(model, "get_vocab") else getattr(model, "vocab", None)
    if not isinstance(vocab, dict):
        raise ValueError("vocab 정보가 dict 형태가 아닙니다.")
    merges = (
        model.get_merges()
        if hasattr(model, "get_merges")
        else getattr(model, "merges", None)
    )
    if merges is None:
        raise ValueError("BPE merges 정보를 가져올 수 없습니다.")
    if len(new_id_to_token) != len(vocab):
        raise ValueError("vocab 크기가 일치하지 않아 교체를 중단합니다.")
    # 기존 id 체계를 유지하면서 token 문자열만 교체합니다.
    new_vocab = {token: idx for idx, token in enumerate(new_id_to_token)}
    try:
        from tokenizers.models import BPE
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("tokenizers 패키지가 필요합니다.") from exc
    new_model = BPE(new_vocab, merges)
    backend_tokenizer.model = new_model


def reinitialize_embeddings(
    model: PreTrainedModel,
    swap_ids: list[int],
) -> None:
    # 교체된 토큰의 임베딩을 무작위 초기화로 리셋합니다.
    # 기존 분포를 참고해 std를 계산해 안정적인 초기화를 유지합니다.
    if not swap_ids:
        return
    embeddings = model.get_input_embeddings()
    if embeddings is None:
        raise ValueError("입력 임베딩을 찾을 수 없습니다.")
    weight = embeddings.weight.data
    std = float(weight.std().item())
    if std == 0.0:
        std = 0.02
    for idx in swap_ids:
        if 0 <= idx < weight.size(0):
            weight[idx].normal_(mean=0.0, std=std)


def swap_tokenizer_with_dataset(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    lines: list[str],
    min_pct: float,
    max_pct: float,
    max_tokens: int,
) -> list[int]:
    # 데이터셋 기반으로 토크나이저 vocab을 일부 교체합니다.
    # 교체 이후 임베딩을 재초기화하여 학습으로 재적응할 수 있게 합니다.
    candidate_tokens = extract_candidate_tokens(lines, max_tokens)
    if not candidate_tokens:
        raise ValueError("교체에 사용할 토큰을 추출하지 못했습니다.")
    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, dict):
        raise ValueError("tokenizer.get_vocab 결과가 dict가 아닙니다.")
    vocab_size = len(vocab)
    start, end = resolve_swap_range(vocab_size, min_pct, max_pct)
    id_to_token = [""] * vocab_size
    for token, idx in vocab.items():
        if 0 <= idx < vocab_size:
            id_to_token[idx] = token
    special_tokens = set(tokenizer.all_special_tokens)
    new_id_to_token, swap_ids = build_swapped_vocab(
        id_to_token=id_to_token,
        candidate_tokens=candidate_tokens,
        start=start,
        end=end,
        special_tokens=special_tokens,
    )
    apply_swapped_vocab(tokenizer, new_id_to_token)
    model.resize_token_embeddings(len(tokenizer))
    reinitialize_embeddings(model, swap_ids)
    return swap_ids
def resolve_total_ram_gb() -> float:
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if isinstance(pages, int) and isinstance(page_size, int):
                return float(pages * page_size) / (1024**3)
        except (ValueError, OSError):
            return 0.0
    return 0.0


def resolve_total_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    return float(props.total_memory) / (1024**3)


def suggest_batch_size(device: torch.device) -> int:
    if device.type == "cuda":
        # CUDA 사용 시 VRAM을 기준으로 배치 크기를 단계적으로 올립니다.
        # 고용량 VRAM에서는 학습 안정성과 처리량을 동시에 확보하기 위해 상향값을 허용합니다.
        vram_gb = resolve_total_vram_gb()
        # 32GB 이상은 대형 배치로도 메모리 여유가 충분하다는 가정을 둡니다.
        # 다만 너무 공격적인 값은 OOM 위험을 높이므로 보수적으로 설정합니다.
        if vram_gb >= 32:
            return 32
        if vram_gb >= 24:
            return 16
        if vram_gb >= 16:
            return 8
        if vram_gb >= 8:
            return 4
        return 2
    # CPU 경로는 시스템 RAM 기준으로 배치 크기를 결정합니다.
    # RAM이 많을수록 데이터 로딩과 텐서 버퍼링 여유가 커진다는 전제입니다.
    ram_gb = resolve_total_ram_gb()
    # 64GB 이상에서는 더 큰 배치를 시도해도 메모리 압박이 낮다고 판단합니다.
    # 학습 안정성을 고려해 32보다 작은 값으로 상한을 둡니다.
    if ram_gb >= 64:
        return 32
    if ram_gb >= 32:
        return 16
    if ram_gb >= 16:
        return 8
    if ram_gb >= 8:
        return 4
    return 2


def build_training_lines(
    dataset_rows: list[dict[str, object]],
    max_samples: int,
) -> list[str]:
    # 데이터셋의 다양한 스키마를 처리하기 위해 명시적 분기를 사용합니다.
    # instruction/output, instruction/response, text 형태를 모두 지원합니다.
    lines: list[str] = []
    limit = max_samples if max_samples > 0 else len(dataset_rows)
    for row in dataset_rows[:limit]:
        # 1) Instruct 스키마를 우선 처리합니다.
        instruct_line = build_instruct_line(row)
        if instruct_line:
            lines.append(instruct_line)
            continue
        # 2) 단일 텍스트 필드를 가진 데이터셋은 그대로 사용합니다.
        text_line = build_text_line(row)
        if text_line:
            lines.append(text_line)
    if not lines:
        raise ValueError("데이터셋에서 유효한 텍스트를 만들 수 없습니다.")
    return lines


def build_instruct_line(row: dict[str, object]) -> str | None:
    # Instruct 데이터셋의 필드를 읽어 학습 문장을 구성합니다.
    # 입력이 부족하면 None을 반환해 상위 로직에서 다른 스키마로 넘어가게 합니다.
    # 데이터셋마다 필드명이 다르므로 질문/응답 후보를 모두 확인합니다.
    instruction = row.get("instruction")
    question = row.get("question")
    output = row.get("output")
    response = row.get("response")
    answer_text = row.get("answer")
    input_text = row.get("input")

    # 질문 필드가 없으면 Instruct 스키마로 처리하지 않습니다.
    prompt = (
        instruction
        if isinstance(instruction, str)
        else question if isinstance(question, str) else None
    )
    if not isinstance(prompt, str):
        return None
    # 응답 필드도 여러 후보를 허용합니다.
    answer = (
        output
        if isinstance(output, str)
        else (
            response
            if isinstance(response, str)
            else answer_text if isinstance(answer_text, str) else None
        )
    )
    if not isinstance(answer, str):
        return None

    # Instruct 튜닝용 포맷은 "질문/입력/답변"을 명시적으로 분리합니다.
    # 입력이 없으면 질문과 답변만 사용해 불필요한 토큰을 줄입니다.
    if isinstance(input_text, str) and input_text.strip():
        combined = f"질문: {prompt} 입력: {input_text} 답변: {answer}"
    else:
        combined = f"질문: {prompt} 답변: {answer}"
    line = sanitize_text(combined)
    return line if line else None


def build_text_line(row: dict[str, object]) -> str | None:
    # text 필드를 그대로 사용하는 단순 스키마를 지원합니다.
    # 공백 제거 후 비어 있으면 None을 반환합니다.
    text = row.get("text")
    if isinstance(text, str):
        line = sanitize_text(text)
        return line if line else None
    return None


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


def save_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir.as_posix())
    tokenizer.save_pretrained(output_dir.as_posix())


def save_trainer_state(
    output_dir: Path,
    epoch: int,
    global_step: int,
    optimizer: torch.optim.Optimizer,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(state, output_dir / "trainer_state.pt")


def load_trainer_state(output_dir: Path) -> dict[str, object]:
    state_path = output_dir / "trainer_state.pt"
    if not state_path.exists():
        return {}
    return cast(dict[str, object], torch.load(state_path, map_location="cpu"))


def has_resume_checkpoint(output_dir: Path) -> bool:
    return (output_dir / "config.json").exists()


def ask_resume(input_func) -> bool:
    answer = input_func("이전 학습을 이어서 진행할까요? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def main() -> None:
    args = parse_args()

    train_path = Path(args.train_file)
    output_dir = Path(args.output_dir)
    device = resolve_device(args.device)

    torch.manual_seed(42)

    resume_available = has_resume_checkpoint(output_dir)
    resume = False
    if resume_available:
        if args.resume_mode == "always":
            resume = True
        elif args.resume_mode == "ask":
            resume = ask_resume(input)

    if args.auto_batch_size or args.batch_size <= 0:
        args.batch_size = suggest_batch_size(device)
        print(f"자동 배치 크기 설정: {args.batch_size}")

    needs_prepare = (
        args.force_prepare or not train_path.exists() or train_path.stat().st_size == 0
    )
    if needs_prepare:
        print("학습 파일이 없어 데이터셋을 다운로드합니다.")
        prepare_dataset(
            train_path=train_path,
            dataset_name=args.dataset,
            split_name=args.dataset_split,
            max_samples=args.dataset_max_samples,
        )

    # 학습 데이터 라인을 먼저 읽어 토크나이저 교체에도 활용합니다.
    lines = read_lines(train_path)

    if resume:
        print(f"체크포인트에서 재개합니다: {output_dir}")
        tokenizer = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(output_dir.as_posix()),
        )
        model = cast(
            PreTrainedModel, AutoModelForCausalLM.from_pretrained(output_dir.as_posix())
        )
    else:
        tokenizer = cast(
            PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(args.checkpoint)
        )
        model = cast(
            PreTrainedModel, AutoModelForCausalLM.from_pretrained(args.checkpoint)
        )
    if args.tokenizer_swap:
        print("토크나이저 교체를 시작합니다.")
        swap_ids = swap_tokenizer_with_dataset(
            tokenizer=tokenizer,
            model=model,
            lines=lines,
            min_pct=args.tokenizer_swap_min_pct,
            max_pct=args.tokenizer_swap_max_pct,
            max_tokens=args.tokenizer_swap_max_tokens,
        )
        print(f"토크나이저 교체 완료: {len(swap_ids)}개 토큰")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("pad_token_id 설정을 위해 eos_token이 필요합니다.")
        tokenizer.pad_token = tokenizer.eos_token

    torch.nn.Module.to(model, device)

    dataset = TextLineDataset(lines, tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 0
    global_step = 0
    if resume:
        state = load_trainer_state(output_dir)
        epoch_value = state.get("epoch")
        step_value = state.get("global_step")
        optimizer_state = state.get("optimizer_state")
        if isinstance(epoch_value, int):
            start_epoch = epoch_value
        if isinstance(step_value, int):
            global_step = step_value
        if isinstance(optimizer_state, dict):
            optimizer.load_state_dict(optimizer_state)

    model.train()
    for epoch in range(start_epoch, args.epochs):
        progress = build_progress_bar(dataloader, f"에폭 {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(progress):
            input_ids = cast(torch.Tensor, batch["input_ids"]).to(device)
            attention_mask = cast(torch.Tensor, batch["attention_mask"]).to(device)
            labels = cast(torch.Tensor, batch["labels"]).to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = cast(torch.Tensor, outputs.loss)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if global_step % args.log_every == 0:
                set_progress_postfix(progress, loss.item())
            if (
                args.save_every > 0
                and global_step % args.save_every == 0
                and global_step > 0
            ):
                save_checkpoint(model, tokenizer, output_dir)
                save_trainer_state(output_dir, epoch, global_step, optimizer)

            global_step += 1

        save_checkpoint(model, tokenizer, output_dir)
        save_trainer_state(output_dir, epoch + 1, global_step, optimizer)

    print(f"학습 완료. 저장 위치: {output_dir}")


def build_progress_bar(dataloader: DataLoader, desc: str):
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        return dataloader
    return tqdm(dataloader, desc=desc)


def set_progress_postfix(progress, loss_value: float) -> None:
    if hasattr(progress, "set_postfix"):
        progress.set_postfix({"loss": f"{loss_value:.4f}"})


if __name__ == "__main__":
    main()
