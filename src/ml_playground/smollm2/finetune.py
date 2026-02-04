"""SmolLM2 135M 한국어 미세조정 스크립트."""

from __future__ import annotations

import argparse
import logging
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

from ml_playground.smollm2.common import resolve_device


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
        default="",
        help="모델 저장 경로(지정 시 output-root/run-name보다 우선)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="checkpoints/smollm2",
        help="모델 저장 최상위 폴더",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="ko-instruct",
        help="학습 실행 이름(모델 하위 폴더명)",
    )
    parser.add_argument(
        "--latest-link",
        action="store_true",
        help="output-root 기준 최신 실행 심볼릭 링크를 생성",
    )
    parser.add_argument(
        "--latest-name",
        type=str,
        default="latest",
        help="최신 실행 심볼릭 링크 이름",
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
        help="자동 다운로드에 사용할 단일 데이터셋(레거시 옵션)",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="단일 데이터셋 분할 이름(레거시 옵션)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=(
            "beomi/KoAlpaca-RealQA:train:ko,"
            "Ammad1Ali/Korean-conversational-dataset:train:ko,"
            "Ahren09/empathetic_dialogues:train:en"
        ),
        help=(
            "학습에 사용할 데이터셋 목록(콤마 구분, name[:split[:lang]] 형식)"
        ),
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
        "--tokenizer-rebuild",
        action="store_true",
        help="데이터셋에서 새 토크나이저를 학습해 전체 vocab을 교체",
    )
    parser.add_argument(
        "--tokenizer-vocab-size",
        type=int,
        default=0,
        help="새 토크나이저 vocab 크기(0이면 기존 vocab 크기 사용)",
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


def infer_dataset_lang(dataset_name: str) -> str:
    # 1) 알려진 데이터셋은 사람이 판단한 언어 정보를 우선 사용합니다.
    # 2) 매핑에 없으면 추론하지 않고 "unknown"으로 남겨 잘못된 분류를 피합니다.
    known_map = {
        "beomi/KoAlpaca-RealQA": "ko",
        "Ammad1Ali/Korean-conversational-dataset": "ko",
        "Ahren09/empathetic_dialogues": "en",
    }
    return known_map.get(dataset_name, "unknown")


def parse_dataset_spec(spec: str) -> dict[str, str]:
    # 1) 입력 문자열을 정리해 빈 값 여부를 확인합니다.
    # 2) ":" 기준으로 분리해 name/split/lang 구성으로 해석합니다.
    # 3) 누락된 split/lang은 기본값으로 채워 명시적 설정을 보장합니다.
    cleaned = spec.strip()
    if not cleaned:
        raise ValueError("빈 데이터셋 스펙은 허용하지 않습니다.")
    parts = [part.strip() for part in cleaned.split(":")]
    if len(parts) > 3:
        raise ValueError("데이터셋 스펙은 name[:split[:lang]] 형식이어야 합니다.")
    dataset_name = parts[0]
    split_name = parts[1] if len(parts) >= 2 and parts[1] else "train"
    lang = parts[2] if len(parts) == 3 and parts[2] else infer_dataset_lang(dataset_name)
    if not dataset_name:
        raise ValueError("데이터셋 이름이 비어 있습니다.")
    if not split_name:
        raise ValueError("데이터셋 분할 이름이 비어 있습니다.")
    return {"name": dataset_name, "split": split_name, "lang": lang}


def resolve_dataset_specs(
    datasets_arg: str,
    fallback_name: str,
    fallback_split: str,
) -> list[dict[str, str]]:
    # 1) 새 옵션이 비어 있으면 레거시 단일 옵션으로 복구합니다.
    # 2) 콤마로 분리한 각 스펙을 파싱해 일관된 리스트로 반환합니다.
    if not datasets_arg.strip():
        legacy_lang = infer_dataset_lang(fallback_name)
        return [
            {"name": fallback_name, "split": fallback_split, "lang": legacy_lang}
        ]
    specs: list[dict[str, str]] = []
    for raw in datasets_arg.split(","):
        if not raw.strip():
            continue
        specs.append(parse_dataset_spec(raw))
    if not specs:
        raise ValueError("유효한 데이터셋 스펙이 없습니다.")
    return specs


def format_tokens_for_log(tokens: list[str], max_items: int) -> str:
    # 로그 출력이 과도하게 길어지지 않도록 토큰 목록을 압축합니다.
    # 중간 생략이 필요한 경우 "..."를 넣어 앞/뒤 일부만 유지합니다.
    if max_items < 3:
        raise ValueError("max_items는 3 이상이어야 합니다.")
    if len(tokens) <= max_items:
        return ", ".join(tokens)
    head_count = max_items // 2
    tail_count = max_items - head_count - 1
    head = tokens[:head_count]
    tail = tokens[-tail_count:] if tail_count > 0 else []
    compressed = head + ["..."] + tail
    return ", ".join(compressed)


def reinitialize_all_embeddings(
    model: PreTrainedModel,
    std: float,
) -> None:
    # 전체 임베딩을 동일 분포로 재초기화해 새 토크나이저에 맞춥니다.
    # 입력/출력 임베딩을 모두 초기화해 학습 안정성을 확보합니다.
    embeddings = model.get_input_embeddings()
    if embeddings is None:
        raise ValueError("입력 임베딩을 찾을 수 없습니다.")
    embeddings.weight.data.normal_(mean=0.0, std=std)
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None:
        output_embeddings.weight.data.normal_(mean=0.0, std=std)


def resolve_embedding_std(model: PreTrainedModel) -> float:
    # 임베딩 분포에서 표준편차를 계산해 초기화 스케일로 사용합니다.
    embeddings = model.get_input_embeddings()
    if embeddings is None:
        raise ValueError("입력 임베딩을 찾을 수 없습니다.")
    std = float(embeddings.weight.data.std().item())
    return std if std > 0.0 else 0.02


def rebuild_tokenizer_from_lines(
    tokenizer: PreTrainedTokenizerBase,
    lines: list[str],
    vocab_size: int,
) -> PreTrainedTokenizerBase:
    # 1) 기존 토크나이저에서 특수 토큰을 수집합니다.
    # 2) 동일한 토크나이저 타입으로 새 토크나이저를 학습합니다.
    # 3) 결과 vocab 일부를 로깅해 구성 결과를 확인합니다.
    if not hasattr(tokenizer, "train_new_from_iterator"):
        raise ValueError("토크나이저 학습을 지원하지 않는 타입입니다.")
    size = vocab_size if vocab_size > 0 else len(tokenizer)
    special_tokens = list(tokenizer.all_special_tokens)
    if special_tokens:
        new_tokenizer = tokenizer.train_new_from_iterator(
            lines, vocab_size=size, new_special_tokens=special_tokens
        )
    else:
        new_tokenizer = tokenizer.train_new_from_iterator(lines, vocab_size=size)
    vocab = new_tokenizer.get_vocab()
    if not isinstance(vocab, dict):
        raise ValueError("재구성된 토크나이저 vocab이 dict가 아닙니다.")
    id_to_token = [""] * len(vocab)
    for token, idx in vocab.items():
        if 0 <= idx < len(vocab):
            id_to_token[idx] = token
    preview = format_tokens_for_log(
        [token for token in id_to_token if token], 20
    )
    logging.getLogger(__name__).info(
        "재구성된 토크나이저 vocab 미리보기: %s", preview
    )
    return new_tokenizer


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
        # 2) 대화 리스트 형태를 지원합니다.
        dialogue_line = build_dialogue_line(row)
        if dialogue_line:
            lines.append(dialogue_line)
            continue
        # 3) 단일 텍스트 필드를 가진 데이터셋은 그대로 사용합니다.
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
    short_question = row.get("short_question")
    short_answer = row.get("short_answer")
    prompt_text = row.get("prompt")
    utterance_text = row.get("utterance")
    input_text = row.get("input")

    # 질문 필드가 없으면 Instruct 스키마로 처리하지 않습니다.
    prompt = (
        instruction
        if isinstance(instruction, str)
        else (
            question
            if isinstance(question, str)
            else (
                short_question
                if isinstance(short_question, str)
                else prompt_text if isinstance(prompt_text, str) else None
            )
        )
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
            else (
                answer_text
                if isinstance(answer_text, str)
                else (
                    short_answer
                    if isinstance(short_answer, str)
                    else utterance_text if isinstance(utterance_text, str) else None
                )
            )
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


def build_dialogue_line(row: dict[str, object]) -> str | None:
    # 대화 리스트를 하나의 문장으로 합쳐 학습 입력으로 사용합니다.
    # utterances/dialog/dialogue 등 일반적으로 쓰이는 필드를 순서대로 확인합니다.
    utterances = row.get("utterances")
    dialog = row.get("dialog")
    dialogue = row.get("dialogue")
    candidates = [utterances, dialog, dialogue]
    for candidate in candidates:
        if not isinstance(candidate, list):
            continue
        parts: list[str] = []
        for item in candidate:
            if not isinstance(item, str):
                continue
            cleaned = sanitize_text(item)
            if cleaned:
                parts.append(cleaned)
        if parts:
            return " ".join(parts)
    return None


def write_training_lines(path: Path, lines: list[str]) -> None:
    # 1) 출력 경로 상위 디렉터리를 생성합니다.
    # 2) 한 줄씩 저장해 이후 학습에서 재사용할 수 있게 합니다.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def prepare_datasets(
    train_path: Path,
    dataset_specs: list[dict[str, str]],
    max_samples: int,
) -> list[str]:
    # 1) 데이터셋별로 학습 라인을 생성해 누적합니다.
    # 2) 결과를 파일로 저장해 재실행 비용을 줄입니다.
    all_lines: list[str] = []
    for spec in dataset_specs:
        dataset_name = spec["name"]
        split_name = spec["split"]
        print(f"데이터셋 로딩: {dataset_name} ({split_name})")
        dataset = load_hf_dataset(dataset_name, split_name)
        raw_rows = cast(list[dict[str, object]], list(dataset))
        lines = build_training_lines(raw_rows, max_samples)
        all_lines.extend(lines)
    if not all_lines:
        raise ValueError("여러 데이터셋에서 유효한 학습 라인을 만들지 못했습니다.")
    write_training_lines(train_path, all_lines)
    return all_lines


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


def resolve_output_dir(
    output_dir_arg: str,
    output_root_arg: str,
    run_name_arg: str,
) -> Path:
    # 1) 사용자가 output_dir을 지정했으면 그 값을 그대로 사용합니다.
    # 2) 지정하지 않았으면 output_root/run_name을 합쳐 경로를 만듭니다.
    # 3) 결과 경로는 Path로 변환해 이후 로직을 단순화합니다.
    if output_dir_arg:
        return Path(output_dir_arg)
    return Path(output_root_arg) / run_name_arg


def update_latest_symlink(
    output_root: Path,
    run_name: str,
    latest_name: str,
) -> None:
    # 1) output_root/run_name을 최신 대상 경로로 확정합니다.
    # 2) 같은 위치에 latest_name 심볼릭 링크를 만들어 최신 경로를 가리키게 합니다.
    # 3) 기존 링크가 있으면 제거하고 새 링크로 교체합니다.
    target_path = output_root / run_name
    link_path = output_root / latest_name
    output_root.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    os.symlink(target_path.as_posix(), link_path.as_posix())


def main() -> None:
    args = parse_args()

    train_path = Path(args.train_file)
    output_dir = resolve_output_dir(
        output_dir_arg=args.output_dir,
        output_root_arg=args.output_root,
        run_name_arg=args.run_name,
    )
    device = resolve_device(args.device)

    # 기본 로거는 INFO까지 출력하도록 설정합니다.
    # 이미 핸들러가 있으면 덮어쓰지 않아 외부 설정을 존중합니다.
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO)

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

    dataset_specs = resolve_dataset_specs(
        datasets_arg=args.datasets,
        fallback_name=args.dataset,
        fallback_split=args.dataset_split,
    )
    needs_prepare = (
        args.force_prepare or not train_path.exists() or train_path.stat().st_size == 0
    )
    if needs_prepare:
        print("학습 파일이 없어 데이터셋을 다운로드합니다.")
        lines = prepare_datasets(
            train_path=train_path,
            dataset_specs=dataset_specs,
            max_samples=args.dataset_max_samples,
        )
    else:
        # 이미 준비된 파일이 있으면 그대로 사용합니다.
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
    if resume and args.tokenizer_rebuild:
        # 재개 학습에서는 기존 토크나이저를 그대로 유지합니다.
        print("재개 학습에서는 토크나이저 재구성을 건너뜁니다.")
    elif args.tokenizer_rebuild:
        print("데이터셋 기반 토크나이저 재구성을 시작합니다.")
        std = resolve_embedding_std(model)
        tokenizer = rebuild_tokenizer_from_lines(
            tokenizer=tokenizer,
            lines=lines,
            vocab_size=args.tokenizer_vocab_size,
        )
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        reinitialize_all_embeddings(model, std)
        print(f"토크나이저 재구성 완료: vocab={len(tokenizer)}")
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

    if args.latest_link and not args.output_dir:
        update_latest_symlink(
            output_root=Path(args.output_root),
            run_name=args.run_name,
            latest_name=args.latest_name,
        )
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
