"""SmolLM2 체크포인트를 GGUF로 변환하는 스크립트."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmolLM2 GGUF 변환")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/smollm2-ko",
        help="변환할 체크포인트 경로",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="checkpoints/smollm2-ko.gguf",
        help="출력 GGUF 경로",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        type=str,
        default="",
        help="llama.cpp 경로(미지정 시 환경 변수 LLAMA_CPP_DIR 사용)",
    )
    parser.add_argument(
        "--outtype",
        type=str,
        default="",
        help="GGUF 출력 타입(예: q8_0). 미지정 시 기본값 사용",
    )
    return parser.parse_args()


def resolve_llama_cpp_dir(arg_value: str) -> Path:
    if arg_value:
        return Path(arg_value)
    env_value = os.environ.get("LLAMA_CPP_DIR", "")
    if env_value:
        return Path(env_value)
    raise FileNotFoundError("LLAMA_CPP_DIR 환경 변수 또는 --llama-cpp-dir이 필요합니다.")


def build_convert_command(
    script_path: Path,
    checkpoint_dir: Path,
    output_path: Path,
    outtype: str,
) -> list[str]:
    command = [
        "python",
        script_path.as_posix(),
        "--outdir",
        output_path.parent.as_posix(),
        "--outfile",
        output_path.stem,
        checkpoint_dir.as_posix(),
    ]
    if outtype:
        command.extend(["--outtype", outtype])
    return command


def run_convert(command: list[str]) -> None:
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError("GGUF 변환에 실패했습니다.")


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    output_path = Path(args.output_path)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"체크포인트 경로가 없습니다: {checkpoint_dir}")

    llama_cpp_dir = resolve_llama_cpp_dir(args.llama_cpp_dir)
    convert_script = llama_cpp_dir / "convert-hf-to-gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"변환 스크립트를 찾을 수 없습니다: {convert_script}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_convert_command(convert_script, checkpoint_dir, output_path, args.outtype)
    print("변환 명령:", " ".join(command))
    run_convert(command)
    print(f"GGUF 변환 완료: {output_path}")


if __name__ == "__main__":
    main()
