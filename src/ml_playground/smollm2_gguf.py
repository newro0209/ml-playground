"""SmolLM2 체크포인트를 GGUF로 변환하는 스크립트."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    # 커맨드라인 인자를 모아서 스크립트 동작을 명시적으로 제어합니다.
    # 실행 환경마다 경로가 다를 수 있어 기본값을 제공하되 덮어쓰기를 허용합니다.
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
    # 인자로 경로가 들어오면 가장 명시적인 설정이므로 우선 사용합니다.
    if arg_value:
        return Path(arg_value)
    # 인자가 없을 때는 환경 변수로 경로를 주입할 수 있도록 허용합니다.
    env_value = os.environ.get("LLAMA_CPP_DIR", "")
    if env_value:
        return Path(env_value)
    # 두 경로 모두 없으면 변환 스크립트를 찾을 수 없으므로 즉시 실패시킵니다.
    raise FileNotFoundError("LLAMA_CPP_DIR 환경 변수 또는 --llama-cpp-dir이 필요합니다.")


def build_convert_command(
    script_path: Path,
    checkpoint_dir: Path,
    output_path: Path,
    outtype: str,
) -> list[str]:
    # llama.cpp 변환 스크립트의 최신 사용법에 맞춰 명령을 구성합니다.
    # --outfile에는 전체 경로를 전달해야 하므로 outdir/outfile 분리를 제거합니다.
    command = [
        "python",
        script_path.as_posix(),
        "--outfile",
        output_path.as_posix(),
        checkpoint_dir.as_posix(),
    ]
    # 출력 타입은 선택 사항이므로 값이 있을 때만 추가합니다.
    if outtype:
        command.extend(["--outtype", outtype])
    return command


def run_convert(command: list[str]) -> None:
    # 외부 스크립트 실행 결과를 확인하여 실패 시 즉시 알립니다.
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError("GGUF 변환에 실패했습니다.")


def main() -> None:
    # 1) 인자 해석으로 경로/옵션을 확보합니다.
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    output_path = Path(args.output_path)

    # 2) 입력 체크포인트 경로가 유효한지 먼저 검증합니다.
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"체크포인트 경로가 없습니다: {checkpoint_dir}")

    # 3) 변환 스크립트 위치를 결정하고 존재 여부를 확인합니다.
    llama_cpp_dir = resolve_llama_cpp_dir(args.llama_cpp_dir)
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"변환 스크립트를 찾을 수 없습니다: {convert_script}")

    # 4) 출력 디렉터리를 준비하고 변환 명령을 구성합니다.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_convert_command(convert_script, checkpoint_dir, output_path, args.outtype)
    print("변환 명령:", " ".join(command))
    # 5) 변환을 실행하고 성공 메시지를 출력합니다.
    run_convert(command)
    print(f"GGUF 변환 완료: {output_path}")


if __name__ == "__main__":
    main()
