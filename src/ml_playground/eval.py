"""평가 엔트리포인트(스켈레톤)."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="평가 실행")
    parser.add_argument("--ckpt", type=str, required=True, help="체크포인트 경로")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # TODO: 체크포인트 로딩, 평가 루프 구현
    print(f"평가 시작: ckpt={args.ckpt}")


if __name__ == "__main__":
    main()
