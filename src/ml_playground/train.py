"""학습 엔트리포인트(스켈레톤)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class TrainConfig:
    """학습 설정 기본값."""

    seed: int = 42
    epochs: int = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="학습 실행")
    parser.add_argument("--epochs", type=int, default=3, help="학습 에폭 수")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(epochs=args.epochs)

    # TODO: 데이터 로딩, 모델 구성, 학습 루프 구현
    print(f"학습 시작: epochs={cfg.epochs}")


if __name__ == "__main__":
    main()
