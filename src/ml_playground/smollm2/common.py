"""SmolLM2 스크립트 공용 유틸리티."""

from __future__ import annotations

import torch


def resolve_device(device: str) -> torch.device:
    # auto 선택 시 사용 가능한 장치를 우선합니다.
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
