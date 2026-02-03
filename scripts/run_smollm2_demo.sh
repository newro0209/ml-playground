#!/usr/bin/env bash
set -euo pipefail

# SmolLM2 135M Base 간단 실행
PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}" \
python -m ml_playground.smollm2_demo --prompt "Gravity is" --max-new-tokens 64
