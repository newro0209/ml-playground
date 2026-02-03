#!/usr/bin/env bash
set -euo pipefail

: "${LLAMA_CPP_DIR:?LLAMA_CPP_DIR 환경 변수를 설정하세요}"

PYTHONPATH=./src python -m ml_playground.smollm2_gguf \
  --checkpoint-dir checkpoints/smollm2-ko \
  --output-path checkpoints/smollm2-ko.gguf \
  --llama-cpp-dir "$LLAMA_CPP_DIR"
