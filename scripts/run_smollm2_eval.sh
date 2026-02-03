#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=./src python -m ml_playground.smollm2_eval \
  --base-checkpoint HuggingFaceTB/SmolLM2-135M-Instruct \
  --finetuned-checkpoint checkpoints/smollm2-ko-instruct
