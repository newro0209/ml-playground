#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=./src python -m ml_playground.smollm2_finetune \
  --train-file data/processed/korean_train.txt \
  --output-dir checkpoints/smollm2-ko \
  --epochs 1 \
  --batch-size 2 \
  --max-length 256 \
  --dataset ohilikeit/empathetic_dialogues_mutli_turn_ko
