#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=./src python -m ml_playground.smollm2_finetune \
  --train-file data/processed/ko_instruct_train.txt \
  --output-dir checkpoints/smollm2-ko-instruct \
  --epochs 1 \
  --batch-size 32 \
  --max-length 256 \
  --dataset beomi/KoAlpaca-RealQA
