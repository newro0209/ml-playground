#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=./src python -m ml_playground.smollm2_finetune \
  --train-file data/processed/ko_instruct_train.txt \
  --output-dir checkpoints/smollm2-ko-instruct \
  --epochs 8 \
  --lr 1e-4 \
  --batch-size 32 \
  --max-length 384 \
  --dataset beomi/KoAlpaca-RealQA
