#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=./src python -m ml_playground.smollm2.finetune \
  --train-file data/processed/ko_instruct_train.txt \
  --output-root checkpoints/smollm2 \
  --run-name ko-instruct \
  --latest-link \
  --epochs 3 \
  --lr 1e-4 \
  --batch-size 32 \
  --max-length 256 \
  --tokenizer-rebuild \
  --tokenizer-vocab-size 0 \
  --datasets beomi/KoAlpaca-RealQA:train:ko,Ammad1Ali/Korean-conversational-dataset:train:ko,Ahren09/empathetic_dialogues:train:en

PYTHONPATH=./src python -m ml_playground.smollm2_gguf \
  --checkpoint-dir checkpoints/smollm2/ko-instruct \
  --output-path checkpoints/smollm2/ko-instruct/smollm2-ko-instruct.gguf \
  --llama-cpp-dir "$LLAMA_CPP_DIR"
