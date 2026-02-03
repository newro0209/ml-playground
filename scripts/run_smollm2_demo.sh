#!/usr/bin/env bash
set -euo pipefail

# 기본/미세조정 모델을 같은 프롬프트로 비교합니다.
BASE_CHECKPOINT="${BASE_CHECKPOINT:-HuggingFaceTB/SmolLM2-135M-Instruct}"
FINETUNED_CHECKPOINT="${FINETUNED_CHECKPOINT:-checkpoints/smollm2-ko-instruct}"
PROMPTS_FILE="${PROMPTS_FILE:-data/prompts/ko_instruct_eval.txt}"

echo "=== 기본 모델 ==="
PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}" \
python -m ml_playground.smollm2_demo \
  --checkpoint "$BASE_CHECKPOINT" \
  --prompts-file "$PROMPTS_FILE" \
  --max-new-tokens 256

if [[ -d "$FINETUNED_CHECKPOINT" || -f "$FINETUNED_CHECKPOINT" ]]; then
  echo "=== 미세조정 모델 ==="
  PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}" \
  python -m ml_playground.smollm2_demo \
    --checkpoint "$FINETUNED_CHECKPOINT" \
    --prompts-file "$PROMPTS_FILE" \
    --max-new-tokens 256
else
  echo "미세조정 체크포인트가 없어 비교를 건너뜁니다: $FINETUNED_CHECKPOINT"
fi
