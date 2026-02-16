#!/usr/bin/env bash
set -euo pipefail

METHOD="weighted_delta"
SPLIT="test"
LAMBDAS=(0.2 0.4 0.6 0.8)

# Task pairs to evaluate
TASK_PAIRS=(
  "intent langid"
  "asr intent"
  "kws langid"
  "asr speaker_ver"
  "emotion speaker_ver"
)

for pair in "${TASK_PAIRS[@]}"; do
  read -r TASK1 TASK2 <<< "${pair}"
  for LAMBDA in "${LAMBDAS[@]}"; do
    echo "Evaluating ${TASK1}+${TASK2} @ lambda=${LAMBDA}"
    python main.py evaluate-merged \
      --method "${METHOD}" \
      --tasks "${TASK1}" "${TASK2}" \
      --lambda "${LAMBDA}" \
      --eval-tasks "${TASK1}" "${TASK2}" \
      --split "${SPLIT}"
  done
done
