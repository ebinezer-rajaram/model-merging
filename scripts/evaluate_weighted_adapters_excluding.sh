#!/usr/bin/env bash
set -euo pipefail

SPLIT="${SPLIT:-test}"
BASE_DIR="artifacts/merged/weighted"
SKIP_PAIRS=("asr_intent" "asr_langid" "speaker_id_asr" "asr_kws")

if [[ ! -d "${BASE_DIR}" ]]; then
  echo "Weighted merge directory not found: ${BASE_DIR}" >&2
  exit 1
fi

echo "Evaluating weighted merged adapters (split=${SPLIT})"
echo "Skipping pairs: ${SKIP_PAIRS[*]}"

for pair_dir in "${BASE_DIR}"/*; do
  [[ -d "${pair_dir}" ]] || continue
  pair_slug="$(basename "${pair_dir}")"
  for skip_slug in "${SKIP_PAIRS[@]}"; do
    if [[ "${pair_slug}" == "${skip_slug}" ]]; then
      continue 2
    fi
  done

  for lambda_dir in "${pair_dir}"/*; do
    [[ -d "${lambda_dir}" ]] || continue

    if [[ ! -e "${lambda_dir}/best" && ! -e "${lambda_dir}/latest" && ! -d "${lambda_dir}/runs" ]]; then
      continue
    fi

    echo ""
    echo "==> ${lambda_dir}"
    python main.py evaluate-merged --adapter-path "${lambda_dir}" --split "${SPLIT}"
  done
done
