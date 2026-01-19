#!/usr/bin/env bash
set -euo pipefail

SPLIT="${SPLIT:-test}"
BASE_DIR="artifacts/merged/weighted"

if [[ ! -d "${BASE_DIR}" ]]; then
  echo "Weighted merge directory not found: ${BASE_DIR}" >&2
  exit 1
fi

echo "Evaluating weighted merged adapters (split=${SPLIT})"

for pair_dir in "${BASE_DIR}"/*; do
  [[ -d "${pair_dir}" ]] || continue
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
