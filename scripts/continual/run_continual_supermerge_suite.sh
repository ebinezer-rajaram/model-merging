#!/usr/bin/env bash
set -euo pipefail

STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="logs"
LOG_PATH="${LOG_DIR}/continual_suite_supermerge_${STAMP}.log"

mkdir -p "${LOG_DIR}"

source venv/bin/activate

nohup python3 scripts/continual/run_continual_suite.py \
  --suite-config configs/continual/suite_paths_supermerge.yaml \
  --mode merge \
  --execute \
  > "${LOG_PATH}" 2>&1 &

echo "started pid=$!"
echo "log=${LOG_PATH}"
