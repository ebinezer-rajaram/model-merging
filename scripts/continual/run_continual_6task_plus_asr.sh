#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

SOURCE_RUN="artifacts/merged/weighted_delta_n/emotion_intent_kws_langid_speaker_ver_vocalsound/runs/run_supermerge_layer_wise_20260310_203741"
SWEEP_CONFIG="configs/merge/continual/merge_continual_6task_materialized_plus_asr.yaml"
ENERGY_THRESHOLD="0.99"

STAMP="$(date +%Y%m%d_%H%M%S)"
MATERIALIZED_DIR="artifacts/continual/materialized/6task_supermerge_${STAMP}"
SWEEP_DIR="artifacts/merged/continual/6task_plus_asr/sweeps"

python main.py materialize-merged-artifact \
  --merged-run-path "${SOURCE_RUN}" \
  --output "${MATERIALIZED_DIR}" \
  --energy-threshold "${ENERGY_THRESHOLD}" \
  --store-dtype float16

python main.py merge-sweep \
  --config "${SWEEP_CONFIG}" \
  --adapters "${MATERIALIZED_DIR}" asr \
  --split test \
  --eval-tasks emotion intent kws langid speaker_ver vocalsound asr speech_qa \
  --output-dir "${SWEEP_DIR}"

LATEST_SUMMARY="$(ls -t "${SWEEP_DIR}"/sweep_*.json | head -n 1)"
BEST_ARTIFACT="$(python - "${LATEST_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
data = json.loads(summary_path.read_text())
best_index = data.get("best_index")
runs = data.get("runs") or []
artifact_dir = ""
if isinstance(best_index, int) and 0 <= best_index < len(runs):
    continual = runs[best_index].get("continual") or {}
    artifact_dir = str(continual.get("artifact_dir", ""))
print(artifact_dir)
PY
)"

printf '\nDone.\nSweep config: %s\nMaterialized artifact: %s\nSweep summary: %s\nBest merged artifact: %s\n' \
  "${SWEEP_CONFIG}" "${MATERIALIZED_DIR}" "${LATEST_SUMMARY}" "${BEST_ARTIFACT}"
