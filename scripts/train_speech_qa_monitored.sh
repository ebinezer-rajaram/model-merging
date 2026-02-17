#!/usr/bin/env bash
set -euo pipefail

CONFIG="speech_qa.yaml"
INTERVAL=15
LOG_FILE=""
PID_FILE="logs/speech_qa_train.pid"
ATTACH_ONLY=0
PID_OVERRIDE=""

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Launch and monitor Speech-QA training with live status updates.

Options:
  --config NAME           Config filename under configs/ (default: speech_qa.yaml)
  --interval SEC          Monitor refresh interval in seconds (default: 15)
  --log-file PATH         Explicit log file path (default: logs/speech_qa_<timestamp>.log)
  --pid-file PATH         PID file path (default: logs/speech_qa_train.pid)
  --attach                Attach monitor to existing PID from --pid-file (do not launch)
  --pid PID               Attach monitor to explicit PID (do not launch)
  -h, --help              Show this help

Examples:
  $(basename "$0")
  $(basename "$0") --config speech_qa.yaml --interval 10
  $(basename "$0") --attach
  $(basename "$0") --pid 12345
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --interval)
      INTERVAL="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --pid-file)
      PID_FILE="$2"
      shift 2
      ;;
    --attach)
      ATTACH_ONLY=1
      shift
      ;;
    --pid)
      PID_OVERRIDE="$2"
      ATTACH_ONLY=1
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

mkdir -p logs

if [[ -n "$PID_OVERRIDE" ]]; then
  PID="$PID_OVERRIDE"
elif [[ "$ATTACH_ONLY" -eq 1 ]]; then
  if [[ ! -f "$PID_FILE" ]]; then
    echo "PID file not found: $PID_FILE"
    exit 1
  fi
  PID="$(cat "$PID_FILE")"
else
  if [[ -z "$LOG_FILE" ]]; then
    TS="$(date +%Y%m%d_%H%M%S)"
    LOG_FILE="logs/speech_qa_${TS}.log"
  fi

  CMD=("venv/bin/python" "main.py" "train" "--task" "speech_qa" "--config" "$CONFIG")

  echo "Launching: ${CMD[*]}"
  echo "Log file: $LOG_FILE"

  nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
  PID=$!
  echo "$PID" > "$PID_FILE"
  echo "Started Speech-QA training. PID=$PID"
fi

if ! kill -0 "$PID" >/dev/null 2>&1; then
  echo "PID $PID is not running."
  exit 1
fi

if [[ -z "$LOG_FILE" ]]; then
  LOG_FILE=$(ls -1t logs/speech_qa_*.log 2>/dev/null | head -n1 || true)
fi

echo "Monitoring PID=$PID"
if [[ -n "$LOG_FILE" ]]; then
  echo "Reading log: $LOG_FILE"
fi
echo "Press Ctrl+C to stop monitoring (process keeps running)."

stop_monitor=0
trap 'stop_monitor=1' INT TERM

while [[ "$stop_monitor" -eq 0 ]]; do
  if ! kill -0 "$PID" >/dev/null 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Process $PID has exited."
    break
  fi

  PROC_LINE=$(ps -p "$PID" -o pid=,stat=,etime=,%cpu=,%mem=,cmd=)

  LAST_LOSS=""
  LAST_EVAL=""
  if [[ -n "$LOG_FILE" && -f "$LOG_FILE" ]]; then
    LAST_LOSS=$(grep -F "{'loss':" "$LOG_FILE" | tail -n1 || true)
    LAST_EVAL=$(grep -F "'eval_f1':" "$LOG_FILE" | tail -n1 || true)
  fi

  CONCAT_COUNT=0
  if [[ -d "data/datasets/Spoken-SQuAD/wav/_concatenated" ]]; then
    CONCAT_COUNT=$(find data/datasets/Spoken-SQuAD/wav/_concatenated -type f -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')
  fi

  echo ""
  echo "[$(date '+%Y-%m-%d %H:%M:%S')]"
  echo "  proc: $PROC_LINE"
  echo "  concat_wavs: $CONCAT_COUNT"
  if [[ -n "$LAST_LOSS" ]]; then
    echo "  last_loss: $LAST_LOSS"
  fi
  if [[ -n "$LAST_EVAL" ]]; then
    echo "  last_eval: $LAST_EVAL"
  fi

  sleep "$INTERVAL"
done

echo "Monitor stopped."
if kill -0 "$PID" >/dev/null 2>&1; then
  echo "Process $PID is still running."
fi
