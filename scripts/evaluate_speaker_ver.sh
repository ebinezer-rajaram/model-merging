#!/bin/bash
# Evaluate speaker verification in two passes using evaluate_task.py:
# 1) speaker_ver adapter on every task
# 2) every best adapter (and base model) on speaker_ver

set -u

export PYTHONPATH="${PYTHONPATH:-.}"

run_eval() {
    local label="$1"
    shift
    echo ""
    echo "==> ${label}"
    if ! "$@"; then
        echo "⚠️  Evaluation failed: ${label} (continuing)"
        return 0
    fi
}

SPLIT="test"
BATCH_SIZE=""
CONFUSION_MATRIX="--confusion-matrix"
OUTPUT_DIR=""
INCLUDE_BASE_ON_ALL=""

TASKS="asr emotion intent speaker_id kws langid"
TASKS1="kws langid speaker_ver"

while [[ $# -gt 0 ]]; do
    case $1 in
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="--batch-size $2"
            shift 2
            ;;
        --no-confusion-matrix)
            CONFUSION_MATRIX="--no-confusion-matrix"
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --tasks)
            shift
            TASKS=""
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                TASKS="${TASKS} $1"
                shift
            done
            ;;
        --include-base-on-all)
            INCLUDE_BASE_ON_ALL="yes"
            shift
            ;;
        --help|-h)
            echo "Usage: ./evaluate_speaker_ver.sh [OPTIONS]"
            echo ""
            echo "Evaluate speaker verification in two passes:"
            echo "  1) speaker_ver adapter on every task"
            echo "  2) every best adapter (and base model) on speaker_ver"
            echo ""
            echo "Options:"
            echo "  --split SPLIT              Dataset split to evaluate (default: test)"
            echo "  --batch-size SIZE          Per-device evaluation batch size"
            echo "  --no-confusion-matrix      Disable confusion matrix generation"
            echo "  --output-dir DIR           Save summary JSONs into this directory"
            echo "  --tasks TASK1 TASK2 ...    Tasks to evaluate (default: all tasks)"
            echo "  --include-base-on-all      Also evaluate the base model on all tasks in pass 1"
            echo "  --help, -h                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Evaluate speaker_ver adapter across all tasks"
            echo "  # and all adapters + base on speaker_ver (test split)"
            echo "  ./evaluate_speaker_ver.sh"
            echo ""
            echo "  # Use validation split and custom batch size"
            echo "  ./evaluate_speaker_ver.sh --split validation --batch-size 8"
            echo ""
            echo "  # Save summaries to a directory"
            echo "  ./evaluate_speaker_ver.sh --output-dir artifacts/evaluation"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

OUTPUT_SPEAKER_VER_ADAPTER=""
OUTPUT_ALL_ADAPTERS_ON_SPKR_VER=""
if [[ -n "${OUTPUT_DIR}" ]]; then
    TS="$(date +%Y%m%d_%H%M%S)"
    OUTPUT_SPEAKER_VER_ADAPTER="${OUTPUT_DIR}/speaker_ver_adapter_on_all_tasks_${SPLIT}_${TS}"
    OUTPUT_ALL_ADAPTERS_ON_SPKR_VER="${OUTPUT_DIR}/all_adapters_on_speaker_ver_${SPLIT}_${TS}"
fi

echo "========================================"
echo "Speaker Verification Evaluation"
echo "========================================"
echo "Split: $SPLIT"
echo "========================================"
echo ""

echo "Step 1: speaker_ver adapter on every task"
echo "----------------------------------------"
for task in $TASKS1; do
    SAVE_JSON=""
    if [[ -n "${OUTPUT_SPEAKER_VER_ADAPTER}" ]]; then
        SAVE_JSON="--save-json ${OUTPUT_SPEAKER_VER_ADAPTER}_${task}.json"
    fi
    run_eval "speaker_ver adapter on ${task}" \
        python experiments/evaluate_task.py \
            --task "${task}" \
            --adapter speaker_ver \
            --split "${SPLIT}" \
            $BATCH_SIZE \
            $CONFUSION_MATRIX \
            $SAVE_JSON

    if [[ -n "${INCLUDE_BASE_ON_ALL}" ]]; then
        SAVE_JSON=""
        if [[ -n "${OUTPUT_SPEAKER_VER_ADAPTER}" ]]; then
            SAVE_JSON="--save-json ${OUTPUT_SPEAKER_VER_ADAPTER}_${task}_base.json"
        fi
        run_eval "base model on ${task}" \
            python experiments/evaluate_task.py \
                --task "${task}" \
                --split "${SPLIT}" \
                $BATCH_SIZE \
                $CONFUSION_MATRIX \
                $SAVE_JSON
    fi
done

echo ""
echo "Step 2: every best adapter + base model on speaker_ver"
echo "-----------------------------------------------------"
SAVE_JSON=""
if [[ -n "${OUTPUT_ALL_ADAPTERS_ON_SPKR_VER}" ]]; then
    SAVE_JSON="--save-json ${OUTPUT_ALL_ADAPTERS_ON_SPKR_VER}_base.json"
fi
run_eval "base model on speaker_ver" \
    python experiments/evaluate_task.py \
        --task speaker_ver \
        --split "${SPLIT}" \
        $BATCH_SIZE \
        $CONFUSION_MATRIX \
        $SAVE_JSON

for adapter in $TASKS; do
    SAVE_JSON=""
    if [[ -n "${OUTPUT_ALL_ADAPTERS_ON_SPKR_VER}" ]]; then
        SAVE_JSON="--save-json ${OUTPUT_ALL_ADAPTERS_ON_SPKR_VER}_${adapter}.json"
    fi
    run_eval "${adapter} adapter on speaker_ver" \
        python experiments/evaluate_task.py \
            --task speaker_ver \
            --adapter "${adapter}" \
            --split "${SPLIT}" \
            $BATCH_SIZE \
            $CONFUSION_MATRIX \
            $SAVE_JSON
done

echo ""
echo "========================================"
echo "Speaker Verification evaluation complete!"
echo "========================================"
