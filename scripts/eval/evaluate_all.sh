#!/bin/bash
# Evaluate all adapters on all tasks (cross-task evaluation matrix)

set -e

# Default values
SPLIT="test"
BATCH_SIZE=""
TASKS=""
ADAPTERS=""
USE_CACHE=""
SKIP_BASE=""
CONFUSION_MATRIX="--confusion-matrix"
OUTPUT_SUMMARY=""

# Parse command line arguments
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
        --tasks)
            shift
            TASKS="--tasks"
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                TASKS="$TASKS $1"
                shift
            done
            ;;
        --adapters)
            shift
            ADAPTERS="--adapters"
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                ADAPTERS="$ADAPTERS $1"
                shift
            done
            ;;
        --use-cache)
            USE_CACHE="--use-cache"
            shift
            ;;
        --skip-base)
            SKIP_BASE="--skip-base"
            shift
            ;;
        --no-confusion-matrix)
            CONFUSION_MATRIX="--no-confusion-matrix"
            shift
            ;;
        --output-summary)
            OUTPUT_SUMMARY="--output-summary $2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./evaluate_all.sh [OPTIONS]"
            echo ""
            echo "Evaluate all adapters on all tasks in a cross-task evaluation matrix."
            echo ""
            echo "Options:"
            echo "  --split SPLIT              Dataset split to evaluate (default: test)"
            echo "  --batch-size SIZE          Per-device evaluation batch size"
            echo "  --tasks TASK1 TASK2 ...    List of tasks to evaluate on (default: all)"
            echo "  --adapters TASK1 TASK2 ... List of adapter tasks to evaluate (default: all)"
            echo "  --use-cache                Reuse cached base model metrics"
            echo "  --skip-base                Skip base model evaluation"
            echo "  --no-confusion-matrix      Disable confusion matrix generation"
            echo "  --output-summary PATH      Path to save summary JSON"
            echo "  --help, -h                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Evaluate all adapters on all tasks (test split)"
            echo "  ./evaluate_all.sh"
            echo ""
            echo "  # Evaluate only emotion and intent adapters on all tasks"
            echo "  ./evaluate_all.sh --adapters emotion intent"
            echo ""
            echo "  # Evaluate all adapters on only asr and emotion tasks"
            echo "  ./evaluate_all.sh --tasks asr emotion"
            echo ""
            echo "  # Use validation split and custom batch size"
            echo "  ./evaluate_all.sh --split validation --batch-size 8"
            echo ""
            echo "  # Skip base model and use cached metrics"
            echo "  ./evaluate_all.sh --skip-base --use-cache"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Cross-Task Evaluation Matrix"
echo "========================================"
echo "Split: $SPLIT"
echo "========================================"
echo ""

# Run the evaluation script
python experiments/evaluate_all_cross_task.py \
    --split "$SPLIT" \
    $BATCH_SIZE \
    $TASKS \
    $ADAPTERS \
    $USE_CACHE \
    $SKIP_BASE \
    $CONFUSION_MATRIX \
    $OUTPUT_SUMMARY

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "========================================"
