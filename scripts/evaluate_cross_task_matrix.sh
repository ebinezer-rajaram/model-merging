#!/bin/bash
# Cross-task evaluation matrix for all 6 tasks
# This evaluates each adapter and base model on all tasks

set -e

# Define the 6 tasks with trained adapters
TASKS="asr emotion intent speaker_id kws langid"

echo "=================================="
echo "Cross-Task Evaluation Matrix"
echo "=================================="
echo "Tasks: $TASKS"
echo "This will evaluate:"
echo "  1. Base model on all 6 tasks"
echo "  2. Each of 6 adapters on all 6 tasks"
echo "Total: 7 models × 6 tasks = 42 evaluations"
echo "=================================="
echo ""

# Run the cross-task evaluation
python -m experiments.evaluate_all_cross_task \
    --tasks $TASKS \
    --adapters $TASKS \
    --split test \
    --use-cache \
    --confusion-matrix \
    --output-summary artifacts/evaluation/cross_task_matrix_all.json

echo ""
echo "=================================="
echo "Evaluation Complete!"
echo "=================================="
echo "Results saved to: artifacts/evaluation/cross_task_matrix_all.json"
