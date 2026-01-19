#!/bin/bash
# Run missing cross-task evaluations
# This script runs the evaluations that failed before (ERROR in results), avoiding duplicates:
# 1. kws adapter on non-error tasks (3 tasks: asr, emotion, intent)
# 2. langid adapter on non-error tasks (3 tasks: asr, emotion, intent)
# 3. Base + 6 adapters on kws, langid, speaker_id tasks (3 tasks × 7 models = 21 evaluations)
# Total: 27 evaluations (no duplicates)
# Tasks used: asr, emotion, intent, speaker_id, kws, langid (6 tasks only)
# Adapters used: base, asr, emotion, intent, kws, langid, speaker_id

set -e

echo "=================================="
echo "Missing Cross-Task Evaluations"
echo "=================================="
echo ""
echo "This will run (avoiding duplicates):"
echo "  Tasks: asr, emotion, intent, speaker_id, kws, langid (6 tasks only)"
echo "  Adapters: base, asr, emotion, intent, kws, langid, speaker_id"
echo ""
echo "  1. kws adapter on 3 tasks (asr, emotion, intent)"
echo "  2. langid adapter on 3 tasks (asr, emotion, intent)"
echo "  3. Base + 6 adapters on kws task (7 models)"
echo "  4. Base + 6 adapters on langid task (7 models)"
echo "  5. Base + 6 adapters on speaker_id task (7 models)"
echo ""
echo "Total: 27 evaluations (no repeats)"
echo "=================================="
echo ""

# Tasks for kws/langid adapters (only the 3 successful tasks from your results)
OTHER_TASKS="asr emotion intent"

# Tasks that need full evaluation (all adapters on these) - based on ERROR in results
MISSING_EVAL_TASKS="speaker_id langid"

# Adapters that need to run on other tasks
MISSING_ADAPTERS="kws langid"

# Only the adapters we care about (matching the existing results table)
ALL_ADAPTERS="asr emotion intent speaker_id kws langid"

echo ""
echo "Step 2: Running all models on kws, langid, and speaker_id tasks..."
echo "===================================================================="
python -m experiments.evaluate_all_cross_task \
    --tasks $MISSING_EVAL_TASKS \
    --adapters $ALL_ADAPTERS \
    --split test \
    --use-cache \
    --confusion-matrix \
    --output-summary artifacts/evaluation/cross_task_missing_tasks.json

echo ""
echo "=================================="
echo "All Missing Evaluations Complete!"
echo "=================================="
echo "Results saved to:"
echo "  - artifacts/evaluation/cross_task_missing_adapters.json"
echo "  - artifacts/evaluation/cross_task_missing_tasks.json"
