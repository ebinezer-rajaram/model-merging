#!/bin/bash
# Train emotion and ASR tasks sequentially, continue on failure

tasks=("emotion" "asr")

for task in "${tasks[@]}"; do
    echo "Starting training for $task at $(date)"
    python3 main.py train --task $task || echo "Training for $task failed, continuing..."
    echo "Completed training for $task at $(date)"
    echo "----------------------------------------"
done

echo "All training tasks completed at $(date)"
