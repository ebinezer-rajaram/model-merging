#!/bin/sh
set -eu

for pair in "asr intent" "asr langid" "speaker_id asr"; do
  pair_slug=$(printf "%s" "${pair}" | tr ' ' '_')
  for lambda in 0.1 0.3 0.5 0.7 0.9; do
    output_dir="artifacts/merged/weighted/${pair_slug}/${lambda}"
    mkdir -p "${output_dir}"
    echo "Running weighted merge for adapters: ${pair} (lambda=${lambda})"
    python main.py merge --adapters ${pair} --method weighted --lambda "${lambda}" --output "${output_dir}"
  done
done
