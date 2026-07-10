#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

venv/bin/python main.py merge-sweep --config configs/merge/ties/merge_ties_emotion_intent_kws_langid_speaker_ver_asr_vocalsound.yaml
venv/bin/python main.py merge-sweep --config configs/merge/dare/merge_dare_emotion_intent_kws_langid_speaker_ver_asr_vocalsound.yaml
