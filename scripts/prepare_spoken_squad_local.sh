#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-data/datasets/Spoken-SQuAD}"
ZIP_PATH="${ROOT_DIR}/Spoken-SQuAD_audio.zip"
TMP_DIR="${ROOT_DIR}/.tmp_extract"
TRAIN_ZIP="${TMP_DIR}/Spoken-SQuAD_audio/train_wav.zip"
TEST_ZIP="${TMP_DIR}/Spoken-SQuAD_audio/dev_wav.zip"
TRAIN_OUT="${ROOT_DIR}/wav/train"
TEST_OUT="${ROOT_DIR}/wav/test"

if [[ ! -f "${ZIP_PATH}" ]]; then
  echo "Missing archive: ${ZIP_PATH}" >&2
  exit 1
fi

mkdir -p "${TMP_DIR}" "${TRAIN_OUT}" "${TEST_OUT}"

echo "Extracting nested train/dev zip files from ${ZIP_PATH} ..."
unzip -n "${ZIP_PATH}" "Spoken-SQuAD_audio/train_wav.zip" "Spoken-SQuAD_audio/dev_wav.zip" -d "${TMP_DIR}"

if [[ ! -f "${TRAIN_ZIP}" || ! -f "${TEST_ZIP}" ]]; then
  echo "Failed to locate nested train/dev zip files under ${TMP_DIR}" >&2
  exit 1
fi

echo "Extracting train wav files to ${TRAIN_OUT} ..."
unzip -n "${TRAIN_ZIP}" -d "${TRAIN_OUT}"

echo "Extracting test wav files to ${TEST_OUT} ..."
unzip -n "${TEST_ZIP}" -d "${TEST_OUT}"

train_count=$(find "${TRAIN_OUT}" -type f -name "*.wav" | wc -l | tr -d ' ')
test_count=$(find "${TEST_OUT}" -type f -name "*.wav" | wc -l | tr -d ' ')

echo "Done."
echo "train wav count: ${train_count}"
echo "test wav count:  ${test_count}"
