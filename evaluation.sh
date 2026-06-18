#! /usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
OUTPUT="${OUTPUT:-output/dancer_test}"
START_FRAME="${START_FRAME:-1}"
END_FRAME="${END_FRAME:-2}"
GS_TYPE="${GS_TYPE:-gs_mesh}"

model_paths=()
use_frame_dirs=0
for frame in $(seq "${START_FRAME}" "${END_FRAME}"); do
  frame_id="$(printf "%04d" "${frame}")"
  if [[ -d "${OUTPUT}/frame_${frame_id}" ]]; then
    use_frame_dirs=1
    break
  fi
done

if [[ "${use_frame_dirs}" == "1" ]]; then
  for frame in $(seq "${START_FRAME}" "${END_FRAME}"); do
    frame_id="$(printf "%04d" "${frame}")"
    model_path="${OUTPUT}/frame_${frame_id}"
    if [[ -d "${model_path}/test" ]]; then
      model_paths+=("${model_path}")
    else
      echo "[WARN] Skipping ${model_path}; render output not found at ${model_path}/test"
    fi
  done
else
  if [[ -d "${OUTPUT}/test" ]]; then
    model_paths+=("${OUTPUT}")
  fi
fi

if [[ "${#model_paths[@]}" == "0" ]]; then
  echo "[ERROR] No rendered test outputs found. Run render.sh first."
  exit 1
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" python metrics.py \
  -m "${model_paths[@]}" \
  --gs_type "${GS_TYPE}"
