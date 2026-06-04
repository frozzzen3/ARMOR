#! /usr/bin/env bash
set -e

GPU_ID="${GPU_ID:-1}"
DATASET="${DATASET:-data/dancer}"
OUTPUT="${OUTPUT:-output/dancer_seq}"
MESH_DIR="${MESH_DIR:-data/dancer/mesh_dynamic}"
MESH_PREFIX="${MESH_PREFIX:-dancer_}"
START_FRAME="${START_FRAME:-1}"
END_FRAME="${END_FRAME:-10}"
TOTAL_SPLATS="${TOTAL_SPLATS:-100000}"
ALLOC_POLICY="${ALLOC_POLICY:-distortion}"
SEQUENCE_WEIGHT_REDUCTION="${SEQUENCE_WEIGHT_REDUCTION:-max}"
POLICY_PATH="${POLICY_PATH:-${OUTPUT}/sequence_policy/${ALLOC_POLICY}_sequence_${SEQUENCE_WEIGHT_REDUCTION}_${TOTAL_SPLATS}.npy}"
START_MESH="${MESH_DIR}/${MESH_PREFIX}$(printf "%04d" "${START_FRAME}").obj"

echo "Running sequence-aware warmup for ${START_MESH} frames ${START_FRAME}-${END_FRAME}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
python train.py --eval \
  --warmup_only \
  -s "${DATASET}" \
  -m "${OUTPUT}" \
  --texture_obj_path "${START_MESH}" \
  --mesh_start "${START_FRAME}" \
  --mesh_end "${END_FRAME}" \
  --mesh_type sugar \
  --gs_type gs_mesh \
  --debugging \
  --debug_freq 100 \
  --total_splats "${TOTAL_SPLATS}" \
  --alloc_policy "${ALLOC_POLICY}" \
  --policy_path "${POLICY_PATH}" \
  --sequence_weight_reduction "${SEQUENCE_WEIGHT_REDUCTION}" \
  --precaptured_mesh_img_path "${DATASET}/mesh" \
  -w \
  --iteration 10
