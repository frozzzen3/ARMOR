#! /usr/bin/env bash
set -e

GPU_ID="${GPU_ID:-1}"
DATASET="${DATASET:-data/dancer}"
OUTPUT="${OUTPUT:-output/dancer_network}"
MESH_DIR="${MESH_DIR:-data/dancer/mesh_dynamic}"
MESH_PREFIX="${MESH_PREFIX:-dancer_}"
START_FRAME="${START_FRAME:-1}"
END_FRAME="${END_FRAME:-10}"
TOTAL_SPLATS="${TOTAL_SPLATS:-100000}"
ALLOC_POLICY="${ALLOC_POLICY:-distortion}"
SEQUENCE_WEIGHT_REDUCTION="${SEQUENCE_WEIGHT_REDUCTION:-max}"
POLICY_PATH="${POLICY_PATH:-${OUTPUT}/sequence_policy/${ALLOC_POLICY}_sequence_${SEQUENCE_WEIGHT_REDUCTION}_${TOTAL_SPLATS}.npy}"
TEMPORAL_ATTRIBUTES="${TEMPORAL_ATTRIBUTES:-1}"
TEMPORAL_ATTR_WIDTH="${TEMPORAL_ATTR_WIDTH:-64}"
TEMPORAL_ATTR_DEPTH="${TEMPORAL_ATTR_DEPTH:-3}"
TEMPORAL_ATTR_LATENT_DIM="${TEMPORAL_ATTR_LATENT_DIM:-8}"
TEMPORAL_START_ITER="${TEMPORAL_START_ITER:-100}"

temporal_args=()
if [[ "${TEMPORAL_ATTRIBUTES}" == "1" || "${TEMPORAL_ATTRIBUTES}" == "true" ]]; then
  temporal_args+=(
    --temporal_attributes
    --temporal_attr_width "${TEMPORAL_ATTR_WIDTH}"
    --temporal_attr_depth "${TEMPORAL_ATTR_DEPTH}"
    --temporal_attr_latent_dim "${TEMPORAL_ATTR_LATENT_DIM}"
    --temporal_start_iter "${TEMPORAL_START_ITER}"
  )
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" python train.py --eval \
  -s "${DATASET}" \
  -m "${OUTPUT}" \
  --texture_obj_path "${MESH_DIR}/${MESH_PREFIX}0001.obj" \
  --mesh_start "${START_FRAME}" \
  --mesh_end "${END_FRAME}" \
  --canonical_frame "${START_FRAME}" \
  --temporal_iterations 1000 \
  --mesh_type sugar \
  --gs_type gs_mesh \
  --debugging \
  --debug_freq 100 \
  --occlusion \
  --total_splats "${TOTAL_SPLATS}" \
  --alloc_policy "${ALLOC_POLICY}" \
  --policy_path "${POLICY_PATH}" \
  --sequence_weight_reduction "${SEQUENCE_WEIGHT_REDUCTION}" \
  "${temporal_args[@]}" \
  --precaptured_mesh_img_path "${DATASET}/mesh" \
  -w --iteration 1000
