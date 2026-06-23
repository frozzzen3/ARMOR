#! /usr/bin/env bash
set -e

GPU_ID="${GPU_ID:-1}"
DATASET="${DATASET:-data/dancer}"
OUTPUT="${OUTPUT:-output/dancer_test}"
MESH_DIR="${MESH_DIR:-data/dancer/meshes_distorted}"
MESH_PREFIX="${MESH_PREFIX:-dancer_}"
START_FRAME="${START_FRAME:-1}"
END_FRAME="${END_FRAME:-2}"
TOTAL_SPLATS="${TOTAL_SPLATS:-100000}"
ALLOC_POLICY="${ALLOC_POLICY:-distortion}"
SEQUENCE_WEIGHT_REDUCTION="${SEQUENCE_WEIGHT_REDUCTION:-max}"
POLICY_PATH="${POLICY_PATH:-${OUTPUT}/sequence_policy/${ALLOC_POLICY}_sequence_${SEQUENCE_WEIGHT_REDUCTION}_${TOTAL_SPLATS}.npy}"
START_MESH="${MESH_DIR}/${MESH_PREFIX}$(printf "%04d" "${START_FRAME}").obj"
# Variable-topology mode: set VARIABLE_TOPOLOGY=1 when per-frame meshes have
# different topology. The sequence policy is then based on the first frame only.
VARIABLE_TOPOLOGY="${VARIABLE_TOPOLOGY:-1}"
TRACK_METHOD="${TRACK_METHOD:-tvm}"

vartopo_args=()
if [[ "${VARIABLE_TOPOLOGY}" == "1" || "${VARIABLE_TOPOLOGY}" == "true" ]]; then
  vartopo_args+=(--variable_topology --track_method "${TRACK_METHOD}")
fi

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
  "${vartopo_args[@]}" \
  --precaptured_mesh_img_path "${DATASET}/mesh" \
  -w \
  --iteration 10
