#! /usr/bin/env bash
set -e

GPU_ID="${GPU_ID:-1}"
DATASET="${DATASET:-data/dancer}"
OUTPUT="${OUTPUT:-output/dancer}"
MESH_DIR="${MESH_DIR:-data/dancer/meshes_distorted}"
MESH_PREFIX="${MESH_PREFIX:-dancer_}"
START_FRAME="${START_FRAME:-1}"
END_FRAME="${END_FRAME:-1}"

for frame in $(seq "${START_FRAME}" "${END_FRAME}"); do
  frame_id="$(printf "frame_%04d" "${frame}")"
  mesh_path="${MESH_DIR}/${MESH_PREFIX}$(printf "%04d" "${frame}").obj"
  echo "Running warmup for ${mesh_path}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python train.py --eval \
    --warmup_only \
    -s "${DATASET}" \
    -m "${OUTPUT}/${frame_id}" \
    --texture_obj_path "${mesh_path}" \
    --mesh_type sugar \
    --gs_type gs_mesh \
    --debugging \
    --debug_freq 100 \
    --total_splats 100000 \
    --alloc_policy distortion \
    --precaptured_mesh_img_path "${DATASET}/mesh" \
    -w \
    --iteration 10
done
