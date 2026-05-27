#! /usr/bin/env bash
set -e

GPU_ID="${GPU_ID:-1}"
DATASET="${DATASET:-data/dancer}"
OUTPUT="${OUTPUT:-output/dancer}"
MESH_DIR="${MESH_DIR:-data/dancer/meshes_distorted}"
MESH_PREFIX="${MESH_PREFIX:-dancer_}"
START_FRAME="${START_FRAME:-1}"
END_FRAME="${END_FRAME:-1}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python train.py --eval \
  -s "${DATASET}" \
  -m "${OUTPUT}" \
  --texture_obj_path "${MESH_DIR}/${MESH_PREFIX}0001.obj" \
  --mesh_start "${START_FRAME}" \
  --mesh_end "${END_FRAME}" \
  --canonical_frame "${START_FRAME}" \
  --temporal_iterations 500 \
  --mesh_type sugar \
  --gs_type gs_mesh \
  --debugging \
  --debug_freq 100 \
  --occlusion \
  --total_splats 100000 \
  --alloc_policy distortion \
  --policy_path "${OUTPUT}/frame_0001/distortion_100000.npy" \
  --precaptured_mesh_img_path "${DATASET}/mesh" \
  -w --iteration 1000
