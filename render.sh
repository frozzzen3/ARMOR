#! /usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-1}"
DATASET="${DATASET:-data/dancer}"
OUTPUT="${OUTPUT:-output/dancer_test}"
MESH_DIR="${MESH_DIR:-data/dancer/meshes_distorted}"
MESH_PREFIX="${MESH_PREFIX:-dancer_}"
MESH_EXT="${MESH_EXT:-obj}"
START_FRAME="${START_FRAME:-1}"
END_FRAME="${END_FRAME:-3}"
GS_TYPE="${GS_TYPE:-gs_mesh}"
MESH_TYPE="${MESH_TYPE:-sugar}"
TOTAL_SPLATS="${TOTAL_SPLATS:-100000}"
ALLOC_POLICY="${ALLOC_POLICY:-distortion}"
SEQUENCE_WEIGHT_REDUCTION="${SEQUENCE_WEIGHT_REDUCTION:-max}"
SEQUENCE_POLICY_PATH="${SEQUENCE_POLICY_PATH:-${OUTPUT}/sequence_policy/${ALLOC_POLICY}_sequence_${SEQUENCE_WEIGHT_REDUCTION}_${TOTAL_SPLATS}.npy}"
PRECAPTURED_MESH_IMG_PATH="${PRECAPTURED_MESH_IMG_PATH:-${PRECATURED_MESH_IMG_PATH:-${DATASET}/mesh}}"
MESH_RASTERIZER_TYPE="${MESH_RASTERIZER_TYPE:-pytorch3d}"
ITERATION="${ITERATION:-}"
CANONICAL_FRAME="${CANONICAL_FRAME:-${START_FRAME}}"
CANONICAL_FRAME_ID="$(printf "%04d" "${CANONICAL_FRAME}")"
# Per-frame checkpoints are saved at these iteration counts by train.sh.
CANONICAL_ITERATIONS="${CANONICAL_ITERATIONS:-5000}"
TEMPORAL_ITERATIONS="${TEMPORAL_ITERATIONS:-5000}"
SKIP_TRAIN="${SKIP_TRAIN:-1}"
OCCLUSION="${OCCLUSION:-1}"
WHITE_BACKGROUND="${WHITE_BACKGROUND:-1}"

occlusion_args=()
if [[ "${OCCLUSION}" == "1" || "${OCCLUSION}" == "true" ]]; then
  occlusion_args+=(--occlusion)
fi

white_background_args=()
if [[ "${WHITE_BACKGROUND}" == "1" || "${WHITE_BACKGROUND}" == "true" ]]; then
  white_background_args+=(-w)
fi

skip_args=()
if [[ "${SKIP_TRAIN}" == "1" || "${SKIP_TRAIN}" == "true" ]]; then
  skip_args+=(--skip_train)
fi

# Each frame is rendered from its own per-frame checkpoint written by training under
# <OUTPUT>/frame_XXXX/. (A single-frame run falls back to the root <OUTPUT> checkpoint.)
use_frame_dirs=0
for frame in $(seq "${START_FRAME}" "${END_FRAME}"); do
  frame_id="$(printf "%04d" "${frame}")"
  if [[ -d "${OUTPUT}/frame_${frame_id}/point_cloud" ]]; then
    use_frame_dirs=1
    break
  fi
done

if [[ "${use_frame_dirs}" == "0" && "${START_FRAME}" != "${END_FRAME}" ]]; then
  echo "[ERROR] No per-frame checkpoints found under ${OUTPUT}/frame_XXXX/point_cloud."
  echo "        Re-run train.sh or set START_FRAME and END_FRAME to a single root checkpoint."
  exit 1
fi

for frame in $(seq "${START_FRAME}" "${END_FRAME}"); do
  frame_id="$(printf "%04d" "${frame}")"
  frame_subdir="frame_${frame_id}"
  mesh_path="${MESH_DIR}/${MESH_PREFIX}${frame_id}.${MESH_EXT}"

  if [[ "${use_frame_dirs}" == "1" ]]; then
    model_path="${OUTPUT}/${frame_subdir}"
  else
    model_path="${OUTPUT}"
  fi

  if [[ -n "${ITERATION}" ]]; then
    render_iteration="${ITERATION}"
  elif [[ "${use_frame_dirs}" == "1" && "${frame_id}" == "${CANONICAL_FRAME_ID}" ]]; then
    render_iteration="${CANONICAL_ITERATIONS}"
  elif [[ "${use_frame_dirs}" == "1" ]]; then
    render_iteration="${TEMPORAL_ITERATIONS}"
  else
    render_iteration="-1"
  fi

  if [[ ! -d "${model_path}/point_cloud" ]]; then
    echo "[ERROR] Missing checkpoint directory: ${model_path}/point_cloud"
    exit 1
  fi
  if [[ ! -f "${mesh_path}" ]]; then
    echo "[ERROR] Missing mesh file: ${mesh_path}"
    exit 1
  fi

  policy_path="${POLICY_PATH:-${SEQUENCE_POLICY_PATH}}"
  if [[ ! -f "${policy_path}" && -f "${model_path}/${ALLOC_POLICY}_${TOTAL_SPLATS}.npy" ]]; then
    policy_path="${model_path}/${ALLOC_POLICY}_${TOTAL_SPLATS}.npy"
  elif [[ ! -f "${policy_path}" && -f "${OUTPUT}/${ALLOC_POLICY}_${TOTAL_SPLATS}.npy" ]]; then
    policy_path="${OUTPUT}/${ALLOC_POLICY}_${TOTAL_SPLATS}.npy"
  fi

  echo "[INFO] Rendering ${frame_subdir}: ${model_path} at iteration ${render_iteration}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" python render_mesh_splat.py \
    -s "${DATASET}" \
    -m "${model_path}" \
    --iteration "${render_iteration}" \
    --gs_type "${GS_TYPE}" \
    "${skip_args[@]}" \
    "${occlusion_args[@]}" \
    "${white_background_args[@]}" \
    --rebind_decoded_mesh \
    --total_splats "${TOTAL_SPLATS}" \
    --alloc_policy "${ALLOC_POLICY}" \
    --texture_obj_path "${mesh_path}" \
    --mesh_type "${MESH_TYPE}" \
    --policy_path "${policy_path}" \
    --precaptured_mesh_img_path "${PRECAPTURED_MESH_IMG_PATH}" \
    --mesh_rasterizer_type "${MESH_RASTERIZER_TYPE}"
done
