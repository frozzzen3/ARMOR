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
TEMPORAL_ATTRIBUTES="${TEMPORAL_ATTRIBUTES:-1}"
TEMPORAL_ATTR_CHECKPOINT="${TEMPORAL_ATTR_CHECKPOINT:-${OUTPUT}/temporal_attr_model.pth}"
COMPACT_TEMPORAL_RENDER="${COMPACT_TEMPORAL_RENDER:-${TEMPORAL_ATTRIBUTES}}"
# Variable-topology render: set VARIABLE_TOPOLOGY=1 to re-bind the persistent base
# checkpoint to each frame's mesh from cached bindings written by training. Default off.
VARIABLE_TOPOLOGY="${VARIABLE_TOPOLOGY:-1}"
BINDING_CACHE_DIR="${BINDING_CACHE_DIR:-${OUTPUT}/bindings}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${OUTPUT}/frame_$(printf "%04d" "${CANONICAL_FRAME:-${START_FRAME}}")}"
PRECAPTURED_MESH_IMG_PATH="${PRECAPTURED_MESH_IMG_PATH:-${PRECATURED_MESH_IMG_PATH:-${DATASET}/mesh}}"
MESH_RASTERIZER_TYPE="${MESH_RASTERIZER_TYPE:-pytorch3d}"
ITERATION="${ITERATION:-}"
CANONICAL_FRAME="${CANONICAL_FRAME:-${START_FRAME}}"
CANONICAL_FRAME_ID="$(printf "%04d" "${CANONICAL_FRAME}")"
CANONICAL_ITERATIONS="${CANONICAL_ITERATIONS:-10000}"
TEMPORAL_ITERATIONS="${TEMPORAL_ITERATIONS:-10000}"
SKIP_TRAIN="${SKIP_TRAIN:-1}"
OCCLUSION="${OCCLUSION:-1}"
WHITE_BACKGROUND="${WHITE_BACKGROUND:-1}"

# Render directly from a self-contained training bundle (sequence_bundle/): point
# SEQUENCE_BUNDLE at it and the base checkpoint, bindings, and temporal model are all
# resolved from that one folder. Default uses <OUTPUT>/sequence_bundle if present.
SEQUENCE_BUNDLE="${SEQUENCE_BUNDLE:-${OUTPUT}/sequence_bundle}"
if [[ -d "${SEQUENCE_BUNDLE}" ]]; then
  echo "[INFO] Using self-contained sequence bundle: ${SEQUENCE_BUNDLE}"
  BASE_MODEL_PATH="${SEQUENCE_BUNDLE}/base"
  BINDING_CACHE_DIR="${SEQUENCE_BUNDLE}/bindings"
  TEMPORAL_ATTR_CHECKPOINT="${SEQUENCE_BUNDLE}/temporal_attr_model.pth"
fi

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

temporal_args=()
if [[ "${TEMPORAL_ATTRIBUTES}" == "1" || "${TEMPORAL_ATTRIBUTES}" == "true" ]]; then
  if [[ -f "${TEMPORAL_ATTR_CHECKPOINT}" ]]; then
    temporal_args+=(
      --temporal_attributes
      --temporal_attr_checkpoint "${TEMPORAL_ATTR_CHECKPOINT}"
      --mesh_start "${START_FRAME}"
      --mesh_end "${END_FRAME}"
    )
  else
    echo "[WARNING] Temporal attributes requested but checkpoint is missing: ${TEMPORAL_ATTR_CHECKPOINT}"
    echo "          Rendering base Gaussian attributes only."
  fi
fi

compact_render=0
if [[ "${COMPACT_TEMPORAL_RENDER}" == "1" || "${COMPACT_TEMPORAL_RENDER}" == "true" ]]; then
  compact_render=1
fi

binding_args=()
if [[ "${VARIABLE_TOPOLOGY}" == "1" || "${VARIABLE_TOPOLOGY}" == "true" ]]; then
  # variable-topology implies the compact base+bindings+temporal render path
  compact_render=1
  binding_args+=(--binding_cache_dir "${BINDING_CACHE_DIR}")
fi

use_frame_dirs=0
for frame in $(seq "${START_FRAME}" "${END_FRAME}"); do
  frame_id="$(printf "%04d" "${frame}")"
  if [[ -d "${OUTPUT}/frame_${frame_id}/point_cloud" ]]; then
    use_frame_dirs=1
    break
  fi
done

if [[ "${compact_render}" == "1" ]]; then
  if [[ ! -d "${BASE_MODEL_PATH}/point_cloud" ]]; then
    echo "[ERROR] Compact temporal render needs a base checkpoint at: ${BASE_MODEL_PATH}/point_cloud"
    echo "        Set BASE_MODEL_PATH or keep the canonical frame checkpoint."
    exit 1
  fi
elif [[ "${use_frame_dirs}" == "0" && "${START_FRAME}" != "${END_FRAME}" ]]; then
  echo "[ERROR] No per-frame checkpoints found under ${OUTPUT}/frame_XXXX/point_cloud."
  echo "        Re-run train.sh or set START_FRAME and END_FRAME to a single root checkpoint."
  exit 1
fi

for frame in $(seq "${START_FRAME}" "${END_FRAME}"); do
  frame_id="$(printf "%04d" "${frame}")"
  frame_subdir="frame_${frame_id}"
  mesh_path="${MESH_DIR}/${MESH_PREFIX}${frame_id}.${MESH_EXT}"

  if [[ "${compact_render}" == "1" ]]; then
    model_path="${OUTPUT}/${frame_subdir}"
    load_model_path="${BASE_MODEL_PATH}"
    mkdir -p "${model_path}"
  elif [[ "${use_frame_dirs}" == "1" ]]; then
    model_path="${OUTPUT}/${frame_subdir}"
    load_model_path=""
  else
    model_path="${OUTPUT}"
    load_model_path=""
  fi

  if [[ -n "${ITERATION}" ]]; then
    render_iteration="${ITERATION}"
  elif [[ "${compact_render}" == "1" ]]; then
    render_iteration="${CANONICAL_ITERATIONS}"
  elif [[ "${use_frame_dirs}" == "1" && "${frame_id}" == "${CANONICAL_FRAME_ID}" ]]; then
    render_iteration="${CANONICAL_ITERATIONS}"
  elif [[ "${use_frame_dirs}" == "1" ]]; then
    render_iteration="${TEMPORAL_ITERATIONS}"
  else
    render_iteration="-1"
  fi

  if [[ "${compact_render}" == "1" ]]; then
    checkpoint_model_path="${load_model_path}"
  else
    checkpoint_model_path="${model_path}"
  fi

  if [[ ! -d "${checkpoint_model_path}/point_cloud" ]]; then
    echo "[ERROR] Missing checkpoint directory: ${checkpoint_model_path}/point_cloud"
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
  load_args=()
  if [[ -n "${load_model_path}" ]]; then
    load_args+=(--load_model_path "${load_model_path}")
  fi

  CUDA_VISIBLE_DEVICES="${GPU_ID}" python render_mesh_splat.py \
    -s "${DATASET}" \
    -m "${model_path}" \
    --iteration "${render_iteration}" \
    --gs_type "${GS_TYPE}" \
    "${load_args[@]}" \
    "${skip_args[@]}" \
    "${occlusion_args[@]}" \
    "${white_background_args[@]}" \
    "${temporal_args[@]}" \
    "${binding_args[@]}" \
    --total_splats "${TOTAL_SPLATS}" \
    --alloc_policy "${ALLOC_POLICY}" \
    --texture_obj_path "${mesh_path}" \
    --mesh_type "${MESH_TYPE}" \
    --policy_path "${policy_path}" \
    --precaptured_mesh_img_path "${PRECAPTURED_MESH_IMG_PATH}" \
    --mesh_rasterizer_type "${MESH_RASTERIZER_TYPE}"
done
