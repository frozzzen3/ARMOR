#! /usr/bin/env bash
set -e

GPU_ID="${GPU_ID:-1}"
DATASET="${DATASET:-data/dancer}"
OUTPUT="${OUTPUT:-output/dancer_test}"
MESH_DIR="${MESH_DIR:-data/dancer/meshes_distorted}"
MESH_PREFIX="${MESH_PREFIX:-dancer_}"
START_FRAME="${START_FRAME:-1}"
END_FRAME="${END_FRAME:-3}"
TOTAL_SPLATS="${TOTAL_SPLATS:-100000}"
ALLOC_POLICY="${ALLOC_POLICY:-distortion}"
SEQUENCE_WEIGHT_REDUCTION="${SEQUENCE_WEIGHT_REDUCTION:-max}"
POLICY_PATH="${POLICY_PATH:-${OUTPUT}/sequence_policy/${ALLOC_POLICY}_sequence_${SEQUENCE_WEIGHT_REDUCTION}_${TOTAL_SPLATS}.npy}"
# Per-frame fine-tuning iterations for subsequent (non-canonical) frames.
TEMPORAL_ITERATIONS="${TEMPORAL_ITERATIONS:-5000}"
CANONICAL_ITERATIONS="${CANONICAL_ITERATIONS:-5000}"
# Variable-topology mode: set VARIABLE_TOPOLOGY=1 when per-frame meshes have
# different topology (no consistent-topology preprocessing). Each frame is re-bound to
# its mesh and the full per-frame appearance is trained and saved as that frame's own
# checkpoint (frame_XXXX/). Default off, so the default invocation is unchanged.
VARIABLE_TOPOLOGY="${VARIABLE_TOPOLOGY:-1}"
TRACK_METHOD="${TRACK_METHOD:-tvm}"
# External ARAP+TVM tracker (only used when TRACK_METHOD=tvm). Defaults point at the
# submodules; override if built elsewhere.
TVM_ARAP_DIR="${TVM_ARAP_DIR:-submodules/arap-volume-tracking}"
TVM_EDITOR_EXE="${TVM_EDITOR_EXE:-submodules/tvm-editing/TVMEditor.Test/bin/Release/net5.0/TVMEditor.Test}"
TVM_CONFIG_TEMPLATE="${TVM_CONFIG_TEMPLATE:-submodules/arap-volume-tracking/config/config-dancer-max.xml}"
TVM_POINT_COUNT="${TVM_POINT_COUNT:-2000}"
TVM_VG_RESOLUTION="${TVM_VG_RESOLUTION:-512}"
TVM_DOTNET="${TVM_DOTNET:-dotnet}"

vartopo_args=()
if [[ "${VARIABLE_TOPOLOGY}" == "1" || "${VARIABLE_TOPOLOGY}" == "true" ]]; then
  vartopo_args+=(--variable_topology --track_method "${TRACK_METHOD}")
  if [[ "${TRACK_METHOD}" == "tvm" ]]; then
    vartopo_args+=(
      --tvm_arap_dir "${TVM_ARAP_DIR}"
      --tvm_editor_exe "${TVM_EDITOR_EXE}"
      --tvm_config_template "${TVM_CONFIG_TEMPLATE}"
      --tvm_point_count "${TVM_POINT_COUNT}"
      --tvm_vg_resolution "${TVM_VG_RESOLUTION}"
      --tvm_dotnet "${TVM_DOTNET}"
    )
  fi
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" python train.py --eval \
  -s "${DATASET}" \
  -m "${OUTPUT}" \
  --texture_obj_path "${MESH_DIR}/${MESH_PREFIX}0001.obj" \
  --mesh_start "${START_FRAME}" \
  --mesh_end "${END_FRAME}" \
  --canonical_frame "${START_FRAME}" \
  --canonical_iterations "${CANONICAL_ITERATIONS}" \
  --temporal_iterations "${TEMPORAL_ITERATIONS}" \
  --mesh_type sugar \
  --gs_type gs_mesh \
  --debugging \
  --debug_freq 100 \
  --occlusion \
  --total_splats "${TOTAL_SPLATS}" \
  --alloc_policy "${ALLOC_POLICY}" \
  --policy_path "${POLICY_PATH}" \
  --sequence_weight_reduction "${SEQUENCE_WEIGHT_REDUCTION}" \
  "${vartopo_args[@]}" \
  --precaptured_mesh_img_path "${DATASET}/mesh" \
  -w --iteration "${CANONICAL_ITERATIONS}"
