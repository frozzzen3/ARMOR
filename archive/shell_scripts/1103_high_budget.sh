#!/bin/bash
# This script runs a pipeline of training, rendering, and metrics for each experiment.
# It does not exit on the first error, but continues to the next experiment.
# set -e

# [NOTE] copy and modify this script for your own experiments

export CUDA_VISIBLE_DEVICES=1

# ======= Config ======

# BUDGETS=(32768 65536 131072 262144 368589 524288 908094 1572865) # increasing order
# BUDGETS=(1572865 908094 524288 368589 262144 131072 65536 32768) # decreasing order



POLICIES=("area" "uniform" "random" "planarity" "distortion")
WHETHER_OCCLUSION=(""  "--occlusion") # sanity check in the logfile


ITERATION=1000

EXP_NAME="1103_high_budget"
# set budget proportional to number of triangles
# e.g. 0.5, 1, 1.5, 2 ... splats per triangle

UNIT_BUDGETS=( 5.0 4.5 4.0 3.5 3.0 2.5 2.0 1.5 1.0 0.5 ) # splats per triangle



SCENE_NAME="hotdog" # add a loop for multiple scenes if needed
DATASET_DIR="/mnt/data1/syjintw/NEU/dataset/hotdog"
BASE_OUTPUT_DIR="output/${EXP_NAME}/${SCENE_NAME}"
BASE_LOG_DIR="log/${EXP_NAME}/${SCENE_NAME}"
PLOT_DIR="${BASE_OUTPUT_DIR}/for_plot"

# ======= Helpers ======

fmt_time() {
    local T=$1
    printf "%02d:%02d:%02d" $((T/3600)) $(((T%3600)/60)) $((T%60))
}
total_start=$(date +%s)
total_exp_seconds=0
failed_experiments=0

# ======= Setup Output Dirs and Logs ======

mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "$BASE_LOG_DIR"
mkdir -p "$PLOT_DIR"

# Timing summary file
TIMING_SUMMARY="${BASE_OUTPUT_DIR}/pipeline_timing_summary.tsv"
echo -e "policy\tbudget\ttrain_secs\trender_secs\tmetrics_secs\ttotal_secs\tstatus" > "$TIMING_SUMMARY"

# Failed experiments log
FAILED_LOG="${BASE_OUTPUT_DIR}/failed_experiments.log"
> "$FAILED_LOG" # Clear the file

# ======= Main Loop ======
for policy in "${POLICIES[@]}"; do
    for budget in "${UNIT_BUDGETS[@]}"; do
        for IS_OCCLUSION in "${WHETHER_OCCLUSION[@]}"; do

            occlusion_tag="no_occlusion"
            if [ "$IS_OCCLUSION" == "--occlusion" ]; then
                occlusion_tag="occlusion"
            fi
            
            SAVE_DIR="${BASE_OUTPUT_DIR}/${policy}_${budget}_${occlusion_tag}/"
            LOG_FILE="${SAVE_DIR}/log_pipeline_${policy}_${budget}_${occlusion_tag}.log"

            # Ensure the save directory exists
            mkdir -p "$SAVE_DIR"

            # Reset timers and status for this experiment
            train_secs=0
            render_secs=0
            metrics_secs=0
            exp_status="PENDING"

            {
                echo "================================================================="
                echo "Starting pipeline: policy=${policy}, budget=${budget}, occlusion=${occlusion_tag}"
                echo "Running on $(hostname), on branch $(git branch --show-current)"
                echo "Dataset: $DATASET_DIR"
                echo "Output will be saved to: $SAVE_DIR"
                date +"%Y-%m-%d %H:%M:%S"
                echo "GPU cores: $CUDA_VISIBLE_DEVICES"
                echo "================================================================="
            } | tee "$LOG_FILE"

            exp_start=$(date +%s)

            # ======= Step 1: Train ======
            echo "Step 1/3: Running training..." | tee -a "$LOG_FILE"
            train_start=$(date +%s)
            if python train.py --eval \
                -s "$DATASET_DIR" \
                -m "$SAVE_DIR" \
                --texture_obj_path /mnt/data1/syjintw/NEU/dataset/hotdog/mesh.obj \
                --debugging \
                --debug_freq 100 \
                $IS_OCCLUSION \
                --budget_per_tri "$budget" \
                --alloc_policy "$policy" \
                --gs_type gs_mesh -w --iteration "$ITERATION" >> "$LOG_FILE"; then
                
                # can use the pre-computed policy instead
                # --policy_path "$DATASET_DIR/policy/${POLICY}_${BUDGET}.npy" \
                

                train_end=$(date +%s)
                train_secs=$((train_end - train_start))
                echo "Training completed in $(fmt_time $train_secs) (${train_secs}s)." | tee -a "$LOG_FILE"
                exp_status="TRAIN_SUCCESS"
            else
                train_end=$(date +%s)
                train_secs=$((train_end - train_start))
                exp_status="TRAIN_FAILED"
                failed_experiments=$((failed_experiments + 1))
                echo "ERROR: Training failed for policy=${policy}, budget=${budget}, occlusion=${occlusion_tag} after ${train_secs}s." | tee -a "$LOG_FILE" "$FAILED_LOG"
            fi

            # ======= Step 2: Render ======
            if [ "$exp_status" = "TRAIN_SUCCESS" ]; then
                echo "Step 2/3: Running render..." | tee -a "$LOG_FILE"
                render_start=$(date +%s)
                if python render_mesh_splat.py \
                    -m "$SAVE_DIR" \
                    --gs_type gs_mesh \
                    --skip_train \
                    $IS_OCCLUSION \
                    --budget_per_tri "$budget" \
                    --alloc_policy "$policy" \
                    --texture_obj_path /mnt/data1/syjintw/NEU/dataset/hotdog/mesh.obj \
                    --policy_path "${SAVE_DIR}/${policy}_${budget}.npy" >> "$LOG_FILE"; then
                    # policy is computed during training already

                    render_end=$(date +%s)
                    render_secs=$((render_end - render_start))
                    echo "Render completed in $(fmt_time $render_secs) (${render_secs}s)." | tee -a "$LOG_FILE"
                    exp_status="RENDER_SUCCESS"
                else
                    render_end=$(date +%s)
                    render_secs=$((render_end - render_start))
                    exp_status="RENDER_FAILED"
                    failed_experiments=$((failed_experiments + 1))
                    echo "ERROR: Render failed for policy=${policy}, budget=${budget}, occlusion=${occlusion_tag} after ${render_secs}s." | tee -a "$LOG_FILE" "$FAILED_LOG"
                fi
            fi

            # ======= Step 3: Metrics ======
            if [ "$exp_status" = "RENDER_SUCCESS" ]; then
                echo "Step 3/3: Running metrics evaluation..." | tee -a "$LOG_FILE"
                metrics_start=$(date +%s)
                if python metrics.py \
                    -m "$SAVE_DIR" \
                    --gs_type gs_mesh >> "$LOG_FILE"; then
                    
                    metrics_end=$(date +%s)
                    metrics_secs=$((metrics_end - metrics_start))
                    echo "Metrics completed in $(fmt_time $metrics_secs) (${metrics_secs}s)." | tee -a "$LOG_FILE"
                    exp_status="SUCCESS"

                    # Copy results JSON for plotting
                    RESULTS_JSON="${SAVE_DIR}/results_gs_mesh.json"
                    PLOT_JSON="${PLOT_DIR}/${policy}_${budget}_${occlusion_tag}.json"
                    if [ -f "$RESULTS_JSON" ]; then
                        cp "$RESULTS_JSON" "$PLOT_JSON"
                        echo "Results copied to: $PLOT_JSON" | tee -a "$LOG_FILE"
                    else
                        echo "WARNING: Results file not found at $RESULTS_JSON" | tee -a "$LOG_FILE"
                    fi
                else
                    metrics_end=$(date +%s)
                    metrics_secs=$((metrics_end - metrics_start))
                    exp_status="METRICS_FAILED"
                    failed_experiments=$((failed_experiments + 1))
                    echo "ERROR: Metrics failed for policy=${policy}, budget=${budget}, occlusion=${occlusion_tag} after ${metrics_secs}s." | tee -a "$LOG_FILE" "$FAILED_LOG"
                fi
            fi

            exp_end=$(date +%s)
            exp_secs=$((exp_end - exp_start))
            total_exp_seconds=$((total_exp_seconds + exp_secs))

            {
                echo ""
                echo "-----------------------------------------------------------------"
                echo "Finished pipeline: policy=${policy}, budget=${budget}, occlusion=${occlusion_tag}"
                echo "Final Status: ${exp_status}"
                echo "Total duration: $(fmt_time $exp_secs) (${exp_secs}s)"
                echo "  - Train:   $(fmt_time $train_secs) (${train_secs}s)"
                echo "  - Render:  $(fmt_time $render_secs) (${render_secs}s)"
                echo "  - Metrics: $(fmt_time $metrics_secs) (${metrics_secs}s)"
                echo "-----------------------------------------------------------------"
                echo ""
            } | tee -a "$LOG_FILE"

            # Copy log file to centralized log directory
            LOG_FILE_COPY="${BASE_LOG_DIR}/log_pipeline_${policy}_${budget}_${occlusion_tag}.log"
            cp "$LOG_FILE" "$LOG_FILE_COPY"

            printf "%s\t%s\t%s\t%d\t%d\t%d\t%d\t%s\n" \
                "$policy" "$budget" "$occlusion_tag" "$train_secs" "$render_secs" "$metrics_secs" "$exp_secs" "$exp_status" >> "$TIMING_SUMMARY"
        done
    done
done

total_end=$(date +%s)
wall_secs=$((total_end - total_start))
echo "================================================================="
echo "All pipelines completed."
echo "Wall-clock total: $(fmt_time "$wall_secs") (${wall_secs}s)"
echo "Sum of experiment durations: $(fmt_time "$total_exp_seconds") (${total_exp_seconds}s)"
printf "TOTAL\t\t%d\t%d\t%d\t%d\tTOTAL_SUM\n" "$train_secs" "$render_secs" "$metrics_secs" "$total_exp_seconds" >> "$TIMING_SUMMARY"
echo "Failed experiments: ${failed_experiments}"
echo "Timing summary saved to: ${TIMING_SUMMARY}"
if [ $failed_experiments -gt 0 ]; then
    echo "Failed experiments log: ${FAILED_LOG}"
fi
echo "================================================================="