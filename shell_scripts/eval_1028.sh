#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=1



# ====== Simple timers ======
fmt_time() {
    local T=$1
    printf "%02d:%02d:%02d" $((T/3600)) $(((T%3600)/60)) $((T%60))
}
total_start=$(date +%s)
total_exp_seconds=0
failed_experiments=0

# ======= Experiment Parameters ======
BUDGETS=(8192 16384 32768 65536 131072 262144 368589 524288 908094 1572865) # Add your budgets here
POLICIES=("planarity" "area" "rand_uni")
# POLICIES=("texture" "mse_mask")




SCENE_NAME="hotdog"
DATASET_DIR="/mnt/data1/syjintw/NEU/dataset/hotdog"
BASE_LOAD_DIR="output/1028_without_occ/${SCENE_NAME}"
PLOT_DIR="output/1028_without_occ/for_plot"

# Create the base output directory if it doesn't exist
mkdir -p "$BASE_LOAD_DIR"
mkdir -p "$PLOT_DIR"

# Timing summary file
TIMING_SUMMARY="${BASE_LOAD_DIR}/render_timing_summary.tsv"
echo -e "policy\tbudget\trender_secs\trender_hms\tmetrics_secs\tmetrics_hms\ttotal_secs\ttotal_hms\tstatus" > "$TIMING_SUMMARY"

# Failed experiments log
FAILED_LOG="${BASE_LOAD_DIR}/failed_renders.log"
> "$FAILED_LOG"  # Clear the file

# ======= Main Loop ======
for policy in "${POLICIES[@]}"; do
    for budget in "${BUDGETS[@]}"; do
        
        LOAD_DIR="${BASE_LOAD_DIR}/${policy}_${budget}/"
        LOG_FILE="log_render_metrics_${policy}_${budget}.log"

        {
            echo "================================================================="
            echo "Starting render+metrics: policy=${policy}, budget=${budget}"
            echo "Running on $(hostname), on branch $(git branch --show-current)"
            echo "Loading model from: $LOAD_DIR"
            date +"%Y-%m-%d %H:%M:%S"
            echo "GPU cores: $CUDA_VISIBLE_DEVICES"
            echo "================================================================="
        } | tee "$LOG_FILE"

        exp_start=$(date +%s)
        exp_status="SUCCESS"

        # ======= Step 1: Render ======
        echo "Step 1/3: Running render..." | tee -a "$LOG_FILE"
        render_start=$(date +%s)


        # NO --occlusion \
        if python render_mesh_splat.py \
            -m "$LOAD_DIR" \
            --gs_type gs_mesh \
            --skip_train \
            --texture_obj_path /mnt/data1/syjintw/NEU/dataset/hotdog/mesh.obj \
            --total_splats "$budget" \
            --alloc_policy "$policy" \
            2>&1 | tee -a "$LOG_FILE"; then
            
            render_end=$(date +%s)
            render_secs=$((render_end - render_start))
            render_hms=$(fmt_time "$render_secs")
            echo "Render completed in ${render_hms} (${render_secs}s)" | tee -a "$LOG_FILE"
        else
            exp_status="RENDER_FAILED"
            failed_experiments=$((failed_experiments + 1))
            echo "ERROR: Render failed: policy=${policy}, budget=${budget}" | tee -a "$FAILED_LOG"
            echo "Check log file for details: $LOG_FILE" | tee -a "$FAILED_LOG"
            cp "$LOG_FILE" "$LOAD_DIR/$LOG_FILE" 2>/dev/null || true
            
            render_end=$(date +%s)
            render_secs=$((render_end - render_start))
            render_hms=$(fmt_time "$render_secs")
            metrics_secs=0
            metrics_hms="00:00:00"
            
            exp_end=$(date +%s)
            exp_secs=$((exp_end - exp_start))
            exp_hms=$(fmt_time "$exp_secs")
            total_exp_seconds=$((total_exp_seconds + exp_secs))
            
            printf "%s\t%s\t%d\t%s\t%d\t%s\t%d\t%s\t%s\n" \
                "$policy" "$budget" "$render_secs" "$render_hms" \
                "$metrics_secs" "$metrics_hms" "$exp_secs" "$exp_hms" "$exp_status" >> "$TIMING_SUMMARY"
            
            continue
        fi

        # ======= Step 2: Metrics ======
        echo "Step 2/3: Running metrics evaluation..." | tee -a "$LOG_FILE"
        metrics_start=$(date +%s)
        
        if python metrics.py \
            -m "$LOAD_DIR" \
            --gs_type gs_mesh \
            2>&1 | tee -a "$LOG_FILE"; then
            
            metrics_end=$(date +%s)
            metrics_secs=$((metrics_end - metrics_start))
            metrics_hms=$(fmt_time "$metrics_secs")
            echo "Metrics completed in ${metrics_hms} (${metrics_secs}s)" | tee -a "$LOG_FILE"
        else
            exp_status="METRICS_FAILED"
            failed_experiments=$((failed_experiments + 1))
            echo "ERROR: Metrics failed: policy=${policy}, budget=${budget}" | tee -a "$FAILED_LOG"
            echo "Check log file for details: $LOG_FILE" | tee -a "$FAILED_LOG"
            
            metrics_end=$(date +%s)
            metrics_secs=$((metrics_end - metrics_start))
            metrics_hms=$(fmt_time "$metrics_secs")
        fi

        # ======= Step 3: Copy Results JSON ======

        if [ "$exp_status" = "SUCCESS" ]; then
            echo "Step 3/3: Copying results JSON for plotting..." | tee -a "$LOG_FILE"
            RESULTS_JSON="${LOAD_DIR}/results_gs_mesh.json"
            PLOT_JSON="${PLOT_DIR}/${policy}_${budget}.json"
            
            if [ -f "$RESULTS_JSON" ]; then
                cp "$RESULTS_JSON" "$PLOT_JSON"
                echo "Results copied to: $PLOT_JSON" | tee -a "$LOG_FILE"
            else
                echo "WARNING: Results file not found at $RESULTS_JSON" | tee -a "$LOG_FILE"
                echo "WARNING: Results file missing: policy=${policy}, budget=${budget}" | tee -a "$FAILED_LOG"
            fi
        else
            echo "Step 3/3: Skipping JSON copy due to previous failure" | tee -a "$LOG_FILE"
        fi

        # Copy the log file to the load directory
        cp "$LOG_FILE" "$LOAD_DIR/$LOG_FILE"
        echo "Log file saved to $LOAD_DIR/$LOG_FILE" | tee -a "$LOG_FILE"

        exp_end=$(date +%s)
        exp_secs=$((exp_end - exp_start))
        exp_hms=$(fmt_time "$exp_secs")
        total_exp_seconds=$((total_exp_seconds + exp_secs))

        {
            echo "Total duration: ${exp_hms} (${exp_secs}s)"
            echo "  - Render: ${render_hms} (${render_secs}s)"
            echo "  - Metrics: ${metrics_hms} (${metrics_secs}s)"
            echo "Experiment status: ${exp_status}"
            echo "Finished: policy=${policy}, budget=${budget}"
            echo ""
        } | tee -a "$LOG_FILE"

        printf "%s\t%s\t%d\t%s\t%d\t%s\t%d\t%s\t%s\n" \
            "$policy" "$budget" "$render_secs" "$render_hms" \
            "$metrics_secs" "$metrics_hms" "$exp_secs" "$exp_hms" "$exp_status" >> "$TIMING_SUMMARY"

    done
done

total_end=$(date +%s)
wall_secs=$((total_end - total_start))
echo "================================================================="
echo "All evaluation completed."
echo "Wall-clock total: $(fmt_time "$wall_secs") (${wall_secs}s)"
echo "Sum of experiment durations: $(fmt_time "$total_exp_seconds") (${total_exp_seconds}s)"
printf "TOTAL\t\t\t\t\t\t%d\t%s\tTOTAL_SUM\n" "$total_exp_seconds" "$(fmt_time "$total_exp_seconds")" >> "$TIMING_SUMMARY"
echo "Failed evaluation: ${failed_experiments}"
echo "Timing summary saved to: ${TIMING_SUMMARY}"
echo "Plot data directory: ${PLOT_DIR}"
if [ $failed_experiments -gt 0 ]; then
    echo "Failed evaluation log: ${FAILED_LOG}"
fi
echo "================================================================="