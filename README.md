# Layered Mesh-Gaussian

Layered Mesh-Gaussian (LMG) is a research implementation built upon the official ["GaMeS: Mesh-Based Adapting and Modification of Gaussian Splatting"](https://arxiv.org/abs/2402.01459).  
This project extends the original [official codebase](https://waczjoan.github.io/gaussian-mesh-splatting/) with additional utilities and experimental workflows for mesh-driven Gaussian Splatting and hybrid 3D representation rendering.

## Installation

See [INSTALL.md](doc/INSTALL.md) for instructions environment setup.

## Getting Started

## Quick Start: Full Pipeline with Debug Script

For a complete pipeline (warmup → training → rendering → metrics) with a specific scene, policy, and budget:

```bash
bash debug_pipeline.sh
```

Configure the script by editing these variables at the top:

```bash
export CUDA_VISIBLE_DEVICES=2

UNIT_BUDGET=1.5                    # Budget proportional to number of triangles
POLICY="planarity"                 # Options: planarity, area, distortion, uniform, random
DATASET_DIR="/path/to/dataset"
SCENE_NAME="bicycle"
MESH_TYPE="colmap"                 # Options: "sugar" or "colmap"
MESH_FILE="/path/to/mesh.ply"      # .ply for colmap, .obj for sugar
RESOLUTION=""                      # Or "--resolution 4" for faster debugging
IS_WHITE_BG="-w"                   # Or empty string for black background
```

## Batch Experiments: Multiple Budgets and Policies

For running experiments with multiple budgets, policies, and occlusion settings:

```bash
bash 1113_pipeline.sh
```

Configure at the top of the script:

```bash
export CUDA_VISIBLE_DEVICES=3

DATASET_DIR="/path/to/dataset"
SAVE_DIR="/path/to/output"

# Splat budgets to test (0 = mesh only, no splats)
BUDGETS=( 1 3000000 2000000 1000000 524288 262144 131072 )

# Allocation policies to test
POLICIES=("area" "distortion" "planarity" "uniform" "random")

# Test with and without occlusion
WHETHER_OCCLUSION=("--occlusion" "")

ITERATION="5000"
EXP_NAME="1113_downsampled"

SCENE_NAME="bicycle"
MESH_TYPE="colmap"
MESH_FILE="/path/to/mesh.ply"
```

This script automatically:

- Runs warmup, training, rendering, and metrics for each combination
- Logs timing for each stage
- Tracks failed experiments
- Generates a timing summary TSV file: `output/EXPERIMENT_NAME/SCENE_NAME/pipeline_timing_summary.tsv`
- Saves results of metrics as JSON files to `output/EXPERIMENT_NAME/SCENE_NAME/for_plot/`

## Individual Pipeline Stages

### Step 0: Warmup (Optional - Pre-render Mesh Backgrounds, Precalculate Allocation Policies)

Warmup pre-renders mesh backgrounds and depth maps for all training cameras. This is optional and can speed up initialization, but is not required.

```bash
CUDA_VISIBLE_DEVICES=2 python train.py --eval \
  --warmup_only \
  -s /path/to/dataset \
  -m output/exp_name \
  --texture_obj_path /path/to/mesh.ply \
  --mesh_type colmap \
  --gs_type gs_mesh \
  --debugging \
  --debug_freq 100 \
  --total_splats 1000000 \
  --alloc_policy planarity \
  --precaptured_mesh_img_path /path/to/mesh/dir \
  -w --iteration 10
```

**What warmup does:**

- Pre-renders mesh backgrounds and depth maps for all training cameras
- Saves to `precaptured_mesh_img_path/mesh_texture/` and `mesh_depth/` directories
- Generates or validates policy allocation file (.npy)
- Exits after completion, does not enter training loop
- Optional for both `--occlusion` and non-occlusion modes

### Step 1: Training

```bash
CUDA_VISIBLE_DEVICES=2 python train.py --eval \
  -s /path/to/dataset \
  -m output/exp_name \
  --texture_obj_path /path/to/mesh.ply \
  --mesh_type colmap \
  --gs_type gs_mesh \
  --debugging \
  --debug_freq 100 \
  --occlusion \
  --total_splats 1000000 \
  --alloc_policy planarity \
  --policy_path output/exp_name/policy.npy \
  --precaptured_mesh_img_path /path/to/mesh/images \
  -w --iteration 5000
```

**Alternative: Use `--budget_per_tri` instead of `--total_splats`:**

```bash
python train.py --eval \
  -s /path/to/dataset \
  -m output/exp_name \
  --texture_obj_path /path/to/mesh.ply \
  --mesh_type colmap \
  --gs_type gs_mesh \
  --budget_per_tri 1.5 \
  --alloc_policy planarity \
  -w --iteration 5000
```

### Step 2: Rendering

```bash
python render_mesh_splat.py \
  -m output/exp_name \
  --gs_type gs_mesh \
  --skip_train \
  --occlusion \
  --total_splats 1000000 \
  --alloc_policy planarity \
  --texture_obj_path /path/to/mesh.ply \
  --mesh_type colmap \
  --policy_path output/exp_name/policy.npy
```

### Step 3: Evaluation Metrics

```bash
python metrics.py \
  -m output/exp_name \
  --gs_type gs_mesh
```

## Key Command-Line Arguments

### Dataset & Mesh

| Argument                 | Description                           | Type           |
| ------------------------ | ------------------------------------- | -------------- |
| `-s, --source_path`      | Path to dataset directory             | str (required) |
| `-m, --model_path`       | Output model directory                | str (required) |
| `--texture_obj_path`     | Path to mesh file (.obj or .ply)      | str            |
| `--mesh_type`            | Mesh source type: `sugar` or `colmap` | str            |
| `-w, --white_background` | Use white background (not black)      | flag           |

### Mesh-Splat Configuration

| Argument           | Description                                                    | Default   |
| ------------------ | -------------------------------------------------------------- | --------- |
| `--gs_type`        | Renderer type: `gs`, `gs_flat`, or `gs_mesh`                   | `gs_mesh` |
| `--total_splats`   | Total number of splats for entire scene, int                   | None      |
| `--budget_per_tri` | Splats per triangle (multiplier), float                        | 1.0       |
| `--alloc_policy`   | Policy: `uniform`, `random`, `area`, `planarity`, `distortion` | `area`    |

| Argument                      | Description                                                      | Default  |
| ----------------------------- | ---------------------------------------------------------------- | -------- |
| `--occlusion`                 | Enable occlusion-aware rendering                                 | Disabled |
| `--policy_path`               | Path to pre-computed policy `.npy` file                          | None     |
| `--precaptured_mesh_img_path` | Dir with `mesh_texture/` and `mesh_depth/` subdirs (from warmup) | None     |

### Training Configuration

| Argument        | Description                       | Default |
| --------------- | --------------------------------- | ------- |
| `--iteration`   | Number of training iterations     | 1000    |
| `--eval`        | Enable evaluation during training | False   |
| `--warmup_only` | Only run warmup stage and exit    | False   |
|                 |                                   |         |

### Debugging

| Argument       | Description                               | Default |
| -------------- | ----------------------------------------- | ------- |
| `--debugging`  | Save debug visualizations during training | False   |
| `--debug_freq` | Frequency of saving debug images          | 1       |

## Comparison: Different Rendering Types

### Original Gaussian Splatting (pure GS)

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --eval \
  -s /path/to/dataset \
  -m output/gs_only \
  --gs_type gs \
  -w --iteration 5000 \
  --debugging --debug_freq 100
```

```bash
python render_gs.py -m output/gs_only --gs_type gs --skip_train
python metrics.py -m output/gs_only --gs_type gs
```

### Mesh-Splat WITH Occlusion

```bash
# Step 1: Training (warmup optional)
CUDA_VISIBLE_DEVICES=1 python train.py --eval \
  -s /path/to/dataset \
  -m output/meshsplat_with_occ \
  --texture_obj_path /path/to/mesh.ply \
  --mesh_type colmap \
  --gs_type gs_mesh \
  --occlusion \
  --budget_per_tri 1.5 \
  --alloc_policy planarity \
  -w --iteration 5000

# Step 2: Rendering
python render_mesh_splat.py \
  -m output/meshsplat_with_occ \
  --gs_type gs_mesh \
  --skip_train \
  --occlusion \
  --texture_obj_path /path/to/mesh.ply \
  --mesh_type colmap
```

### Mesh-Splat WITHOUT Occlusion

```bash
# Step 1: Training
CUDA_VISIBLE_DEVICES=2 python train.py --eval \
  -s /path/to/dataset \
  -m output/meshsplat_no_occ \
  --texture_obj_path /path/to/mesh.ply \
  --mesh_type colmap \
  --gs_type gs_mesh \
  --budget_per_tri 1.5 \
  --alloc_policy planarity \
  -w --iteration 5000

# Step 2: Rendering
python render_mesh_splat.py \
  -m output/meshsplat_no_occ \
  --gs_type gs_mesh \
  --skip_train \
  --texture_obj_path /path/to/mesh.ply \
  --mesh_type colmap
```

## Mesh Format Notes

**SuGaR meshes (.obj):**

```bash
--mesh_type sugar --texture_obj_path /path/to/mesh.obj
```

**Colmap meshes (.ply):**

```bash
--mesh_type colmap --texture_obj_path /path/to/mesh.ply
```

## Output Structure

```
output/
├── EXPERIMENT_NAME/
│   ├── SCENE_NAME/
│   │   ├── policy_1.0_occlusion/
│   │   │   ├── log_pipeline_*.log
│   │   │   ├── policy.npy
│   │   │   ├── results_gs_mesh.json
│   │   │   └── ...
│   │   ├── pipeline_timing_summary.tsv
│   │   ├── failed_experiments.log
│   │   └── for_plot/
│   │       └── *.json (results for plotting)
│   └── log/
│       └── *.log
```

## Notes

- **Warmup is optional:** Pre-renders mesh backgrounds for faster initialization, but not required
- **Budget modes:** Use either `--total_splats` for absolute budget or `--budget_per_tri` for relative budget
- **Policy files:** Generated during training or warmup, can be reused across experiments
- **Precaptured images:** Optional. If not provided, will be computed on-the-fly during training
- **Debugging:** Enable `--debugging` and set `--debug_freq` to inspect intermediate visualizations
- **Occlusion flag:** Works independently - use with or without warmup as needed

## Citation

If you find this repository/work helpful in your research, welcome to cite these papers and give a ⭐.

```
@InProceedings{Sun_2025,
    author    = {Sun, Yuan-Chun and Chen, Guodong and Kondori, Sam Ziaie and Dasari, Mallesham and Hsu, Cheng-Hsin},
    title     = {Layered Mesh-Gaussian},
    year      = {2025},
}
```

Last update: Nov 30, 2025