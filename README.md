# Layered Mesh-Gaussian (4D)

This repository extends [**Layered Mesh-Gaussian (LMG)**](https://waczjoan.github.io/gaussian-mesh-splatting/) —
itself built on ["GaMeS: Mesh-Based Adapting and Modification of Gaussian
Splatting"](https://arxiv.org/abs/2402.01459) and the original 3D Gaussian
Splatting — from static scenes to **4D dynamic scenes**.

On top of LMG's hybrid mesh-Gaussian representation and distortion-based splat
allocation, this fork adds:

- **Dynamic mesh sequences.** Train and render over a sequence of per-frame
  meshes (`--mesh_start` / `--mesh_end`) that share topology, with a canonical
  frame plus per-frame fine-tuning.
- **Sequence-aware allocation.** A single splat budget is allocated once across
  the whole sequence by reducing per-frame budgeting weights
  (`--sequence_weight_reduction` ∈ {`mean`, `max`, `mean_max`}), so the layout
  stays stable over time.
- **Compact temporal attribute module.** A small neural network
  (`--temporal_attributes`) predicts per-Gaussian residuals (uvw / scaling /
  opacity / color) as a function of time, instead of storing a full set of
  Gaussian attributes per frame.

## Installation

See [INSTALL.md](doc/INSTALL.md) for environment setup. The pinned environment
(see `environment.yml`) is load-bearing for the custom CUDA submodules under
`submodules/` (`diff-gaussian-rasterization`, `simple-knn`).

## Quick start: the 4D pipeline (dancer example)

The four shell wrappers run the full dynamic pipeline and default to the bundled
`data/dancer` sequence. Each is configured by environment variables at the top
of the file (GPU, dataset, mesh dir/prefix, frame range, budget, policy,
temporal-module settings).

```bash
# 0) (Optional) Warmup: pre-render mesh backgrounds/depth for all cameras and
#    precompute the sequence-aware allocation policy.
bash warmup.sh

# 1) Train: canonical frame + per-frame fine-tuning, with the compact temporal
#    attribute module enabled by default.
bash train.sh

# 2) Render: writes per-frame test renders (compact temporal render by default).
bash render.sh

# 3) Evaluate: PSNR / SSIM / LPIPS over the rendered test views.
bash evaluation.sh
```

Common overrides (same variables across scripts):

```bash
GPU_ID=1                       # CUDA device
DATASET=data/dancer            # source dataset (cameras + images)
MESH_DIR=data/dancer/mesh_dynamic
MESH_PREFIX=dancer_            # frame files look like dancer_0001.obj ...
START_FRAME=1 END_FRAME=10     # inclusive frame range
TOTAL_SPLATS=100000            # global splat budget
ALLOC_POLICY=distortion        # uniform|random|area|planarity|distortion
SEQUENCE_WEIGHT_REDUCTION=max  # mean|max|mean_max
TEMPORAL_ATTRIBUTES=1          # enable the compact temporal module
```

## Individual stages (raw commands)

### Warmup (optional)

Pre-renders mesh RGB/depth backgrounds for every camera and generates/validates
the allocation policy `.npy`, then exits before the training loop.

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --eval --warmup_only \
  -s data/dancer -m output/dancer_seq \
  --texture_obj_path data/dancer/mesh_dynamic/dancer_0001.obj \
  --mesh_start 1 --mesh_end 10 \
  --mesh_type sugar --gs_type gs_mesh \
  --total_splats 100000 --alloc_policy distortion \
  --sequence_weight_reduction max \
  --precaptured_mesh_img_path data/dancer/mesh -w --iteration 10
```

### Training

**4D (mesh sequence + temporal module).** When `--mesh_start`/`--mesh_end` span
more than one frame and `--gs_type gs_mesh`, training runs the sequence path: it
computes a sequence-aware policy, trains the canonical frame, then fine-tunes
each remaining frame.

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --eval \
  -s data/dancer -m output/dancer_network \
  --texture_obj_path data/dancer/mesh_dynamic/dancer_0001.obj \
  --mesh_start 1 --mesh_end 10 --canonical_frame 1 \
  --mesh_type sugar --gs_type gs_mesh --occlusion \
  --total_splats 100000 --alloc_policy distortion \
  --sequence_weight_reduction max \
  --temporal_attributes --temporal_attr_width 64 --temporal_attr_depth 3 \
  --temporal_attr_latent_dim 8 --temporal_start_iter 100 \
  --precaptured_mesh_img_path data/dancer/mesh -w --iteration 1000
```

**Static (single mesh).** Omit `--mesh_start`/`--mesh_end` (and the temporal
flags). Use `--total_splats` for an absolute budget or `--budget_per_tri` for a
per-triangle multiplier.

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --eval \
  -s /path/to/dataset -m output/exp_name \
  --texture_obj_path /path/to/mesh.ply --mesh_type colmap \
  --gs_type gs_mesh --occlusion \
  --budget_per_tri 1.5 --alloc_policy planarity \
  -w --iteration 5000
```

### Rendering

```bash
python render_mesh_splat.py \
  -s data/dancer -m output/dancer_network \
  --gs_type gs_mesh --skip_train --occlusion \
  --total_splats 100000 --alloc_policy distortion \
  --texture_obj_path data/dancer/mesh_dynamic/dancer_0001.obj \
  --mesh_type sugar --policy_path /path/to/policy.npy \
  --temporal_attributes --temporal_attr_checkpoint output/dancer_network/temporal_attr_model.pth \
  --mesh_start 1 --mesh_end 10 -w
```

### Evaluation metrics

```bash
python metrics.py -m output/dancer_network/frame_0001 ... --gs_type gs_mesh
```

## Key command-line arguments

### Dataset & mesh

| Argument                 | Description                           | Type           |
| ------------------------ | ------------------------------------- | -------------- |
| `-s, --source_path`      | Path to dataset directory             | str (required) |
| `-m, --model_path`       | Output model directory                | str (required) |
| `--texture_obj_path`     | Path to mesh file (.obj or .ply)      | str            |
| `--mesh_type`            | Mesh source type: `sugar` or `colmap` | str            |
| `-w, --white_background` | Use white background (not black)      | flag           |

### Mesh-splat configuration

| Argument           | Description                                                    | Default   |
| ------------------ | -------------------------------------------------------------- | --------- |
| `--gs_type`        | Renderer/model type (see Model types below)                    | `gs_mesh` |
| `--total_splats`   | Total number of splats for the entire scene                    | None      |
| `--budget_per_tri` | Splats per triangle (multiplier); used if `--total_splats` unset | `1.0`   |
| `--alloc_policy`   | `uniform`, `random`, `area`, `planarity`, `distortion`         | `area`    |
| `--occlusion`      | Enable occlusion-aware (mesh-depth) compositing                | Disabled  |
| `--policy_path`    | Path to a pre-computed policy `.npy`                           | None      |
| `--precaptured_mesh_img_path` | Dir with `mesh_texture/` and `mesh_depth/` (from warmup) | None |
| `--mesh_rasterizer_type` | Mesh background backend: `pytorch3d` or `nvdiffrast`     | `pytorch3d` |

### 4D / sequence & temporal module

| Argument                       | Description                                              | Default |
| ------------------------------ | -------------------------------------------------------- | ------- |
| `--mesh_start` / `--mesh_end`  | Inclusive frame range of a numbered mesh sequence        | None    |
| `--canonical_frame`            | Frame trained first and reused as the base               | first   |
| `--canonical_iterations`       | Iterations for the canonical frame                       | None    |
| `--temporal_iterations`        | Iterations per fine-tuned frame                          | 500     |
| `--sequence_weight_reduction`  | `mean`, `max`, or `mean_max` over per-frame weights      | `max`   |
| `--recompute_sequence_policy`  | Recompute the sequence policy even if cached             | False   |
| `--strict_sequence_topology`   | Require identical face ordering across frames            | False   |
| `--temporal_attributes`        | Enable the compact temporal attribute module             | False   |
| `--temporal_attr_width/_depth/_latent_dim` | MLP width / depth / per-triangle latent size | 64 / 3 / 8 |
| `--temporal_attr_lr`           | Temporal module learning rate                            | 1e-3    |
| `--temporal_start_iter`        | Iteration at which temporal training begins              | 100     |
| `--temporal_predict_{uvw,scaling,opacity,color}` | Which residuals the module predicts    | flags   |
| `--temporal_max_d_{uvw,scaling,opacity,color}`   | Clamp bounds on predicted residuals    | per-attr |
| `--temporal_attr_checkpoint`   | (render) temporal module weights to load                 | None    |

### Training / debugging

| Argument        | Description                       | Default |
| --------------- | --------------------------------- | ------- |
| `--iteration`   | Number of training iterations     | 1000    |
| `--eval`        | Hold out test views during training | False |
| `--warmup_only` | Run only the warmup stage and exit | False  |
| `--debugging`   | Save debug visualizations          | False   |
| `--debug_freq`  | Frequency of saving debug images   | 1       |

## Model types (`--gs_type`)

- `gs` — standard Gaussian Splatting.
- `gs_mesh` — Gaussians parameterized on a mesh surface (the LMG path; requires a mesh).
- `gs_flat` — flat Gaussians (one scale ≈ epsilon).
- `gs_multi_mesh` — multi-mesh variant of `gs_mesh` (use `--meshes`).
- `gs_flame` — FLAME-model parameterization (requires FLAME files under `games/flame_splatting/FLAME/`).
- `gs_points` — render-only parameterization of the flat model.

## Allocation policies (`--alloc_policy`)

`uniform`, `random`, `area` (face-area proportional), `planarity` (mean
resultant length of the face-normal neighborhood), and `distortion` (per-triangle
rendered distortion vs. ground truth). See `scene/budgeting.py`.

## Mesh format notes

- **SuGaR meshes:** `--mesh_type sugar --texture_obj_path /path/to/mesh.obj`
- **Colmap meshes:** `--mesh_type colmap --texture_obj_path /path/to/mesh.ply`

## Output structure

```
output/
└── EXP_NAME/
    ├── frame_0001/                 # per-frame model + renders (sequence mode)
    │   ├── point_cloud/iteration_*/…
    │   └── test/                    # rendered test views (input to metrics)
    ├── temporal_attr_model.pth      # compact temporal module weights
    ├── sequence_policy/             # sequence-aware allocation .npy + weights
    └── results_gs_mesh.json         # metrics
```

## Repository layout

Entry points live at the root (`train.py`, `render_mesh_splat.py`, `metrics.py`,
`full_eval.py`, `convert.py`); `arguments/` + `arguments_games/` hold parameter
groups; `scene/` holds cameras / dataset readers / the Gaussian model /
`budgeting.py`; `games/` holds the mesh / multi-mesh / FLAME / flat
parameterizations; `renderer/` holds the rasterizer front ends; `utils/` holds
shared helpers (including `mesh_utils.py` and `sequence_utils.py`). Dormant
experimental code is kept under `archive/`.

## Citation

If you find this repository helpful in your research, please consider citing and
giving a ⭐.

```
@InProceedings{Sun_2025,
    author    = {Sun, Yuan-Chun and Chen, Guodong and Kondori, Sam Ziaie and Dasari, Mallesham and Hsu, Cheng-Hsin},
    title     = {Layered Mesh-Gaussian},
    year      = {2025},
}
```
