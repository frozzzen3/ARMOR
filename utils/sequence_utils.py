#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""Argument/dataset helpers, splat-allocation policy resolution, training-time
mesh background loading, and sequence-aware (4D) policy computation.

Extracted verbatim from train.py to keep the training entry point focused on
orchestration. Behavior is unchanged.
"""

import os
import json
import shutil
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

from renderer.mesh_splat_renderer import render
from scene.dataset_readers import (
    infer_mesh_image_subdir,
    readCamerasFromTransforms,
    readColmapCameras,
)
from scene.colmap_loader import (
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)
from scene.budgeting import allocate_splats_from_weights, get_budgeting_policy
from games.mesh_splatting.scene.temporal_attribute_model import (
    estimate_compact_temporal_storage,
)
from utils.mesh_utils import (
    infer_mesh_frame_subdir,
    append_subdir,
    build_precaptured_path,
    load_budgeting_trimesh,
    load_textured_mesh,
    validate_sequence_topology,
    sequence_frame_label,
)


def build_frame_run_args(base_args, mesh_path, use_subdir=True, policy_path=None):
    run_args = Namespace(**vars(base_args))
    run_args.texture_obj_path = str(mesh_path)

    frame_subdir = infer_mesh_frame_subdir(run_args.texture_obj_path)
    if use_subdir and frame_subdir is not None:
        run_args.model_path = append_subdir(base_args.model_path, frame_subdir)

    if policy_path is not None:
        run_args.policy_path = policy_path

    return run_args


def ensure_canonical_policy_file(scene, dataset, requested_policy_path):
    num_tri = scene.point_cloud.triangles.shape[0] if hasattr(scene.point_cloud, "triangles") else 0
    if dataset.total_splats is None:
        total_splats = int(dataset.budget_per_tri * num_tri)
    else:
        total_splats = dataset.total_splats

    copied_policy_path = Path(scene.model_path) / f"{dataset.alloc_policy}_{total_splats}.npy"
    if requested_policy_path:
        requested_policy_path = Path(requested_policy_path)
        requested_policy_path.parent.mkdir(parents=True, exist_ok=True)
        if not requested_policy_path.exists():
            if copied_policy_path.exists():
                shutil.copyfile(copied_policy_path, requested_policy_path)
            else:
                dataset_policy_path = Path(dataset.source_path) / (
                    f"policy/mesh_{dataset.mesh_type}/tri_{num_tri}/{dataset.alloc_policy}/{total_splats}.npy"
                )
                if dataset_policy_path.exists():
                    shutil.copyfile(dataset_policy_path, requested_policy_path)
                else:
                    raise FileNotFoundError(
                        f"Could not locate canonical policy file at {copied_policy_path} "
                        f"or {dataset_policy_path}"
                    )
        return str(requested_policy_path)

    if copied_policy_path.exists():
        return str(copied_policy_path)

    dataset_policy_path = Path(dataset.source_path) / (
        f"policy/mesh_{dataset.mesh_type}/tri_{num_tri}/{dataset.alloc_policy}/{total_splats}.npy"
    )
    if dataset_policy_path.exists():
        return str(dataset_policy_path)

    raise FileNotFoundError("Canonical policy file was not generated.")


def load_training_background(viewpoint_cam, scene, dataset, pipe, precaptured_mesh_img_path,
                             frame_subdir, mesh_rasterizer_type):
    viewpoint_camera_height = viewpoint_cam.image_height
    viewpoint_camera_width = viewpoint_cam.image_width

    transform = T.Compose([
        T.ToTensor(),
    ])

    bg = None
    bg_depth = None
    if precaptured_mesh_img_path:
        cached_bg_path = build_precaptured_path(
            Path(precaptured_mesh_img_path) / mesh_rasterizer_type / "mesh_texture",
            viewpoint_cam.image_name,
            ".png",
            frame_subdir=frame_subdir,
        )
        if cached_bg_path.exists():
            img = Image.open(cached_bg_path).convert("RGB")
            img = img.resize((viewpoint_camera_width, viewpoint_camera_height), Image.BILINEAR)
            bg = transform(img).to(torch.float32).cuda()

        cached_bg_depth_path = build_precaptured_path(
            Path(precaptured_mesh_img_path) / mesh_rasterizer_type / "mesh_depth",
            viewpoint_cam.image_name,
            ".pt",
            frame_subdir=frame_subdir,
        )
        if cached_bg_depth_path.exists():
            bg_depth = torch.load(cached_bg_depth_path).unsqueeze(0).to("cuda")

    if bg is None or bg_depth is None:
        mesh_bg_color = (1, 1, 1) if dataset.white_background else (0, 0, 0)
        render_pkg = render(
            viewpoint_cam, scene.gaussians, pipe,
            bg_color=None, bg_depth=None,
            textured_mesh=scene.textured_mesh,
            mesh_background_color=mesh_bg_color,
            mesh_rasterizer_type=mesh_rasterizer_type,
        )
        if bg is None:
            bg = render_pkg["bg_color"].detach()
        if bg_depth is None:
            bg_depth = render_pkg["bg_depth"].detach()

    return bg, bg_depth


def extract_dataset_args(model_params, run_args):
    dataset = model_params.extract(run_args)
    dataset.total_splats = run_args.total_splats
    dataset.budget_per_tri = run_args.budget_per_tri
    dataset.alloc_policy = run_args.alloc_policy
    dataset.warmup_only = run_args.warmup_only
    dataset.mesh_type = run_args.mesh_type.lower()
    return dataset


def get_total_splats_for_mesh(dataset, num_triangles):
    if dataset.total_splats is None:
        return int(dataset.budget_per_tri * num_triangles)
    return dataset.total_splats


def load_policy_camera_infos(dataset, texture_obj_path):
    if os.path.exists(os.path.join(dataset.source_path, "transforms_train.json")):
        image_subdir = infer_mesh_image_subdir(texture_obj_path)
        train_cam_infos = readCamerasFromTransforms(
            dataset.source_path,
            "transforms_train.json",
            dataset.white_background,
            ".png",
            image_subdir=image_subdir,
        )
        test_cam_infos = readCamerasFromTransforms(
            dataset.source_path,
            "transforms_test.json",
            dataset.white_background,
            ".png",
            image_subdir=image_subdir,
        )
        if not dataset.eval:
            train_cam_infos.extend(test_cam_infos)
        return train_cam_infos

    if os.path.exists(os.path.join(dataset.source_path, "sparse")):
        try:
            cam_extrinsics = read_extrinsics_binary(os.path.join(dataset.source_path, "sparse/0", "images.bin"))
            cam_intrinsics = read_intrinsics_binary(os.path.join(dataset.source_path, "sparse/0", "cameras.bin"))
        except Exception:
            cam_extrinsics = read_extrinsics_text(os.path.join(dataset.source_path, "sparse/0", "images.txt"))
            cam_intrinsics = read_intrinsics_text(os.path.join(dataset.source_path, "sparse/0", "cameras.txt"))

        reading_dir = "images" if dataset.images is None else dataset.images
        cam_infos = readColmapCameras(
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            images_folder=os.path.join(dataset.source_path, reading_dir),
        )
        cam_infos = sorted(cam_infos, key=lambda x: x.image_name)
        if dataset.eval:
            return [c for idx, c in enumerate(cam_infos) if idx % 8 != 0]
        return cam_infos

    raise ValueError("Could not recognize scene type for sequence allocation.")


def reduce_sequence_weights(frame_weights, reduction):
    stacked = np.stack(frame_weights, axis=0)
    if reduction == "mean":
        weights = stacked.mean(axis=0)
    elif reduction == "max":
        weights = stacked.max(axis=0)
    elif reduction == "mean_max":
        weights = 0.5 * stacked.mean(axis=0) + 0.5 * stacked.max(axis=0)
    else:
        raise ValueError(f"Unknown sequence weight reduction: {reduction}")

    return np.maximum(weights.astype(np.float32), 1e-8)


def default_sequence_policy_path(dataset, mesh_paths, num_triangles, total_splats, reduction):
    policy_name = f"{dataset.alloc_policy}_sequence_{reduction}"
    return Path(dataset.source_path) / (
        f"policy/mesh_{dataset.mesh_type}/tri_{num_triangles}/"
        f"{policy_name}/{sequence_frame_label(mesh_paths)}/{total_splats}.npy"
    )


def write_temporal_storage_report(gaussians, temporal_model, num_frames, report_path):
    report = estimate_compact_temporal_storage(gaussians, temporal_model, num_frames)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    duplicated_mb = report["duplicated_per_frame_bytes"] / (1024 * 1024)
    compact_mb = report["compact_temporal_bytes"] / (1024 * 1024)
    saved_pct = report["estimated_savings_ratio"] * 100.0
    print(
        "[INFO] Compact temporal storage estimate: "
        f"duplicated={duplicated_mb:.2f} MiB, compact={compact_mb:.2f} MiB, "
        f"saved={saved_pct:.1f}% ({report_path})"
    )
    return report


def ensure_sequence_policy_file(base_args, model_params, mesh_paths, requested_policy_path=""):
    first_run_args = build_frame_run_args(base_args, mesh_paths[0], use_subdir=False)
    dataset = extract_dataset_args(model_params, first_run_args)
    reduction = base_args.sequence_weight_reduction
    recompute = base_args.recompute_sequence_policy

    if requested_policy_path and Path(requested_policy_path).exists() and not recompute:
        print(f"[INFO] Using existing sequence policy: {requested_policy_path}")
        return requested_policy_path

    print(f"[INFO] Computing sequence-aware allocation over {len(mesh_paths)} mesh frames.")
    reference_faces = None
    frame_weights = []
    num_triangles = None

    for mesh_path in mesh_paths:
        run_args = build_frame_run_args(base_args, mesh_path, use_subdir=False)
        frame_dataset = extract_dataset_args(model_params, run_args)
        mesh_scene = load_budgeting_trimesh(str(mesh_path))
        faces = np.asarray(mesh_scene.faces)

        if reference_faces is None:
            reference_faces = faces.copy()
            num_triangles = int(faces.shape[0])
        else:
            validate_sequence_topology(
                reference_faces,
                faces,
                mesh_path,
                strict=base_args.strict_sequence_topology,
            )

        train_cam_infos = load_policy_camera_infos(frame_dataset, str(mesh_path))
        p3d_mesh = None
        if frame_dataset.alloc_policy.startswith("distortion"):
            p3d_mesh = load_textured_mesh(frame_dataset, str(mesh_path))

        budgeting_policy = get_budgeting_policy(
            frame_dataset.alloc_policy,
            mesh=mesh_scene,
            viewpoint_camera_infos=train_cam_infos,
            dataset_path=frame_dataset.source_path,
            mesh_type=frame_dataset.mesh_type,
            p3d_mesh=p3d_mesh,
        )
        frame_weights.append(np.asarray(budgeting_policy.weights, dtype=np.float32))
        if p3d_mesh is not None:
            del p3d_mesh
            torch.cuda.empty_cache()

    total_splats = get_total_splats_for_mesh(dataset, num_triangles)
    sequence_weights = reduce_sequence_weights(frame_weights, reduction)
    num_splats_per_triangle = allocate_splats_from_weights(sequence_weights, total_splats)

    if requested_policy_path:
        allocation_save_path = Path(requested_policy_path)
    else:
        allocation_save_path = default_sequence_policy_path(
            dataset,
            mesh_paths,
            num_triangles,
            total_splats,
            reduction,
        )

    allocation_save_path.parent.mkdir(parents=True, exist_ok=True)
    weights_save_path = allocation_save_path.parent / "weights.npy"
    np.save(allocation_save_path, num_splats_per_triangle)
    np.save(weights_save_path, sequence_weights)

    print(f"[INFO] Saved sequence allocation policy to: {allocation_save_path}")
    print(f"[INFO] Saved sequence weights to: {weights_save_path}")
    print(f"[INFO] Sequence policy splats: total={num_splats_per_triangle.sum()}, "
          f"min={num_splats_per_triangle.min()}, max={num_splats_per_triangle.max()}, "
          f"mean={num_splats_per_triangle.mean():.2f}")

    return str(allocation_save_path)
