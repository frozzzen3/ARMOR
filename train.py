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

# from pytorch3d.io import load_objs_as_meshes

from itertools import count
import os
import json
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
# from renderer.gaussian_renderer import render, network_gui
from renderer.mesh_splat_renderer import render, network_gui
import sys
from scene import Scene
from games import (
    optimizationParamTypeCallbacks,
    gaussianModel
)

from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import numpy as np
from pathlib import Path
import re
import shutil

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.io import load_ply
from pytorch3d.renderer import (
    AmbientLights,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    )

import open3d as o3d
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor
)
from scene.cameras import convert_camera_from_gs_to_pytorch3d
from pytorch3d.renderer.blending import BlendParams
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from renderer.mesh_renderer.mesh_utils import ensure_mesh_has_texture

import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
    CompactTemporalAttributeModel,
    estimate_compact_temporal_storage,
)


# [good to have] loss-informed stop criteria
LOSS_CONVG_THRESH = 0.01


def infer_mesh_frame_subdir(texture_obj_path, prefix="frame"):
    if not texture_obj_path:
        return None

    match = re.search(r"(\d+)$", Path(texture_obj_path).stem)
    if match is None:
        return None

    return f"{prefix}_{match.group(1)}"


def build_precaptured_path(base_dir, name, suffix, frame_subdir=None):
    target_dir = Path(base_dir)
    if frame_subdir is not None:
        target_dir = target_dir / frame_subdir
    return target_dir / f"{name}{suffix}"


def resolve_mesh_sequence(mesh_path, mesh_start, mesh_end):
    if mesh_start is None and mesh_end is None:
        return [Path(mesh_path)]

    if mesh_start is None or mesh_end is None:
        raise ValueError("Both --mesh_start and --mesh_end must be provided together")

    mesh_path = Path(mesh_path)
    match = re.search(r"(\d+)$", mesh_path.stem)
    if match is None:
        raise ValueError("Mesh filename must end with digits when using --mesh_start/--mesh_end")

    width = len(match.group(1))
    prefix = mesh_path.stem[:match.start(1)]
    suffix = mesh_path.suffix
    return [
        mesh_path.with_name(f"{prefix}{index:0{width}d}{suffix}")
        for index in range(mesh_start, mesh_end + 1)
    ]


def append_subdir(path_value, subdir):
    if not path_value:
        return path_value
    return str(Path(path_value) / subdir)


def append_policy_subdir(policy_path, subdir):
    if not policy_path:
        return policy_path
    policy_path = Path(policy_path)
    return str(policy_path.parent / subdir / policy_path.name)


def extract_frame_index(mesh_path: Path):
    match = re.search(r"(\d+)$", mesh_path.stem)
    if match is None:
        return None
    return int(match.group(1))


def resolve_canonical_mesh(mesh_paths, canonical_frame=None):
    if canonical_frame is None:
        return mesh_paths[0]

    for mesh_path in mesh_paths:
        frame_index = extract_frame_index(mesh_path)
        if frame_index == canonical_frame:
            return mesh_path

    raise ValueError(f"Could not find canonical frame {canonical_frame} in mesh sequence.")


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


def load_budgeting_trimesh(texture_obj_path):
    mesh_scene = trimesh.load(texture_obj_path, force="mesh", process=False)
    mesh_scene.apply_transform(trimesh.transformations.rotation_matrix(
        angle=-np.pi / 2,
        direction=[1, 0, 0],
        point=[0, 0, 0],
    ))
    return mesh_scene


def validate_sequence_topology(reference_faces, mesh_faces, mesh_path, strict=False):
    if reference_faces.shape != mesh_faces.shape:
        raise ValueError(
            "Sequence-aware allocation requires identical mesh topology. "
            f"{mesh_path} has faces shape {mesh_faces.shape}, expected {reference_faces.shape}."
        )
    if strict and not np.array_equal(reference_faces, mesh_faces):
        raise ValueError(
            "Sequence-aware allocation requires stable face ordering/indices. "
            f"{mesh_path} does not match the first frame's face array."
        )
    if not strict and not np.array_equal(reference_faces, mesh_faces):
        print(
            "[WARNING] Sequence-aware allocation: face indices differ from the first frame "
            f"for {mesh_path}, but face array shape matches. Assuming triangle row order "
            "is the temporal correspondence."
        )


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


def sequence_frame_label(mesh_paths):
    first = extract_frame_index(mesh_paths[0])
    last = extract_frame_index(mesh_paths[-1])
    if first is not None and last is not None:
        return f"frames_{first:04d}_{last:04d}"
    return f"{mesh_paths[0].stem}_to_{mesh_paths[-1].stem}"


def default_sequence_policy_path(dataset, mesh_paths, num_triangles, total_splats, reduction):
    policy_name = f"{dataset.alloc_policy}_sequence_{reduction}"
    return Path(dataset.source_path) / (
        f"policy/mesh_{dataset.mesh_type}/tri_{num_triangles}/"
        f"{policy_name}/{sequence_frame_label(mesh_paths)}/{total_splats}.npy"
    )


def normalized_frame_time(mesh_path, mesh_paths):
    if len(mesh_paths) <= 1:
        return 0.0
    try:
        index = mesh_paths.index(Path(mesh_path))
    except ValueError:
        index = 0
    return float(index) / float(len(mesh_paths) - 1)


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


def run_training_loop(gs_type, scene, dataset, gaussians, opt, pipe, save_xyz,
                      debugging, debug_freq, occlusion, precaptured_mesh_img_path,
                      texture_obj_path, mesh_rasterizer_type, num_iterations,
                      save_at_end=True, temporal_model=None,
                      temporal_frame_time=None, temporal_start_iteration=0,
                      temporal_num_frames=1):
    if debugging:
        print("[DEBUG] [INFO] Debugging mode is on.")
        check_path = Path(scene.model_path) / "debugging" / "training_check"
        check_path.mkdir(parents=True, exist_ok=True)
    else:
        check_path = None

    print("[INFO] Start Training..." )

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(1, num_iterations + 1), desc="Training progress")
    viewpoint_stack = None
    frame_subdir = infer_mesh_frame_subdir(texture_obj_path) if precaptured_mesh_img_path else None
    gaussians.optimizer.zero_grad(set_to_none=True)
    if temporal_model is not None:
        temporal_model.optimizer.zero_grad(set_to_none=True)

    for iteration in range(1, num_iterations + 1):
        os.makedirs(f"{scene.model_path}/xyz", exist_ok=True)
        if save_xyz and (iteration % 5000 == 1 or iteration == num_iterations):
            torch.save(gaussians.get_xyz, f"{scene.model_path}/xyz/{iteration}.pt")

        gaussians.update_learning_rate(iteration)
        if temporal_model is not None and iteration >= temporal_start_iteration:
            gaussians.apply_temporal_attributes(temporal_model, temporal_frame_time or 0.0)
        elif hasattr(gaussians, "clear_temporal_attributes"):
            gaussians.clear_temporal_attributes()

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            print(f"[DEBUG] Train:: current SH degree: {gaussians.active_sh_degree}")

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        rand_cam_id = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_cam_id)

        bg, bg_depth = load_training_background(
            viewpoint_cam=viewpoint_cam,
            scene=scene,
            dataset=dataset,
            pipe=pipe,
            precaptured_mesh_img_path=precaptured_mesh_img_path,
            frame_subdir=frame_subdir,
            mesh_rasterizer_type=mesh_rasterizer_type,
        )

        pure_bg_template = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        pure_bg = torch.tensor(pure_bg_template, dtype=torch.float32, device="cuda").view(3, 1, 1)
        pure_bg = pure_bg.expand(3, viewpoint_cam.image_height, viewpoint_cam.image_width)
        pure_bg_depth = torch.full(
            (1, viewpoint_cam.image_height, viewpoint_cam.image_width),
            0,
            dtype=torch.float32,
            device="cuda",
        )

        if gs_type == "gs":
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg_color=pure_bg, bg_depth=pure_bg_depth)
        elif gs_type == "gs_mesh":
            if occlusion:
                render_pkg = render(
                    viewpoint_cam, gaussians, pipe,
                    bg_color=bg, bg_depth=bg_depth,
                    textured_mesh=scene.textured_mesh,
                    mesh_rasterizer_type=mesh_rasterizer_type,
                )
            else:
                render_pkg = render(
                    viewpoint_cam, gaussians, pipe,
                    bg_color=bg, bg_depth=pure_bg_depth,
                    textured_mesh=scene.textured_mesh,
                    mesh_rasterizer_type=mesh_rasterizer_type,
                )
        else:
            raise ValueError(f"Unsupported gs_type for temporal training: {gs_type}")

        image = render_pkg["render"]

        if iteration % debug_freq == 0:
            print(f"[DEBUG] Training Iteration {iteration}, viewpoint: {viewpoint_cam.image_name}")

        gt_image = viewpoint_cam.original_image.cuda()

        if debugging and iteration % debug_freq == 0 and check_path is not None:
            gt_img_to_save = gt_image.detach().clamp(0, 1).cpu()
            TF.to_pil_image(gt_img_to_save).save(check_path / f"{iteration}_gt.png")

            img_to_save = image.detach().clamp(0, 1).cpu()
            TF.to_pil_image(img_to_save).save(check_path / f"{iteration}_training.png")

            bg_to_save = render_pkg["bg_color"].detach().clamp(0, 1).cpu()
            TF.to_pil_image(bg_to_save).save(check_path / f"{iteration}_training_mesh_bg.png")

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}"})
            progress_bar.update(1)

            if iteration < num_iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                if temporal_model is not None and iteration >= temporal_start_iteration:
                    temporal_model.optimizer.step()
                    temporal_model.optimizer.zero_grad(set_to_none=True)

        if hasattr(gaussians, 'update_alpha'):
            gaussians.update_alpha()
        if hasattr(gaussians, 'prepare_scaling_rot'):
            gaussians.prepare_scaling_rot()

    gaussians.optimizer.step()
    gaussians.optimizer.zero_grad(set_to_none=True)
    if temporal_model is not None:
        temporal_model.optimizer.step()
        temporal_model.optimizer.zero_grad(set_to_none=True)
    if hasattr(gaussians, 'update_alpha'):
        gaussians.update_alpha()
    if hasattr(gaussians, 'prepare_scaling_rot'):
        gaussians.prepare_scaling_rot()
    progress_bar.close()

    if save_at_end:
        if temporal_model is not None and hasattr(gaussians, "clear_temporal_attributes"):
            gaussians.clear_temporal_attributes()
        scene.save(num_iterations)
        if temporal_model is not None:
            temporal_path = Path(scene.model_path) / "point_cloud" / f"iteration_{num_iterations}" / "temporal_attr_model.pth"
            temporal_model.save(temporal_path)
            write_temporal_storage_report(
                gaussians,
                temporal_model,
                temporal_num_frames,
                Path(scene.model_path) / "temporal_storage_report.json",
            )


def training_sequence(gs_type, base_args, opt, pipe, mesh_paths,
                      save_xyz, debugging, debug_freq, occlusion,
                      requested_policy_path, precaptured_mesh_img_path,
                      mesh_rasterizer_type, canonical_frame,
                      canonical_iterations, temporal_iterations, model_params):
    canonical_mesh = resolve_canonical_mesh(mesh_paths, canonical_frame)
    remaining_meshes = [mesh for mesh in mesh_paths if mesh != canonical_mesh]
    ordered_meshes = [canonical_mesh] + remaining_meshes

    gaussians = gaussianModel[gs_type](base_args.sh_degree)
    canonical_policy_path = requested_policy_path or ""

    if len(mesh_paths) > 1 and (not canonical_policy_path or not Path(canonical_policy_path).exists()):
        canonical_policy_path = ensure_sequence_policy_file(
            base_args,
            model_params,
            mesh_paths,
            requested_policy_path=canonical_policy_path,
        )

    first_run_args = build_frame_run_args(base_args, canonical_mesh, use_subdir=len(mesh_paths) > 1)
    dataset_args = extract_dataset_args(model_params, first_run_args)

    prepare_output_and_logger(dataset_args)
    print(f"[INFO] Canonical frame: {canonical_mesh}")
    textured_mesh = load_textured_mesh(dataset_args, str(canonical_mesh)) if mesh_rasterizer_type == "pytorch3d" else load_textured_mesh_for_nvdiffrast(dataset_args, str(canonical_mesh))
    scene = Scene(
        dataset_args,
        gaussians,
        policy_path=canonical_policy_path,
        texture_obj_path=str(canonical_mesh),
        textured_mesh=textured_mesh,
        initialize_gaussians=True,
    )
    gaussians.training_setup(opt)
    temporal_model = None
    if base_args.temporal_attributes:
        num_triangles = int(scene.point_cloud.triangles.shape[0])
        temporal_model = CompactTemporalAttributeModel(
            num_triangles=num_triangles,
            latent_dim=base_args.temporal_attr_latent_dim,
            hidden_dim=base_args.temporal_attr_width,
            depth=base_args.temporal_attr_depth,
            time_frequencies=base_args.temporal_attr_time_frequencies,
            max_d_uvw=base_args.temporal_max_d_uvw,
            max_d_scaling=base_args.temporal_max_d_scaling,
            max_d_opacity=base_args.temporal_max_d_opacity,
            max_d_color=base_args.temporal_max_d_color,
            predict_uvw=base_args.temporal_predict_uvw,
            predict_scaling=base_args.temporal_predict_scaling,
            predict_opacity=base_args.temporal_predict_opacity,
            predict_color=base_args.temporal_predict_color,
            lr=base_args.temporal_attr_lr,
        ).cuda()
        print(
            "[INFO] Compact temporal attribute model enabled: "
            f"{temporal_model.parameter_count} parameters for {len(mesh_paths)} frames"
        )
        write_temporal_storage_report(
            gaussians,
            temporal_model,
            len(mesh_paths),
            Path(base_args.model_path) / "temporal_storage_report_initial.json",
        )
    canonical_policy_path = ensure_canonical_policy_file(scene, dataset_args, requested_policy_path)

    run_training_loop(
        gs_type=gs_type,
        scene=scene,
        dataset=dataset_args,
        gaussians=gaussians,
        opt=opt,
        pipe=pipe,
        save_xyz=save_xyz,
        debugging=debugging,
        debug_freq=debug_freq,
        occlusion=occlusion,
        precaptured_mesh_img_path=precaptured_mesh_img_path,
        texture_obj_path=str(canonical_mesh),
        mesh_rasterizer_type=mesh_rasterizer_type,
        num_iterations=canonical_iterations,
        temporal_model=temporal_model,
        temporal_frame_time=normalized_frame_time(canonical_mesh, mesh_paths),
        temporal_start_iteration=base_args.temporal_start_iter,
        temporal_num_frames=len(mesh_paths),
    )

    for mesh_path in ordered_meshes[1:]:
        run_args = build_frame_run_args(base_args, mesh_path, use_subdir=len(mesh_paths) > 1, policy_path=canonical_policy_path)
        dataset = extract_dataset_args(model_params, run_args)

        prepare_output_and_logger(dataset)
        print(f"[INFO] Temporal frame: {mesh_path}")
        if mesh_rasterizer_type == "pytorch3d":
            textured_mesh = load_textured_mesh(dataset, str(mesh_path))
        else:
            textured_mesh = load_textured_mesh_for_nvdiffrast(dataset, str(mesh_path))

        scene = Scene(
            dataset,
            gaussians,
            policy_path=canonical_policy_path,
            texture_obj_path=str(mesh_path),
            textured_mesh=textured_mesh,
            initialize_gaussians=False,
        )
        gaussians.rebind_to_mesh(scene.point_cloud.vertices, scene.point_cloud.faces)
        gaussians.point_cloud = scene.point_cloud
        scene.gaussians = gaussians

        run_training_loop(
            gs_type=gs_type,
            scene=scene,
            dataset=dataset,
            gaussians=gaussians,
            opt=opt,
            pipe=pipe,
            save_xyz=save_xyz,
            debugging=debugging,
            debug_freq=debug_freq,
            occlusion=occlusion,
            precaptured_mesh_img_path=precaptured_mesh_img_path,
            texture_obj_path=str(mesh_path),
            mesh_rasterizer_type=mesh_rasterizer_type,
            num_iterations=temporal_iterations,
            temporal_model=temporal_model,
            temporal_frame_time=normalized_frame_time(mesh_path, mesh_paths),
            temporal_start_iteration=base_args.temporal_start_iter,
            temporal_num_frames=len(mesh_paths),
        )

    if temporal_model is not None:
        root_temporal_path = Path(base_args.model_path) / "temporal_attr_model.pth"
        temporal_model.save(root_temporal_path)
        write_temporal_storage_report(
            gaussians,
            temporal_model,
            len(mesh_paths),
            Path(base_args.model_path) / "temporal_storage_report.json",
        )
        print(f"[INFO] Saved final compact temporal model to: {root_temporal_path}")


   
def training(gs_type, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
            debug_from, save_xyz,
            # >>>> [YC] add
            texture_obj_path, 
            debugging, debug_freq,
            occlusion,
            policy_path,
            precaptured_mesh_img_path,
            mesh_rasterizer_type="pytorch3d"
            # <<<< [YC] add
            ):
    
    # --------------------------- Warm Up Stage -------------------------- #
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = gaussianModel[gs_type](dataset.sh_degree) # [YC] note: nothing changing here
    print("[INFO] Training() policy_path:", policy_path)
        
    # >>>> [YC] add: if there is textured mesh, load it here (before training loop)
    if gs_type == "gs_mesh":
        if mesh_rasterizer_type == "pytorch3d":
            textured_mesh = load_textured_mesh(dataset, texture_obj_path)
        elif mesh_rasterizer_type == "nvdiffrast":
            textured_mesh = load_textured_mesh_for_nvdiffrast(dataset, texture_obj_path)
    else:
        textured_mesh = None
    # [DONE] pass the textured mesh, to Scene, Policy, renderer and such.
    # because, why pass the path when its already loaded right here?
    # <<<< [YC] add
    
    
    #! [YC] note: main changing point is here
    
    print("[DEBUG] going into Scene initialization...")
    
    scene = Scene(dataset, gaussians, policy_path=policy_path, texture_obj_path=texture_obj_path, textured_mesh=textured_mesh)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    if debugging:
        print("[DEBUG] [INFO] Debugging mode is on.")
        check_path = Path(scene.model_path)/"debugging"/"training_check"
        check_path.mkdir(parents=True, exist_ok=True)
    
    if dataset.warmup_only:
        if not precaptured_mesh_img_path:
            raise ValueError("precaptured_mesh_img_path must be provided for warmup_only mode")

        frame_subdir = infer_mesh_frame_subdir(texture_obj_path)
        
        # ------------------------------ Training Camera ----------------------------- #
        # Precapture mesh_bg and mesh_bg_depth in warmup stage
        precaptured_bg_dir = Path(precaptured_mesh_img_path) / mesh_rasterizer_type /f"mesh_texture"
        precaptured_depth_dir = Path(precaptured_mesh_img_path) / mesh_rasterizer_type / f"mesh_depth"
        
        # Ensure directories exist
        if frame_subdir is not None:
            (precaptured_bg_dir / frame_subdir).mkdir(parents=True, exist_ok=True)
            (precaptured_depth_dir / frame_subdir).mkdir(parents=True, exist_ok=True)
        else:
            precaptured_bg_dir.mkdir(parents=True, exist_ok=True)
            precaptured_depth_dir.mkdir(parents=True, exist_ok=True)
        
        print("[INFO] Warmup stage: Generating precaptured mesh background and depth images...")
        
        for cam in tqdm(scene.getTrainCameras(), desc="Precapturing training backgrounds", unit="camera"):
            # Generate file paths
            bg_save_path = build_precaptured_path(precaptured_bg_dir, cam.image_name, ".png", frame_subdir=frame_subdir)
            depth_save_path = build_precaptured_path(precaptured_depth_dir, cam.image_name, ".pt", frame_subdir=frame_subdir)
            
            # Skip if already exists
            if bg_save_path.exists() and depth_save_path.exists():
                print(f"\t[INFO] Skipping {cam.image_name}, already exists.")
                continue
            
            # Render background and depth
            bg_color = (1,1,1) if dataset.white_background else (0,0,0)
            render_pkg = render(cam, gaussians, pipe, 
                                bg_color=None, bg_depth=None, 
                                textured_mesh=scene.textured_mesh,
                                mesh_background_color=bg_color,
                                mesh_rasterizer_type=mesh_rasterizer_type
                                )
            
            # Save background image
            bg_image = render_pkg["bg_color"].detach().clamp(0, 1).cpu()
            bg_image_pil = TF.to_pil_image(bg_image)
            bg_image_pil.save(bg_save_path)
            
            # Save depth image
            bg_depth = render_pkg["bg_depth"].detach().cpu()
            torch.save(bg_depth, depth_save_path)
            
            print(f"[INFO] Saved precaptured results for [training] {cam.image_name}")
        
        
        # ------------------------------- Testing Camera ------------------------------ #
        precaptured_test_bg_dir = Path(precaptured_mesh_img_path) / mesh_rasterizer_type / "test_mesh_texture"
        precaptured_test_depth_dir = Path(precaptured_mesh_img_path) / mesh_rasterizer_type / "test_mesh_depth"
        
        if frame_subdir is not None:
            (precaptured_test_bg_dir / frame_subdir).mkdir(parents=True, exist_ok=True)
            (precaptured_test_depth_dir / frame_subdir).mkdir(parents=True, exist_ok=True)
        else:
            precaptured_test_bg_dir.mkdir(parents=True, exist_ok=True)
            precaptured_test_depth_dir.mkdir(parents=True, exist_ok=True)
        
        for cam in tqdm(scene.getTestCameras(), desc="Precapturing test backgrounds", unit="camera"):
            bg_save_path = build_precaptured_path(
                precaptured_test_bg_dir,
                cam.image_name,
                ".png",
                frame_subdir=frame_subdir,
            )
            depth_save_path = build_precaptured_path(
                precaptured_test_depth_dir,
                cam.image_name,
                ".pt",
                frame_subdir=frame_subdir,
            )
            
            # Skip if already exists
            if bg_save_path.exists() and depth_save_path.exists():
                print(f"\t[INFO] Skipping {cam.image_name}, already exists.")
                continue
            
            # [DONE] fix black background issue in precapture stage
            # didn't pass bg=[0,0,0] into the mesh_renderer_pytorch3d()
            # Render background and depth
            
            bg_color = (1,1,1) if dataset.white_background else (0,0,0)
            render_pkg = render(cam, gaussians, pipe, 
                                bg_color=None, bg_depth=None, 
                                textured_mesh=scene.textured_mesh,
                                mesh_background_color=bg_color,
                                mesh_rasterizer_type=mesh_rasterizer_type
                                )
            
            # Save background image
            bg_image = render_pkg["bg_color"].detach().clamp(0, 1).cpu()
            bg_image_pil = TF.to_pil_image(bg_image)
            bg_image_pil.save(bg_save_path)
            
            # Save depth image
            bg_depth = render_pkg["bg_depth"].detach().cpu()
            torch.save(bg_depth, depth_save_path)
            
            print(f"[INFO] Saved precaptured results for [testing] {cam.image_name}")
        
        
        print("[INFO] Warmup stage complete.")
        return # [NOTE] early return for warmup-only stage     
    
    
    print("[INFO] Finished Warm-Up, Start Training..." )
    #  ------------------------Warm Up Done--------------------------- #
    
    
    # [NOTE] the background fetched in this part is for network GUI debugger only 
    # (not used by us, and not used by training loop)
    # --------------------------- Load background image -------------------------- #
    background_image_path = f"/home/frozzzen/Documents/Github/layered-mesh-gaussian/data/hotdog/mesh/pytorch3d/mesh_texture/r_0.png"
    img = Image.open(background_image_path).convert("RGB")
    # viewpoint_camera_height = 800
    # viewpoint_camera_width = 800
    viewpoint_camera_height = scene.getTrainCameras()[0].image_height
    viewpoint_camera_width = scene.getTrainCameras()[0].image_width
    img = img.resize((viewpoint_camera_width, viewpoint_camera_height), Image.BILINEAR) # fixed issue, should be (W, H)
    transform = T.Compose([
        T.ToTensor(),  # [0, 255] → [0.0, 1.0], shape (3, H, W)
    ])
    background = transform(img).to(torch.float32).cuda()
    
    # ----------------------------- Load depth image ----------------------------- #
    background_depth_pt_path = "/home/frozzzen/Documents/Github/layered-mesh-gaussian/data/hotdog/mesh/pytorch3d/mesh_depth/r_0.pt"
    background_depth = torch.load(background_depth_pt_path).unsqueeze(0)
    # <<<< [YC]

    # ---------------------------------------------------------------------------- #
    #                              Start Training Loop                             #
    # ---------------------------------------------------------------------------- #
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    # [TODO] test on a gs_type=gs
    
    if gs_type == "gs_mesh":
        
        if occlusion:
            print("[INFO] DTGS training:: using Depth+Texture+GS rasterizer with occlusion for gs_mesh")
        else:
            print("[INFO] TGS training:: using Texture+GS rasterizer for gs_mesh")
    elif gs_type == "gs":
        print("[INFO] GS training:: using original GS rasterizer for gs")
    else: 
        pass        

    frame_subdir = infer_mesh_frame_subdir(texture_obj_path) if precaptured_mesh_img_path else None
    
    
    for iteration in range(first_iter, opt.iterations + 1):
        os.makedirs(f"{scene.model_path}/xyz", exist_ok=True)
        if save_xyz and (iteration % 5000 == 1 or iteration == opt.iterations):
            torch.save(gaussians.get_xyz, f"{scene.model_path}/xyz/{iteration}.pt")
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            print("[INFO] network_gui connected")
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    print("[INFO] Received custom camera for rendering")
                    # net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image = render(custom_cam, gaussians, pipe, background, background_depth, scaling_modifer)["render"] # [YC] add
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            print(f"[DEBUG] Train:: current SH degree: {gaussians.active_sh_degree}")

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        rand_cam_id = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_cam_id)
        
        # ---------------------------------------------------------------------------- #
        #                                Load Background                               #
        # ---------------------------------------------------------------------------- #
        viewpoint_camera_height = viewpoint_cam.image_height
        viewpoint_camera_width = viewpoint_cam.image_width
        print("[DEBUG] viewpoint_camera_height:", viewpoint_camera_height, "viewpoint_camera_width:", viewpoint_camera_width)
        
        transform = T.Compose([
            T.ToTensor(),  # [0, 255] → [0.0, 1.0], shape (3, H, W)
        ])
        
        # ------------------------------ Mesh background ----------------------------- #
        
        if precaptured_mesh_img_path:
            cached_bg_path = build_precaptured_path(
                Path(precaptured_mesh_img_path) / mesh_rasterizer_type / "mesh_texture",
                viewpoint_cam.image_name,
                ".png",
                frame_subdir=frame_subdir,
            )
            if cached_bg_path.exists():
                img = Image.open(cached_bg_path).convert("RGB")
                img = img.resize((viewpoint_camera_width, viewpoint_camera_height), Image.BILINEAR)  # (W, H)
                bg = transform(img).to(torch.float32).cuda()
            #     if iteration % debug_freq == 0:
            #         print(f"[INFO] [DEBUG] Loaded cached background image from {cached_bg_path}")
                
            # else:
            #     if iteration % debug_freq == 0:
            #         print(f"[INFO] Cached background image not found at {cached_bg_path}, skipping...")
        
        # ------------------------------ Mesh depth background ----------------------------- #
        # [TODO] perhaps prefetch everything at the start of training?
        if precaptured_mesh_img_path:
            cached_bg_depth_path = build_precaptured_path(
                Path(precaptured_mesh_img_path) / mesh_rasterizer_type / "mesh_depth",
                viewpoint_cam.image_name,
                ".pt",
                frame_subdir=frame_subdir,
            )
            if cached_bg_depth_path.exists():
                bg_depth = torch.load(cached_bg_depth_path).unsqueeze(0).to("cuda")
            #     if iteration % debug_freq == 0:
            #         print(f"[INFO] [DEBUG] Loaded cached depth image from {cached_bg_depth_path}")
                
            # else:
            #     if iteration % debug_freq == 0:
            #         print(f"[INFO] Cached depth image not found at {cached_bg_depth_path}, skipping...")


        # ------------------------------ Pure background ----------------------------- #
        pure_bg_template = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        pure_bg = torch.tensor(pure_bg_template, dtype=torch.float32, device="cuda").view(3, 1, 1)
        pure_bg = pure_bg.expand(3, viewpoint_camera_height, viewpoint_camera_width) # (H, W)
        
        # --------------------- Pure depth background (all zeros) -------------------- #
        pure_bg_depth = torch.full((1, viewpoint_camera_height, viewpoint_camera_width), 0, dtype=torch.float32, device="cuda")
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # >>>> [YC]
        # -------------------------- Rendering for training -------------------------- #
        if gs_type == "gs":
            render_pkg = render(viewpoint_cam, gaussians, pipe, 
                                bg_color=pure_bg, bg_depth=pure_bg_depth)
        elif gs_type == "gs_mesh":
            if occlusion: # [YC] use occlusion diff-gaussian-rasterizer for training
                render_pkg = render(viewpoint_cam, gaussians, pipe, 
                                    bg_color=bg, bg_depth=bg_depth, 
                                    textured_mesh=scene.textured_mesh)
                # [YC] if there bg or bg_depth isn't provided, but textured mesh is given, it will use mesh renderer to produce bg and bg_depth
                
            else: # [YC] use original diff-gaussian-rasterizer for training
                render_pkg = render(viewpoint_cam, gaussians, pipe, 
                                    bg_color=bg, bg_depth=pure_bg_depth, # [YC] no occlusion handling, use pure_bg_depth
                                    textured_mesh=scene.textured_mesh)
                
                
        image = render_pkg["render"]
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # -------------------------- Load ground truth image ------------------------- #
        
        if iteration % debug_freq == 0:
            print(f"[DEBUG] Training Iteration {iteration}, viewpoint: {viewpoint_cam.image_name}")
        
        
        # [DONE] fix hardcoded old path and handle black/white background
        gt_image = viewpoint_cam.original_image.cuda()
         
        # -------------------------- Save debugging visualizations ------------------------- #
        if debugging:
            # ------------------- Change Tensor to PIL.Image for saving ------------------ #
            if iteration % debug_freq == 0:
                # ---------------------------- Ground truth image ---------------------------- #
                gt_img_to_save = gt_image.detach().clamp(0, 1).cpu()
                gt_img_pil = TF.to_pil_image(gt_img_to_save)
                gt_img_pil.save(check_path/f"{iteration}_gt.png")
                
                # ------------------------ Render image from training ------------------------ #
                img_to_save = image.detach().clamp(0, 1).cpu()
                img_pil = TF.to_pil_image(img_to_save)
                img_pil.save(check_path/f"{iteration}_training.png")
                
                # ----------------------- Background image for training ---------------------- #
                img_to_save = render_pkg["bg_color"].detach().clamp(0, 1).cpu()
                img_pil = TF.to_pil_image(img_to_save)
                img_pil.save(check_path/f"{iteration}_training_mesh_bg.png")
                
                if gs_type == "gs_mesh":
                    # ------------- Render mesh background and depth background ------------- #
                    # [1, 1, 1]
                    render_mesh_with_depth = render(viewpoint_cam, gaussians, pipe, 
                                                    bg_color=bg, bg_depth=bg_depth,
                                                    textured_mesh=scene.textured_mesh)
                    _image = render_mesh_with_depth["render"]

                    img_to_save = _image.detach().clamp(0, 1).cpu()
                    img_pil = TF.to_pil_image(img_to_save)
                    img_pil.save(check_path/f"{iteration}_gs_mesh_with_depth.png")
                
                    # ------------- Render mesh background and fake depth background ------------- #
                    # [0, 1, 1]
                    render_mesh_wo_depth = render(viewpoint_cam, gaussians, pipe, 
                                                    bg_color=bg, bg_depth=pure_bg_depth,
                                                    textured_mesh=scene.textured_mesh)
                    _image = render_mesh_wo_depth["render"]

                    img_to_save = _image.detach().clamp(0, 1).cpu()
                    img_pil = TF.to_pil_image(img_to_save)
                    img_pil.save(check_path/f"{iteration}_gs_mesh_wo_depth.png")

                    # ------------- Render pure background and mesh depth background ------------- #
                    # [1, 0, 1]
                    render_pure_with_depth = render(viewpoint_cam, gaussians, pipe, 
                                                    bg_color=pure_bg, bg_depth=bg_depth,
                                                    textured_mesh=scene.textured_mesh)
                    _image = render_pure_with_depth["render"]
                    
                    img_to_save = _image.detach().clamp(0, 1).cpu()
                    img_pil = TF.to_pil_image(img_to_save)
                    img_pil.save(check_path/f"{iteration}_gs_pure_with_depth.png")
                
                    # ------------- Render pure background and fake depth background ------------- #
                    # [1, 1, 1]
                    render_pure_wo_depth = render(viewpoint_cam, gaussians, pipe, 
                                                bg_color=pure_bg, bg_depth=pure_bg_depth,
                                                textured_mesh=None)
                    _image = render_pure_wo_depth["render"]
                    
                    img_to_save = _image.detach().clamp(0, 1).cpu()
                    img_pil = TF.to_pil_image(img_to_save)
                    img_pil.save(check_path/f"{iteration}_gs_pure_wo_depth.png")
            # <<<< [YC]
            
        # Compute loss and backpropagate
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # # Brutally adjust loss, but keeping the backward information
        # Ll1 = 0.0
        # loss = image.mean() * 0.0 + 0.5

        loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # [good to have] enable training report to observe loss and metrics during training
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
            #                 testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            #! [YC] note: original "gs_mesh" will skip densification
            # Densification
            if (args.gs_type == "gs") or (args.gs_type == "gs_flat"):
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                         radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                                                    size_threshold)

                    if iteration % opt.opacity_reset_interval == 0 or (
                            dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()
            # >>>> [YC] add
            elif args.gs_type == "gs_mesh":
                pass
            # <<<< [YC] add

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        if hasattr(gaussians, 'update_alpha'):
            gaussians.update_alpha()
        if hasattr(gaussians, 'prepare_scaling_rot'):
            gaussians.prepare_scaling_rot()

def load_textured_mesh(dataset, texture_obj_path: str) -> Meshes:
    """
    Load a textured 3D mesh from the given path for background rendering.
    
    This function loads mesh of SuGaR (.obj) or Colmap (.ply) format (or others, add if needed)
    and converts it to a PyTorch3D Meshes object on CUDA
    
    Args:
        dataset: Dataset configuration containing mesh_type attribute.
                Should have mesh_type in ['sugar', 'colmap', ...].
        texture_obj_path: Path to the mesh file. If empty string, raises AssertionError.
    Returns:
        Meshes: A PyTorch3D Meshes object on CUDA
    Raises:
        AssertionError: If texture_obj_path is empty or mesh type is unsupported.
        AssertionError: If file extension doesn't match expected format.
    """
    
    assert texture_obj_path != "", "[ERROR] texture_obj_path cannot be empty"
    textured_mesh = None
    mesh_type = dataset.mesh_type
    if texture_obj_path != "":
        print("[INFO] Loading textured mesh for background rendering...")
        
        if mesh_type == "sugar": # From SuGaR
            assert texture_obj_path.lower().endswith(".obj"), "[ERROR] SuGaR mesh should be .obj file!"
            textured_mesh = load_objs_as_meshes([texture_obj_path]).to("cuda")
            textured_mesh = ensure_mesh_has_texture(textured_mesh)
             
        elif mesh_type == "colmap" or mesh_type == "milo": 
            # From Colmap, download from https://nerfbaselines.github.io/
            assert texture_obj_path.lower().endswith(".ply"), "[ERROR] Colmap mesh should be .ply file!"
            mesh_tm = trimesh.load(texture_obj_path, force='mesh', process=False)
            verts = torch.tensor(mesh_tm.vertices, dtype=torch.float32)
            faces = torch.tensor(mesh_tm.faces, dtype=torch.int64)
            colors = torch.tensor(mesh_tm.visual.vertex_colors[:, :3], dtype=torch.float32) / 255.0
            
            # Combine into a textured mesh
            textured_mesh = Meshes(
                verts=[verts],
                faces=[faces],
                textures=TexturesVertex(verts_features=[colors])
            ).to("cuda")
        else:
            print("[ERROR] Unknown/Unsupported mesh type!")        
            
    assert textured_mesh is not None, "[ERROR] Textured mesh is not loaded properly!"
    
    return textured_mesh

def load_textured_mesh_for_nvdiffrast(dataset, texture_obj_path: str) -> Meshes:
    return trimesh.load(texture_obj_path, force='mesh', process=False)

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("[INFO] Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("[INFO] Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()




if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--gs_type', type=str, default="gs_mesh")
    parser.add_argument("--num_splats", nargs="+", type=int, default=[2])
    parser.add_argument("--meshes", nargs="+", type=str, default=[])
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000]) # not used
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 20_000, 30_000, 60_000, 90_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--save_xyz", action='store_true')
    
    # >>>> [YC] add
    parser.add_argument('--texture_obj_path', type=str, default="")
    parser.add_argument('--mesh_start', type=int, help="Start index for a mesh filename sequence")
    parser.add_argument('--mesh_end', type=int, help="End index for a mesh filename sequence")
    parser.add_argument('--canonical_frame', type=int, default=None,
                        help="Frame index used as the canonical mesh for temporal gs_mesh training")
    parser.add_argument('--debugging', action='store_true')
    parser.add_argument('--debug_freq', type=int, default=1, help="Iteration of saving debugging images")
    parser.add_argument('--occlusion', action='store_true')
    parser.add_argument('--policy_path', type=str, default="", 
        help="Path to the pre-computed .npy file storing num_gs_per_tri[]. \
        When this is provided, it has higher priority than --alloc_policy; \
        otherwise, will overwrite/recompute")
    
    parser.add_argument('--precaptured_mesh_img_path', type=str, default="",
        help="path to the directory containing precaptured mesh (RGB & D) images for background. \
            should contain mesh_texture/ and mesh_depth/ sub-folders."
        ) # [NOTE] better store alongside mesh file
    # <<<< [YC] add
    
    # use either of the two to set total number of splats (bit budget, or gaussian budget for the whole scene)
    parser.add_argument("--total_splats", type=int, help="Total number of splats to allocate")
    parser.add_argument("--budget_per_tri", type=float, default=1.0, help="set the total number of splats to be this number * number of triangles")
    parser.add_argument("--alloc_policy", type=str, default="area", help="Allocation policy for splats (default: area)")
    parser.add_argument("--warmup_only", action='store_true', help="only run warmup stage and exit, no entering training loop")
    parser.add_argument('--mesh_type', type=str, default="sugar", help="textured mesh type: sugar, colmap, or others")
    
    parser.add_argument("--mesh_rasterizer_type", type=str, default="pytorch3d", 
                        help="which mesh rasterizer to use: pytorch3d or nvdiffrast") 
    parser.add_argument("--canonical_iterations", type=int, default=None,
                        help="Number of iterations for the canonical frame in temporal gs_mesh training")
    parser.add_argument("--temporal_iterations", type=int, default=500,
                        help="Number of fine-tuning iterations for subsequent frames in temporal gs_mesh training")
    parser.add_argument("--sequence_weight_reduction", type=str, default="max",
                        choices=["mean", "max", "mean_max"],
                        help="How to aggregate per-frame triangle weights into one sequence policy")
    parser.add_argument("--recompute_sequence_policy", action="store_true",
                        help="Recompute sequence-aware policy even when --policy_path already exists")
    parser.add_argument("--strict_sequence_topology", action="store_true",
                        help="Require every frame to have the exact same face index array as the first frame")
    parser.add_argument("--temporal_attributes", action="store_true",
                        help="Enable compact neural prediction of Gaussian attribute residuals over time")
    parser.add_argument("--temporal_attr_lr", type=float, default=1e-3,
                        help="Learning rate for compact temporal attribute module")
    parser.add_argument("--temporal_attr_width", type=int, default=64,
                        help="Hidden width of compact temporal attribute MLP")
    parser.add_argument("--temporal_attr_depth", type=int, default=3,
                        help="Hidden depth of compact temporal attribute MLP")
    parser.add_argument("--temporal_attr_latent_dim", type=int, default=8,
                        help="Per-triangle latent dimension for compact temporal attributes")
    parser.add_argument("--temporal_attr_time_frequencies", type=int, default=6,
                        help="Number of sinusoidal time frequencies")
    parser.add_argument("--temporal_start_iter", type=int, default=100,
                        help="Iteration before temporal residuals start training")
    parser.add_argument("--temporal_max_d_uvw", type=float, default=0.05,
                        help="Clamp magnitude for raw UVW residuals")
    parser.add_argument("--temporal_max_d_scaling", type=float, default=0.10,
                        help="Clamp magnitude for log-scaling residuals")
    parser.add_argument("--temporal_max_d_opacity", type=float, default=0.50,
                        help="Clamp magnitude for opacity-logit residuals")
    parser.add_argument("--temporal_max_d_color", type=float, default=0.10,
                        help="Clamp magnitude for DC color residuals")
    parser.add_argument("--temporal_predict_uvw", action="store_true",
                        help="Predict temporal UVW residuals")
    parser.add_argument("--temporal_predict_scaling", action="store_true",
                        help="Predict temporal scaling residuals")
    parser.add_argument("--temporal_predict_opacity", action="store_true",
                        help="Predict temporal opacity residuals")
    parser.add_argument("--temporal_predict_color", action="store_true",
                        help="Predict temporal DC color residuals")
    
    lp = ModelParams(parser) # LoadingParams
    args, _ = parser.parse_known_args(sys.argv[1:])
    lp.num_splats = args.num_splats
    lp.meshes = args.meshes
    lp.gs_type = args.gs_type
    
    # >>>> [Sam] add
    lp.total_splats = args.total_splats
    lp.budget_per_tri = args.budget_per_tri
    lp.alloc_policy = args.alloc_policy 
    lp.warmup_only = args.warmup_only
    lp.mesh_type = args.mesh_type.lower()
    # <<<< [Sam] add
    
    op = optimizationParamTypeCallbacks[args.gs_type](parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    if args.temporal_attributes and not any([
        args.temporal_predict_uvw,
        args.temporal_predict_scaling,
        args.temporal_predict_opacity,
        args.temporal_predict_color,
    ]):
        args.temporal_predict_scaling = True
        args.temporal_predict_opacity = True
        args.temporal_predict_color = True

    args.save_iterations.append(args.iterations)

    print("torch cuda: ", torch.cuda.is_available())
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    if len(args.save_iterations) == 0:
        print("[WARN] No save iterations specified, defaulting to saving at the end of training.")
    
    mesh_paths = resolve_mesh_sequence(args.texture_obj_path, args.mesh_start, args.mesh_end)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    for mesh_path in mesh_paths:
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    use_mesh_sequence = args.gs_type == "gs_mesh" and len(mesh_paths) > 1
    sequence_policy_path = args.policy_path
    if use_mesh_sequence:
        sequence_policy_path = ensure_sequence_policy_file(
            args,
            lp,
            mesh_paths,
            requested_policy_path=args.policy_path,
        )

    if args.warmup_only:
        for mesh_path in mesh_paths:
            run_args = Namespace(**vars(args))
            run_args.texture_obj_path = str(mesh_path)
            if use_mesh_sequence:
                frame_subdir = infer_mesh_frame_subdir(run_args.texture_obj_path)
                if frame_subdir is not None:
                    run_args.model_path = append_subdir(args.model_path, frame_subdir)
                run_args.policy_path = sequence_policy_path

            print(f"[INFO] Warmup mesh: {run_args.texture_obj_path}")
            if run_args.model_path:
                print(f"[INFO] Model output: {run_args.model_path}")
            if run_args.policy_path:
                print(f"[INFO] Policy path: {run_args.policy_path}")

            training(
                run_args.gs_type,
                lp.extract(run_args), op.extract(run_args), pp.extract(run_args),
                run_args.test_iterations, run_args.save_iterations, run_args.checkpoint_iterations,
                run_args.start_checkpoint, run_args.debug_from, run_args.save_xyz,
                texture_obj_path=run_args.texture_obj_path,
                debugging=run_args.debugging, debug_freq=run_args.debug_freq,
                occlusion=run_args.occlusion,
                policy_path=run_args.policy_path,
                precaptured_mesh_img_path=run_args.precaptured_mesh_img_path,
                mesh_rasterizer_type=run_args.mesh_rasterizer_type
            )
    elif use_mesh_sequence:
        canonical_iterations = args.canonical_iterations or args.iterations
        training_sequence(
            gs_type=args.gs_type,
            base_args=args,
            opt=op.extract(args),
            pipe=pp.extract(args),
            mesh_paths=mesh_paths,
            save_xyz=args.save_xyz,
            debugging=args.debugging,
            debug_freq=args.debug_freq,
            occlusion=args.occlusion,
            requested_policy_path=sequence_policy_path,
            precaptured_mesh_img_path=args.precaptured_mesh_img_path,
            mesh_rasterizer_type=args.mesh_rasterizer_type,
            canonical_frame=args.canonical_frame,
            canonical_iterations=canonical_iterations,
            temporal_iterations=args.temporal_iterations,
            model_params=lp,
        )
    else:
        for mesh_path in mesh_paths:
            run_args = Namespace(**vars(args))
            run_args.texture_obj_path = str(mesh_path)

            frame_subdir = infer_mesh_frame_subdir(run_args.texture_obj_path)
            if len(mesh_paths) > 1 and frame_subdir is not None:
                run_args.model_path = append_subdir(args.model_path, frame_subdir)
                run_args.policy_path = append_policy_subdir(args.policy_path, frame_subdir)

            print(f"[INFO] Training mesh: {run_args.texture_obj_path}")
            if run_args.model_path:
                print(f"[INFO] Model output: {run_args.model_path}")
            if run_args.policy_path:
                print(f"[INFO] Policy path: {run_args.policy_path}")

            training(
                run_args.gs_type,
                lp.extract(run_args), op.extract(run_args), pp.extract(run_args),
                run_args.test_iterations, run_args.save_iterations, run_args.checkpoint_iterations,
                run_args.start_checkpoint, run_args.debug_from, run_args.save_xyz,
                # >>>> [YC] add
                texture_obj_path=run_args.texture_obj_path,
                debugging=run_args.debugging, debug_freq=run_args.debug_freq,
                occlusion=run_args.occlusion,
                policy_path=run_args.policy_path,
                precaptured_mesh_img_path=run_args.precaptured_mesh_img_path,
                mesh_rasterizer_type=run_args.mesh_rasterizer_type
                # <<<< [YC] add
            )

    # All done
    print("\n[INFO] Training complete.")
