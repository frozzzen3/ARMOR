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

"""Mesh path resolution, sequence indexing, and textured-mesh loading helpers.

Extracted verbatim from train.py to keep the training entry point focused on
orchestration. Behavior is unchanged.
"""

import re
from pathlib import Path

import numpy as np
import torch
import trimesh
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

from renderer.mesh_renderer.mesh_utils import ensure_mesh_has_texture


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


def sequence_frame_label(mesh_paths):
    first = extract_frame_index(mesh_paths[0])
    last = extract_frame_index(mesh_paths[-1])
    if first is not None and last is not None:
        return f"frames_{first:04d}_{last:04d}"
    return f"{mesh_paths[0].stem}_to_{mesh_paths[-1].stem}"


def normalized_frame_time(mesh_path, mesh_paths):
    if len(mesh_paths) <= 1:
        return 0.0
    try:
        index = mesh_paths.index(Path(mesh_path))
    except ValueError:
        index = 0
    return float(index) / float(len(mesh_paths) - 1)


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
