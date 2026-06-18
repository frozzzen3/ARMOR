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
import trimesh.proximity  # noqa: F401  (ensure submodule is importable)
import trimesh.registration  # noqa: F401
import trimesh.triangles  # noqa: F401
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


def _to_numpy(array):
    if torch.is_tensor(array):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def project_points_to_mesh(points_xyz, vertices, faces):
    """Closest-point projection of points onto a mesh surface.

    For each input point this returns the id of the containing face, the
    barycentric coordinates of the nearest surface point inside that face, and
    the signed normal distance ("hover") of the point from that surface point.

    Used by topology-aware re-binding: a persistent Gaussian's warped center is
    snapped back onto the (arbitrary-topology) current-frame mesh to recover its
    new triangle assignment and logical coordinates.

    Args:
        points_xyz: [G, 3] query points (tensor or array).
        vertices:   [V, 3] mesh vertices.
        faces:      [F, 3] mesh faces (vertex indices).

    Returns:
        (face_id [G] long, bary [G, 3] float, signed_hover [G] float) as CUDA tensors.
    """
    points_np = _to_numpy(points_xyz).astype(np.float64)
    verts_np = _to_numpy(vertices).astype(np.float64)
    faces_np = _to_numpy(faces).astype(np.int64)

    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
    closest, _distance, face_id = trimesh.proximity.closest_point(mesh, points_np)
    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[face_id], closest)

    normals = mesh.face_normals[face_id]
    signed_hover = np.einsum("ij,ij->i", points_np - closest, normals)

    device = points_xyz.device if torch.is_tensor(points_xyz) else "cuda"
    return (
        torch.as_tensor(face_id, dtype=torch.long, device=device),
        torch.as_tensor(bary, dtype=torch.float32, device=device),
        torch.as_tensor(signed_hover, dtype=torch.float32, device=device),
    )


def _uniform_graph_laplacian(num_vertices, faces):
    """Purely combinatorial graph Laplacian L = D - A from mesh edges (no edge-length
    weighting), so it is well defined even with zero-length / degenerate edges."""
    import scipy.sparse as sp

    f = np.asarray(faces, dtype=np.int64)
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    vals = np.ones(rows.shape[0], dtype=np.float64)
    adj = sp.coo_matrix((vals, (rows, cols)), shape=(num_vertices, num_vertices)).tocsr()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    return (sp.diags(deg) - adj).tocsc()


def register_mesh_laplacian(src_vertices, src_faces, tgt_vertices, tgt_faces,
                            rigid_prealign=True, reg_schedule=(2.0, 1.0, 0.5, 0.2, 0.1)):
    """Robust Laplacian-regularized non-rigid registration of a source mesh onto a
    target mesh of (possibly) different topology.

    Unlike trimesh's cotangent-based nricp (which divides by edge length and produces a
    singular system on meshes with zero-length edges or isolated vertices), this uses a
    purely combinatorial uniform graph Laplacian and an identity data term, so the solve
    ``(I + reg * LᵀL) V' = corr`` is always SPD and solvable. It runs a few closest-point
    iterations with annealed smoothness (stiff -> flexible), deforming the source
    coherently onto the target so Gaussians bound to the source warp smoothly.

    Returns deformed source vertices ``[V_src, 3]`` (original ordering) as a CUDA tensor,
    or ``None`` on failure so the caller can fall back.
    """
    try:
        import scipy.sparse as sp
        from scipy.sparse.linalg import factorized

        src_v = _to_numpy(src_vertices).astype(np.float64)
        src_f = _to_numpy(src_faces).astype(np.int64)
        tgt_v = _to_numpy(tgt_vertices).astype(np.float64)
        tgt_f = _to_numpy(tgt_faces).astype(np.int64)
        device = src_vertices.device if torch.is_tensor(src_vertices) else "cuda"

        target = trimesh.Trimesh(vertices=tgt_v, faces=tgt_f, process=False)
        deformed = src_v.copy()

        if rigid_prealign:
            try:
                matrix, _, _ = trimesh.registration.icp(deformed, target, max_iterations=20,
                                                        reflection=False, scale=False)
                deformed = trimesh.transformations.transform_points(deformed, matrix)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] register_mesh_laplacian: rigid pre-align failed ({exc}); skipping it.")

        lap = _uniform_graph_laplacian(src_v.shape[0], src_f)
        ltl = lap.T.dot(lap).tocsc()  # LᵀL (symmetric PSD)
        identity = sp.identity(src_v.shape[0], format="csc")

        for reg in reg_schedule:
            corr, _dist, _fid = trimesh.proximity.closest_point(target, deformed)
            corr = np.asarray(corr, dtype=np.float64)
            solve = factorized((identity + float(reg) * ltl).tocsc())
            deformed = np.stack([solve(corr[:, c]) for c in range(3)], axis=1)

        if not np.isfinite(deformed).all():
            raise ValueError("non-finite registration result")
        return torch.as_tensor(deformed, dtype=torch.float32, device=device)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] register_mesh_laplacian failed ({exc}); caller should fall back.")
        return None


def register_mesh_nonrigid(src_vertices, src_faces, tgt_vertices, tgt_faces,
                           method="nricp_amberg", rigid_prealign=True, steps=None):
    """Non-rigidly register a source mesh onto a target mesh of (possibly)
    different topology.

    Returns the deformed source vertices ``[V_src, 3]`` expressed in the target
    frame (aligned to the original ``src_vertices`` ordering), so the source's own
    faces/barycentric coordinates can be reused to warp anything bound to the source
    surface onto the target. This is the "register" half of the register-then-snap
    tracking used for Gaussian re-binding across mesh topology changes.

    Robustness: the solve runs on a *cleaned* copy of the source (duplicate vertices
    merged, degenerate/duplicate faces dropped) because trimesh's non-rigid ICP
    divides by edge length and produces a singular system on meshes with zero-length
    edges (common in textured/UV-seam meshes). The per-vertex deformation is mapped
    back onto the original vertex ordering by nearest position. Returns ``None`` if
    registration fails, so callers can fall back to a simpler tracker.

    Args:
        method: "nricp_amberg" (default) or "nricp_sumner".
        rigid_prealign: run a rigid ICP first so non-rigid ICP starts near the
            target (recommended for large inter-frame motion).
        steps: optional per-stage stiffness schedule forwarded to trimesh.
    """
    if method not in ("nricp_amberg", "nricp_sumner"):
        raise ValueError(f"Unknown non-rigid registration method: {method}")

    src_v = _to_numpy(src_vertices).astype(np.float64)
    src_f = _to_numpy(src_faces).astype(np.int64)
    tgt_v = _to_numpy(tgt_vertices).astype(np.float64)
    tgt_f = _to_numpy(tgt_faces).astype(np.int64)
    device = src_vertices.device if torch.is_tensor(src_vertices) else "cuda"

    try:
        from scipy.spatial import cKDTree

        target = trimesh.Trimesh(vertices=tgt_v, faces=tgt_f, process=False)
        # process=True merges coincident vertices and drops degenerate/duplicate faces,
        # removing the zero-length edges that make trimesh's nricp stiffness singular.
        source = trimesh.Trimesh(vertices=src_v.copy(), faces=src_f.copy(), process=True)
        if len(source.faces) == 0 or len(source.vertices) == 0:
            raise ValueError("source mesh degenerate after cleaning")

        # map each original source vertex to its nearest surviving cleaned vertex
        clean_orig = np.asarray(source.vertices, dtype=np.float64).copy()
        _, idx = cKDTree(clean_orig).query(src_v, k=1)

        if rigid_prealign:
            try:
                # constrain to a true rigid transform: trimesh's ICP otherwise allows
                # scale AND reflection (negative-determinant flips), which corrupts the
                # source and makes the subsequent non-rigid solve singular.
                matrix, _, _ = trimesh.registration.icp(
                    source.vertices, target, max_iterations=20,
                    reflection=False, scale=False,
                )
                source.apply_transform(matrix)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] register_mesh_nonrigid: rigid pre-align failed ({exc}); skipping it.")

        if method == "nricp_amberg":
            deformed = trimesh.registration.nricp_amberg(source, target, steps=steps)
        else:
            deformed = trimesh.registration.nricp_sumner(source, target, steps=steps)

        deformed = np.asarray(deformed, dtype=np.float64)
        if deformed.shape[0] != clean_orig.shape[0] or not np.isfinite(deformed).all():
            raise ValueError("registration produced an invalid result")

        deformed_original = deformed[idx]  # back to original src vertex ordering
        return torch.as_tensor(deformed_original, dtype=torch.float32, device=device)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] register_mesh_nonrigid failed ({exc}); caller should fall back.")
        return None


def write_mesh_obj(vertices: torch.tensor, faces: torch.tensor, filepath, verbose=False):
    """Simple save vertices and face as an obj file."""
    vertices = vertices.detach().cpu().numpy()
    with open(filepath, 'w') as fp:
        for v in vertices:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    if verbose:
        print('mesh saved to: ', f'{filepath}.obj')
