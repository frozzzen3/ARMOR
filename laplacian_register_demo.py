#!/usr/bin/env python3

import argparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized
from scipy.spatial import cKDTree
import trimesh


def to_numpy(x):
    """
    Convert torch tensor or numpy-like array to numpy.
    """
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except ImportError:
        pass

    return np.asarray(x)


def uniform_graph_laplacian(num_vertices, faces):
    """
    Build a purely combinatorial uniform graph Laplacian.

    L[i, i] = degree(i)
    L[i, j] = -1 if vertex i and j are connected by an edge

    This does not use cotangent weights or edge lengths, so it is more robust
    to degenerate geometry.
    """
    faces = np.asarray(faces, dtype=np.int64)

    # Remove invalid faces.
    valid = np.all((faces >= 0) & (faces < num_vertices), axis=1)
    faces = faces[valid]

    edges = []

    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])

        # Skip fully degenerate triangles.
        if len({a, b, c}) < 3:
            continue

        edges.append((a, b))
        edges.append((b, c))
        edges.append((c, a))

    if len(edges) == 0:
        return sp.csr_matrix((num_vertices, num_vertices), dtype=np.float64)

    # Make edges undirected.
    undirected = set()
    for i, j in edges:
        if i == j:
            continue
        if i > j:
            i, j = j, i
        undirected.add((i, j))

    rows = []
    cols = []
    data = []

    degree = np.zeros(num_vertices, dtype=np.float64)

    for i, j in undirected:
        degree[i] += 1.0
        degree[j] += 1.0

        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([-1.0, -1.0])

    # Diagonal degree entries.
    rows.extend(np.arange(num_vertices))
    cols.extend(np.arange(num_vertices))
    data.extend(degree)

    L = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(num_vertices, num_vertices),
        dtype=np.float64,
    ).tocsr()

    return L


def closest_points_on_target(target_mesh, query_points):
    """
    Find closest points on the target surface.

    Uses trimesh.proximity.closest_point when available.
    If trimesh's spatial index dependency is missing, falls back to nearest
    target vertices using scipy cKDTree.
    """
    try:
        corr, dist, face_id = trimesh.proximity.closest_point(
            target_mesh,
            query_points,
        )
        return np.asarray(corr, dtype=np.float64)
    except Exception as exc:
        print(
            "[WARN] trimesh.proximity.closest_point failed. "
            "Falling back to nearest target vertices."
        )
        print(f"[WARN] Reason: {exc}")

        tree = cKDTree(np.asarray(target_mesh.vertices))
        _, idx = tree.query(query_points)
        return np.asarray(target_mesh.vertices[idx], dtype=np.float64)


def register_mesh_laplacian(
    src_vertices,
    src_faces,
    tgt_vertices,
    tgt_faces,
    rigid_prealign=True,
    reg_schedule=(2.0, 1.0, 0.5, 0.2, 0.1),
):
    """
    Laplacian-regularized non-rigid registration of a source mesh to a target mesh.

    The source and target may have different topology.

    Returns
    -------
    deformed : np.ndarray, shape [N_src, 3]
        Deformed source vertices in the original source vertex ordering.
    """
    src_v = to_numpy(src_vertices).astype(np.float64)
    src_f = to_numpy(src_faces).astype(np.int64)

    tgt_v = to_numpy(tgt_vertices).astype(np.float64)
    tgt_f = to_numpy(tgt_faces).astype(np.int64)

    target = trimesh.Trimesh(vertices=tgt_v, faces=tgt_f, process=False)
    deformed = src_v.copy()

    if rigid_prealign:
        try:
            matrix, _, _ = trimesh.registration.icp(
                deformed,
                target,
                max_iterations=20,
                reflection=False,
                scale=False,
            )
            deformed = trimesh.transformations.transform_points(
                deformed,
                matrix,
            )
            print("[INFO] Rigid ICP pre-alignment finished.")
        except Exception as exc:
            print(f"[WARN] Rigid pre-align failed: {exc}")
            print("[WARN] Continuing without rigid pre-alignment.")

    L = uniform_graph_laplacian(src_v.shape[0], src_f)
    LTL = L.T.dot(L).tocsc()

    I = sp.identity(src_v.shape[0], format="csc", dtype=np.float64)

    for step, reg in enumerate(reg_schedule):
        print(f"[INFO] Step {step + 1}/{len(reg_schedule)}, reg={reg}")

        # Closest points from current source position to target surface.
        corr = closest_points_on_target(target, deformed)

        # Solve:
        #   (I + reg * L^T L) V' = corr
        #
        # This is solved independently for x, y, and z.
        A = (I + float(reg) * LTL).tocsc()
        solve = factorized(A)

        deformed = np.stack(
            [
                solve(corr[:, 0]),
                solve(corr[:, 1]),
                solve(corr[:, 2]),
            ],
            axis=1,
        )

        if not np.isfinite(deformed).all():
            raise ValueError("Non-finite registration result.")

    return deformed


def load_mesh(path):
    """
    Load a mesh file with trimesh.

    Supports common mesh formats such as OBJ, PLY, STL, GLB.
    """
    mesh = trimesh.load(path, force="mesh", process=False)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values()]
        )

    if mesh.vertices is None or mesh.faces is None:
        raise ValueError(f"Failed to load valid mesh from {path}")

    return mesh


def save_deformed_mesh(source_mesh, deformed_vertices, out_path):
    """
    Save the deformed source mesh.

    The source topology is preserved.
    """
    out_mesh = trimesh.Trimesh(
        vertices=deformed_vertices,
        faces=source_mesh.faces,
        process=False,
    )
    out_mesh.export(out_path)
    print(f"[INFO] Saved deformed mesh to: {out_path}")


def create_demo_meshes():
    """
    Create a simple demo where the source and target have different topology.

    Source: low-resolution sphere.
    Target: higher-resolution deformed sphere.
    """
    source = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    target = trimesh.creation.icosphere(subdivisions=4, radius=1.0)

    v = np.asarray(target.vertices).copy()

    # Make the target non-spherical.
    x = v[:, 0]
    y = v[:, 1]
    z = v[:, 2]

    v[:, 0] = 1.15 * x
    v[:, 1] = 0.85 * y
    v[:, 2] = z + 0.18 * np.sin(3.0 * x) * np.cos(3.0 * y)

    # Add a small translation so rigid pre-alignment has something to fix.
    v += np.array([0.25, -0.10, 0.15])

    target.vertices = v

    source.export("demo_source.obj")
    target.export("demo_target.obj")

    print("[INFO] Wrote demo_source.obj and demo_target.obj")

    return source, target


def show_scene(source_mesh, target_mesh, deformed_vertices):
    """
    Visualize source, target, and deformed result with trimesh.

    Blue/green/red colors are assigned through vertex colors.
    """
    src = source_mesh.copy()
    tgt = target_mesh.copy()
    deform = trimesh.Trimesh(
        vertices=deformed_vertices,
        faces=source_mesh.faces,
        process=False,
    )

    src.visual.vertex_colors = [80, 80, 255, 120]
    tgt.visual.vertex_colors = [80, 255, 80, 120]
    deform.visual.vertex_colors = [255, 80, 80, 180]

    scene = trimesh.Scene()
    scene.add_geometry(src, node_name="source_original_blue")
    scene.add_geometry(tgt, node_name="target_green")
    scene.add_geometry(deform, node_name="deformed_source_red")

    scene.show()


def main():
    parser = argparse.ArgumentParser(
        description="Laplacian non-rigid mesh registration demo."
    )

    parser.add_argument(
        "source",
        nargs="?",
        help="Path to source mesh, e.g. source.obj",
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Path to target mesh, e.g. target.obj",
    )
    parser.add_argument(
        "--out",
        default="deformed.obj",
        help="Output path for deformed source mesh.",
    )
    parser.add_argument(
        "--no-rigid",
        action="store_true",
        help="Disable rigid ICP pre-alignment.",
    )
    parser.add_argument(
        "--reg",
        nargs="+",
        type=float,
        default=[2.0, 1.0, 0.5, 0.2, 0.1],
        help="Regularization schedule, e.g. --reg 5 2 1 0.5 0.1",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show source, target, and deformed result.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run built-in demo instead of loading input meshes.",
    )

    args = parser.parse_args()

    if args.demo:
        source_mesh, target_mesh = create_demo_meshes()
    else:
        if args.source is None or args.target is None:
            raise ValueError(
                "Please provide source and target mesh paths, "
                "or use --demo."
            )

        source_mesh = load_mesh(args.source)
        target_mesh = load_mesh(args.target)

    deformed_vertices = register_mesh_laplacian(
        source_mesh.vertices,
        source_mesh.faces,
        target_mesh.vertices,
        target_mesh.faces,
        rigid_prealign=not args.no_rigid,
        reg_schedule=args.reg,
    )

    save_deformed_mesh(source_mesh, deformed_vertices, args.out)

    if args.show:
        show_scene(source_mesh, target_mesh, deformed_vertices)


if __name__ == "__main__":
    main()