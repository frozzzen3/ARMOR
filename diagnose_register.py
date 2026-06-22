#
# Standalone diagnostic for the variable-topology mesh tracker.
#
# Feed it two meshes (a source and a target of possibly different topology) and it:
#   - reports mesh defects that make trimesh's nricp singular (degenerate faces,
#     zero-length edges, duplicate vertices, connected components),
#   - runs the tracker(s) and reports how close the registered source lands on the
#     target surface (residual distance, normalized by object size),
#   - compares laplacian vs nricp_amberg vs naive closest-point,
#   - exports source.obj / target.obj / <method>_deformed.obj for visual inspection,
#   - optionally samples points on the source, warps them through the deformation, and
#     measures their residual (this is exactly how bound Gaussians are tracked).
#
# Usage:
#   PYTHONPATH=. python diagnose_register.py --source a.obj --target b.obj
#   PYTHONPATH=. python diagnose_register.py --source a.obj --target b.obj \
#       --methods laplacian nricp_amberg closest_point --out_dir ./register_debug
#

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

from utils.mesh_utils import (
    register_mesh_laplacian,
    register_mesh_nonrigid,
    project_points_to_mesh,
)


def load_mesh(path):
    m = trimesh.load(str(path), force="mesh", process=False)
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def write_obj(path, verts, faces):
    """Write a plain geometry OBJ (vertices + triangular faces only). Explicit so the
    faces are always emitted regardless of trimesh export quirks / texture material refs."""
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    lines = [f"v {x:.8f} {y:.8f} {z:.8f}" for x, y, z in verts]
    lines += [f"f {a} {b} {c}" for a, b, c in (faces + 1)]  # OBJ faces are 1-indexed
    Path(path).write_text("\n".join(lines) + "\n")
    print(f"           wrote {Path(path).name}: {verts.shape[0]} verts, {faces.shape[0]} faces")


def report_defects(name, v, f):
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    # zero-length edges
    e = mesh.edges_unique
    el = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    zero_edges = int((el < 1e-12).sum())
    # degenerate faces (near-zero area)
    areas = mesh.area_faces
    degen = int((areas < 1e-14).sum())
    # duplicate vertices
    try:
        dup = v.shape[0] - int(np.unique(np.round(v, 9), axis=0).shape[0])
    except Exception:
        dup = -1
    try:
        ncomp = int(mesh.body_count)
    except Exception:
        ncomp = -1
    lo, hi = v.min(0), v.max(0)
    print(f"  [{name}] verts={v.shape[0]} faces={f.shape[0]} "
          f"watertight={mesh.is_watertight} components={ncomp}")
    print(f"           zero_length_edges={zero_edges}  degenerate_faces={degen}  duplicate_verts={dup}")
    print(f"           bbox min=[{lo[0]:.3f},{lo[1]:.3f},{lo[2]:.3f}] "
          f"max=[{hi[0]:.3f},{hi[1]:.3f},{hi[2]:.3f}]")
    return mesh


def bbox_str(v):
    lo, hi = v.min(0), v.max(0)
    return (f"min=[{lo[0]:.3f},{lo[1]:.3f},{lo[2]:.3f}] "
            f"max=[{hi[0]:.3f},{hi[1]:.3f},{hi[2]:.3f}]")


def residual_stats(deformed_np, target_mesh, scale):
    _, dist, _ = trimesh.proximity.closest_point(target_mesh, deformed_np)
    d = np.asarray(dist) / scale  # normalize by object size
    return {
        "mean": float(d.mean()), "median": float(np.median(d)),
        "p90": float(np.percentile(d, 90)), "max": float(d.max()),
    }


def coherence_stats(src_v, deformed_np, edges):
    """Relative edge-length change source->deformed. A coherent (near-isometric)
    deformation keeps this low; naive closest-point snapping shreds local structure
    (tangential drift / folds) and inflates it -- this is what surface-distance misses."""
    s = np.linalg.norm(src_v[edges[:, 0]] - src_v[edges[:, 1]], axis=1)
    d = np.linalg.norm(deformed_np[edges[:, 0]] - deformed_np[edges[:, 1]], axis=1)
    typical = np.median(s[s > 1e-9]) if np.any(s > 1e-9) else 1.0  # robust to zero-length edges
    rel = np.abs(d - s) / typical
    return float(rel.mean()), float(np.percentile(rel, 90))


def main():
    ap = argparse.ArgumentParser(description="Diagnose / visualize variable-topology mesh registration")
    ap.add_argument("--source", required=True, help="source mesh (previous frame)")
    ap.add_argument("--target", required=True, help="target mesh (current frame, possibly different topology)")
    ap.add_argument("--methods", nargs="+", default=["laplacian", "nricp_amberg", "closest_point"],
                    choices=["laplacian", "nricp_amberg", "nricp_sumner", "closest_point"])
    ap.add_argument("--out_dir", default="./register_debug")
    ap.add_argument("--no_rigid_prealign", action="store_true")
    ap.add_argument("--sample", type=int, default=20000,
                    help="sample this many points on the source and warp them (Gaussian-style tracking test)")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rigid = not args.no_rigid_prealign

    src_v, src_f = load_mesh(args.source)
    tgt_v, tgt_f = load_mesh(args.target)
    print("=" * 78)
    print("Mesh defect report (why trimesh nricp may be singular):")
    src_mesh = report_defects("source", src_v, src_f)
    tgt_mesh = report_defects("target", tgt_v, tgt_f)
    scale = float(np.linalg.norm(tgt_v.max(0) - tgt_v.min(0))) or 1.0  # bbox diagonal
    print(f"  target bbox diagonal (normalizer) = {scale:.4f}")
    print("=" * 78)

    write_obj(out / "source.obj", src_v, src_f)
    write_obj(out / "target.obj", tgt_v, tgt_f)

    # sample source points for the Gaussian-style warp test (barycentric on a face)
    rng = np.random.default_rng(0)
    fi = rng.integers(0, src_f.shape[0], size=args.sample)
    bary = rng.random((args.sample, 3)); bary /= bary.sum(1, keepdims=True)
    src_pts = np.einsum("ij,ijk->ik", bary, src_v[src_f[fi]])

    src_edges = trimesh.Trimesh(src_v, src_f, process=False).edges_unique
    print(f"{'method':14s} {'time(s)':>8s} {'resid.mean':>11s} {'p90':>9s} {'max':>9s} "
          f"{'edgedist.mean':>13s} {'p90':>9s}  (normalized; edgedist = coherence, lower=better)")
    for method in args.methods:
        t0 = time.time()
        if method == "closest_point":
            # naive baseline: snap each source vertex to nearest target point (no coherence)
            closest, _, _ = trimesh.proximity.closest_point(tgt_mesh, src_v)
            deformed = torch.as_tensor(np.asarray(closest), dtype=torch.float32)
        elif method == "laplacian":
            deformed = register_mesh_laplacian(src_v, src_f, tgt_v, tgt_f, rigid_prealign=rigid)
        else:
            deformed = register_mesh_nonrigid(src_v, src_f, tgt_v, tgt_f, method=method, rigid_prealign=rigid)
        dt = time.time() - t0

        if deformed is None:
            print(f"{method:14s} {dt:8.2f}   FAILED (singular / fell back) -- see warning above")
            continue
        deformed_np = deformed.detach().cpu().numpy().astype(np.float64)
        vs = residual_stats(deformed_np, tgt_mesh, scale)
        ed_mean, ed_p90 = coherence_stats(src_v, deformed_np, src_edges)
        print(f"{method:14s} {dt:8.2f} {vs['mean']:11.5f} {vs['p90']:9.5f} {vs['max']:9.5f} "
              f"{ed_mean:13.5f} {ed_p90:9.5f}")

        print(f"{'':14s}          deformed bbox {bbox_str(deformed_np)}  "
              f"(target bbox {bbox_str(tgt_v)})")
        write_obj(out / f"{method}_deformed.obj", deformed_np, src_f)

        # Gaussian-style warp test: warp the sampled source points via the deformed faces
        warped = np.einsum("ij,ijk->ik", bary, deformed_np[src_f[fi]])
        fid, _, hover = project_points_to_mesh(torch.as_tensor(warped, dtype=torch.float32),
                                               tgt_v, tgt_f)
        h = (hover.abs().cpu().numpy()) / scale
        print(f"{'':14s}          warped sample points -> target surface: "
              f"mean={h.mean():.5f} p90={np.percentile(h,90):.5f} max={h.max():.5f}")

    print("=" * 78)
    print(f"Exported OBJs to {out}/ — open source.obj, target.obj, <method>_deformed.obj together.")
    print("A good tracker: <method>_deformed.obj overlaps target.obj closely (low residual) AND")
    print("looks like a smoothly deformed source.obj (coherent), not a shattered point set.")
    print("If nricp_amberg FAILED while laplacian has low residual, that confirms the singular-")
    print("matrix fallback was the tracking problem and laplacian fixes it.")


if __name__ == "__main__":
    main()
