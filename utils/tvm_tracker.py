"""Orchestrates the external ARAP-volume-tracking + TVM-editing pipeline as the
variable-topology rebinding tracker.

Pipeline (per the submodules):
  1. dotnet <arap>/bin/Client.dll <config.xml>          -> volume tracking, per-frame centers
  2. python <arap>/get_transformation.py ...            -> per-pair dual-quaternion transforms
  3. <tvm_editor_exe> s t <meshDir> <centerDir> <outDir> -> deformed source mesh

This wrapper stages the training mesh sequence into the layout those tools expect,
runs the tracking once over the whole range, then deforms per frame-pair on demand
(cached). The deformed mesh (source topology, source vertex order, target pose) is
loaded and transformed into the dataset reader frame so it aligns 1:1 with the
Gaussian model's `vertices` and can be used directly as `deformed_src` in
`GaussianMeshModel.retrack_to_mesh`.

All external calls are wrapped; failures return None so the caller can fall back.
"""

import os
import re
import sys
import subprocess
from pathlib import Path

import numpy as np
import torch
import trimesh

from utils.mesh_utils import extract_frame_index


def load_mesh_in_reader_frame(path, device="cuda", c=1):
    """Load an OBJ and apply the SAME transform the mesh dataset reader applies to
    Gaussian-mesh vertices (rotate(-pi/2, X) then [0,2,1]-swap + negate Y), so the
    result is in the reader frame and aligns 1:1 (same vertex order) with the model's
    `self.vertices`. Mirrors scene/mesh_dataset_readers.readNerfSyntheticMeshInfo."""
    from scene.mesh_dataset_readers import transform_vertices_function

    mesh = trimesh.load(str(path), force="mesh", process=False)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(
        angle=-np.pi / 2, direction=[1, 0, 0], point=[0, 0, 0]
    ))
    verts = transform_vertices_function(torch.tensor(np.asarray(mesh.vertices)), c=c)
    return verts.to(device=device, dtype=torch.float32)


def read_obj_vertices(path):
    """Read raw `v` lines of an OBJ in file order (no trimesh processing). This matches
    TVMEditor's vertex ordering/count, which can differ from trimesh's (trimesh expands
    vertices at texture seams), so source and TVM-deformed OBJs correspond 1:1 here."""
    verts = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
    return np.asarray(verts, dtype=np.float64)


def _to_reader_frame_np(verts_np, device="cuda", c=1):
    """Apply the reader's vertex transform (rotate(-pi/2,X) then [0,2,1]-swap + negate Y)
    to a raw vertex array. The transform is linear, so it commutes with differences
    (displacements transform correctly)."""
    from scene.mesh_dataset_readers import transform_vertices_function

    rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0], [0, 0, 0])[:3, :3]
    v = np.asarray(verts_np, dtype=np.float64) @ rot.T
    v = transform_vertices_function(torch.tensor(v), c=c)
    return v.to(device=device, dtype=torch.float32)


def read_obj_faces(path):
    """Read raw triangular `f` faces of an OBJ (first index of each `v/vt/vn` corner,
    0-based, file order). Matches the geometric-vertex (`v`) faces TVMEditor uses."""
    faces = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("f "):
                corners = line.split()[1:4]
                faces.append([int(c.split("/")[0]) - 1 for c in corners])
    return np.asarray(faces, dtype=np.int64)


def tvm_warped_centers(deformed_path, source_path, gaussians, device="cuda", tol=1e-3):
    """Warp Gaussian centers using the TVM deformation, index-aligned at the FACE level.

    The Gaussian binding (`triangle_indices` = face id, `alpha` = barycentric) is shared
    across frames, and the deformed OBJ keeps the source's faces (same count + order). So
    each Gaussian's warped center is `alpha · deformed_face_vertices`, read directly from
    the deformed OBJ's raw `v`/`f` — exact, no vertex-count alignment and no nearest-
    neighbor/registration. The source OBJ is used only for a self-check that its raw faces
    reproduce the currently-bound surface (validates face/corner order + frame); on any
    mismatch we return None to fall back. Returns [G,3] or None.
    """
    try:
        verts_d = read_obj_vertices(deformed_path)
        faces_d = read_obj_faces(deformed_path)
        verts_s = read_obj_vertices(source_path)
        faces_s = read_obj_faces(source_path)
        if verts_d.shape[0] == 0 or faces_d.shape[0] == 0 or not np.array_equal(faces_d, faces_s):
            print("[WARN] TVM warp: source/deformed faces differ or empty; falling back.")
            return None

        tri_idx = gaussians.triangle_indices.to(device=device, dtype=torch.long)
        faces_t = torch.as_tensor(faces_d, dtype=torch.long, device=device)
        if int(tri_idx.max()) >= faces_t.shape[0]:
            print(f"[WARN] TVM warp: face count {faces_t.shape[0]} <= triangle index "
                  f"{int(tri_idx.max())}; falling back.")
            return None

        src_r = _to_reader_frame_np(verts_s, device=device)
        dfm_r = _to_reader_frame_np(verts_d, device=device)
        alpha = gaussians.alpha.to(device=device).unsqueeze(1)  # [G,1,3]
        gtri = faces_t[tri_idx]                                  # [G,3] raw vertex ids

        # self-check: source raw faces must reproduce the bound (reader) surface exactly
        src_surface = torch.bmm(alpha, src_r[gtri]).squeeze(1)
        ref_surface = torch.bmm(alpha, gaussians.triangles[tri_idx]).squeeze(1)
        max_err = torch.max(torch.abs(src_surface - ref_surface)).item()
        # `not (max_err <= tol)` also catches NaN: a corrupted binding (NaN alpha
        # from an earlier degenerate-face projection) makes max_err NaN, and a bare
        # `max_err > tol` is False for NaN, which would let the garbage through.
        if not (max_err <= tol):
            print(f"[WARN] TVM warp: source raw mesh does not reproduce bound surface "
                  f"(max err {max_err:.4g} vs tol {tol}); face order/frame mismatch or "
                  f"corrupted binding, falling back.")
            return None

        warped = torch.bmm(alpha, dfm_r[gtri]).squeeze(1)       # [G,3] warped centers
        if not torch.isfinite(warped).all():
            print("[WARN] TVM warp: warped centers contain non-finite values "
                  "(corrupted binding); falling back.")
            return None
        return warped
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] tvm_warped_centers failed ({exc}); falling back.")
        return None


class TvmTracker:
    def __init__(self, arap_dir, tvm_editor_exe, config_template, mesh_paths, work_dir,
                 point_count=2000, vg_resolution=512, dotnet="dotnet", prefix="frame_"):
        self.arap_dir = Path(arap_dir).resolve()
        self.client_dll = self.arap_dir / "bin" / "Client.dll"
        self.get_transform_py = self.arap_dir / "get_transformation.py"
        # Resolve to absolute paths: subprocesses run with cwd set to each tool's own
        # directory, so relative paths (e.g. "submodules/...") would not resolve.
        self.tvm_editor_exe = str(Path(tvm_editor_exe).resolve())
        self.config_template = Path(config_template).resolve()
        self.work_dir = Path(work_dir).resolve()
        self.data_dir = self.work_dir / "data"
        self.out_dir = self.work_dir / "out"
        self.deformed_dir = self.out_dir / "deformed"
        self.point_count = int(point_count)
        self.vg_resolution = int(vg_resolution)
        self.dotnet = dotnet
        self.prefix = prefix

        self.frame_to_path = {}
        for p in mesh_paths:
            idx = extract_frame_index(Path(p))
            if idx is not None:
                self.frame_to_path[idx] = Path(p)
        self.indices = sorted(self.frame_to_path)
        self._tracked_pairs = set()

    # ------------------------------------------------------------------ helpers
    def _run(self, cmd, cwd, desc):
        cmd = [str(x) for x in cmd]
        print(f"[INFO] TvmTracker: {desc}: {' '.join(cmd)}  (cwd={cwd})")
        res = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"{desc} failed (rc={res.returncode}).\nstdout:\n{res.stdout[-1500:]}\n"
                f"stderr:\n{res.stderr[-1500:]}"
            )

    def _stage_meshes(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for idx, path in self.frame_to_path.items():
            link = self.data_dir / f"{self.prefix}{idx:03d}.obj"
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(path.resolve(), link)

    def _write_config(self, first=None, last=None):
        """Write the ARAP config. With `first`/`last` the tracking range is restricted
        to that span (used for per-pair tracking); otherwise the whole frame range is
        used. The config is named per-range so concurrent/sequential pairs don't clobber."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        first = self.indices[0] if first is None else int(first)
        last = self.indices[-1] if last is None else int(last)
        text = self.config_template.read_text()
        overrides = {
            "firstIndex": first,
            "lastIndex": last,
            "inDir": str(self.data_dir),
            "outDir": str(self.out_dir),
            "fileNamePrefix": self.prefix,
            "pointCount": self.point_count,
            "volumeGridResolution": self.vg_resolution,
        }
        for tag, val in overrides.items():
            text, n = re.subn(rf"<{tag}>.*?</{tag}>", f"<{tag}>{val}</{tag}>", text, flags=re.S)
            if n == 0:  # tag absent in template -> insert before close
                text = text.replace("</Config>", f"  <{tag}>{val}</{tag}>\n</Config>")
        cfg = self.work_dir / f"config_{first:03d}_{last:03d}.xml"
        cfg.write_text(text)
        return cfg

    def _editor_cmd(self, s, t):
        exe = self.tvm_editor_exe
        base = [self.dotnet, exe] if exe.endswith(".dll") else [exe]
        return base + [s, t, str(self.data_dir), str(self.out_dir), str(self.deformed_dir)]

    # ------------------------------------------------------------------ pipeline
    def track_pair(self, s, t):
        """Run the volume tracking over just the [s, t] span (two frames), right before
        re-binding that pair -- instead of tracking the whole range once up front. Cached
        per-pair in-memory so a repeated deform() of the same pair doesn't re-track."""
        s, t = int(s), int(t)
        first, last = min(s, t), max(s, t)
        key = (first, last)
        if key in self._tracked_pairs:
            return True
        try:
            if not self.client_dll.exists():
                raise FileNotFoundError(f"Client.dll not found at {self.client_dll}")
            self._stage_meshes()
            cfg = self._write_config(first=first, last=last)
            self._run([self.dotnet, str(self.client_dll), str(cfg)],
                      cwd=self.arap_dir, desc=f"ARAP Client (volume tracking {first}->{last})")
            self._tracked_pairs.add(key)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] TvmTracker.track_pair({s},{t}) failed ({exc})")
            return False

    def deform(self, s, t):
        """Return the path to deformed_{s}_{t}.obj (source frame s deformed to frame t),
        running get_transformation + TVMEditor if not already cached. None on failure."""
        out_obj = self.deformed_dir / "output" / f"deformed_{int(s):03d}_{int(t):03d}.obj"
        if out_obj.exists():
            return out_obj
        if not self.track_pair(s, t):
            return None
        try:
            self._run([sys.executable, str(self.get_transform_py),
                       "--centers_dir", str(self.out_dir),
                       "--sourceIndex", int(s), "--targetIndex", int(t)],
                      cwd=self.arap_dir, desc="get_transformation")
            self._run(self._editor_cmd(int(s), int(t)),
                      cwd=Path(self.tvm_editor_exe).resolve().parent, desc="TVMEditor deform")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] TvmTracker.deform({s},{t}) failed ({exc})")
            return None
        if out_obj.exists():
            return out_obj
        print(f"[WARN] TvmTracker.deform({s},{t}): expected output missing: {out_obj}")
        return None

    def deformed_src_for(self, s, t, expected_vertex_count=None, device="cuda"):
        """Full convenience path: produce + load the deformed source mesh in the reader
        frame, ready to pass as `deformed_src`. Returns None on any failure (caller falls
        back to an in-repo tracker)."""
        path = self.deform(s, t)
        if path is None:
            return None
        try:
            verts = load_mesh_in_reader_frame(path, device=device)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] TvmTracker: failed to load deformed mesh {path} ({exc})")
            return None
        '''
        if expected_vertex_count is not None and verts.shape[0] != expected_vertex_count:
            print(f"[WARN] TvmTracker: deformed vertex count {verts.shape[0]} != expected "
                  f"{expected_vertex_count}; falling back.")
            return None
        '''
        return verts
