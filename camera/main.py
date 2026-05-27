import os
import sys
import math
import json
import numpy as np
import torch
from torch import nn
from typing import NamedTuple
import open3d as o3d

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    MeshRasterizer,
    RasterizationSettings,
    FoVPerspectiveCameras,
)

from pytorch3d.renderer import FoVPerspectiveCameras as P3DCameras
from pytorch3d.renderer.cameras import _get_sfm_calibration_matrix

from colmap_reader import (
    read_extrinsics_text, read_intrinsics_text, qvec2rotmat,
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
)

import py3d_camera

# ---------------------------------------------------------------------------- #
#           Compute cube center distance in camera space (PyTorch3D)           #
# ---------------------------------------------------------------------------- #
def get_cube_center_and_depth_p3d(mesh_path, p3d_camera, device="cuda"):
    """Return cube center (world) and z-distance (depth) in camera space."""
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    bbox = mesh.get_axis_aligned_bounding_box()
    center = np.asarray(bbox.get_center(), dtype=np.float32)
    center_tensor = torch.tensor(center[None, :], dtype=torch.float32, device=device)

    # Transform center to camera space
    cam_space = p3d_camera.get_world_to_view_transform().transform_points(center_tensor)
    z = cam_space[0, 2].item()  # positive z means in front
    return center, z*z
  
def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

# ---------------------------------------------------------------------------- #
#                               Camera structure                               #
# ---------------------------------------------------------------------------- #
class CameraInfo():
    uid: int
    R: np.array
    T: np.array
    FoVy: np.array
    FoVx: np.array
    # image: np.array
    # image_path: str
    # image_name: str
    image_width: int
    image_height: int
    zfar: float
    znear: float

    def __init__(self, uid, R, T, FoVx, FoVy, image_width, image_height, zfar=100.0, znear=0.01):
        self.uid = uid
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = image_width
        self.image_height = image_height
        self.zfar = zfar
        self.znear = znear

def readColmapCameras(cam_extrinsics, cam_intrinsics):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        # print(extr.name)
        # image_path = os.path.join(images_folder, os.path.basename(extr.name))
        # image_name = os.path.basename(image_path).split(".")[0]
        # image = Image.open(image_path)
        # image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FoVy=FovY, FoVx=FovX,
                              image_width=width, image_height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

# ---------------------------------------------------------------------------- #
#                                   Extrinsic                                  #
# ---------------------------------------------------------------------------- #
def get_extrinsic_matrix(cam: CameraInfo):
    Rt = np.eye(4)
    Rt[:3, :3] = cam.R.T      # world→camera rotation
    Rt[:3, 3]  = -cam.R.T @ cam.T
    return Rt

# ---------------------------------------------------------------------------- #
#                                   Intrinsic                                  #
# ---------------------------------------------------------------------------- #
def get_intrinsic_matrix(cam: CameraInfo):
    fx = (cam.width / 2) / np.tan(np.deg2rad(cam.FovX / 2))
    fy = (cam.height / 2) / np.tan(np.deg2rad(cam.FovY / 2))
    cx, cy = cam.width / 2, cam.height / 2

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])
    return K

# -------- Visibility + Save --------
def visible_points_from_camera(points, cam: CameraInfo, save_path=None):
    K = get_intrinsic_matrix(cam)
    extrinsic = get_extrinsic_matrix(cam)
    width, height = cam.width, cam.height

    pts_h = np.hstack((points, np.ones((len(points), 1))))
    pts_cam = (extrinsic @ pts_h.T).T[:, :3]
    in_front = pts_cam[:, 2] > 0
    pts_cam = pts_cam[in_front]

    # Project
    proj = (K @ pts_cam.T).T
    proj[:, 0] /= proj[:, 2]
    proj[:, 1] /= proj[:, 2]

    u, v, z = proj[:, 0], proj[:, 1], pts_cam[:, 2]
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[inside], v[inside], z[inside]
    subset_idx = np.where(in_front)[0][inside]

    depth_buffer = np.full((height, width), np.inf)
    visible_mask = np.zeros(len(points), dtype=bool)

    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    for x, y, depth, idx in zip(ui, vi, z, subset_idx):
        if depth < depth_buffer[y, x] - 1e-3:
            depth_buffer[y, x] = depth
            visible_mask[idx] = True
    
    # ---- Save visible points ----
    if save_path:
        vis_points = points[visible_mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vis_points)

        o3d.io.write_point_cloud(save_path, pcd)
        print(f"✅ Saved {vis_points.shape[0]} visible points → {save_path}")
        
    return visible_mask

def is_cube_in_view(center, radius, cam: CameraInfo, buffer_ratio=0.1):
    """
    Check whether a cube (center + radius) is inside the camera frustum.
    """
    # --- Compute camera-space coordinates ---
    extrinsic = get_extrinsic_matrix(cam)
    center_h = np.hstack([center, 1.0])
    center_cam = extrinsic @ center_h
    x, y, z = center_cam[:3]

    if z <= 0:  # behind camera
        return False

    # --- Compute FoV half-angles with buffer ---
    fov_x = np.deg2rad(cam.FoVx * (1 + buffer_ratio))
    fov_y = np.deg2rad(cam.FoVy * (1 + buffer_ratio))

    # --- Project normalized coordinates ---
    tan_x = abs(x / z)
    tan_y = abs(y / z)

    return (tan_x <= np.tan(fov_x / 2)) and (tan_y <= np.tan(fov_y / 2))

def get_cube_center_and_radius(mesh_path):
    """Load a cube mesh and compute its center + bounding sphere radius."""
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    bbox = mesh.get_axis_aligned_bounding_box()
    center = np.asarray(bbox.get_center())
    radius = np.linalg.norm(bbox.get_extent()) / 2.0
    return center, radius

def visible_mesh_cubes(cube_dir, cam: CameraInfo, buffer_ratio=0.1):
    """
    Iterate through all cube meshes in a directory and return which are visible.
    """
    visible_list = []
    mesh_files = sorted([f for f in os.listdir(cube_dir) if f.endswith(".ply") or f.endswith(".obj")])
    print(f"Checking {len(mesh_files)} cubes...")

    for f in mesh_files:
        mesh_path = os.path.join(cube_dir, f)
        center, radius = get_cube_center_and_radius(mesh_path)
        if is_cube_in_view(center, radius, cam, buffer_ratio=buffer_ratio):
            visible_list.append(f)

    print(f"✅ Found {len(visible_list)} visible cubes.")
    return visible_list

# ---------------------------------------------------------------------------- #
#                         Load cube mesh into PyTorch3D                        #
# ---------------------------------------------------------------------------- #
def load_mesh_as_pytorch3d(mesh_path, device="cuda"):
    mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
    verts = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)

    if len(verts) == 0 or len(faces) == 0:
        return None

    verts = torch.tensor(verts, dtype=torch.float32, device=device)
    faces = torch.tensor(faces, dtype=torch.int64, device=device)
    return Meshes(verts=[verts], faces=[faces])

# ---------------------------------------------------------------------------- #
#                  Visibility Check with PyTorch3D Rasterizer                  #
# ---------------------------------------------------------------------------- #
def visible_mesh_cubes_pytorch3d(
    cube_dir,
    camera,
    image_size=(1080, 1920),
    device="cuda",
):
    """
    Check which cubes (meshes) are visible in the camera frustum.
    Returns:
        visible_list: list of visible cube filenames
    """
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings)

    mesh_files = sorted(
        [f for f in os.listdir(cube_dir) if f.endswith(".ply") or f.endswith(".obj")]
    )
    visible_with_depth = []
    print(f"Checking {len(mesh_files)} cube meshes...")

    for f in mesh_files:
        mesh_path = os.path.join(cube_dir, f)
        mesh = load_mesh_as_pytorch3d(mesh_path, device=device)
        if mesh is None:
            continue

        fragments = rasterizer(mesh)
        pix_to_face = fragments.pix_to_face[0, ..., 0]

        # any pixel covered by this mesh → visible
        # Visible if any face projects into viewport
        if torch.any(pix_to_face >= 0).item():
            _, z_square = get_cube_center_and_depth_p3d(mesh_path, camera, device)
            visible_with_depth.append((f, z_square))

    visible_sorted = sorted(visible_with_depth, key=lambda x: x[1])
    print(f"Found {len(visible_sorted)} visible cubes.")
    return visible_sorted

if __name__ == "__main__":
    path = "/mnt/data1/syjintw/NEU/dataset/bicycle"
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics)
    # cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.uid)
    cam_infos = cam_infos_unsorted.copy()

    # 2. Get PyTorch3D camera (e.g. from your convert_camera_from_gs_to_pytorch3d)
    p3d_cameras, image_widths, image_heights = py3d_camera.convert_camera_from_gs_to_pytorch3d(cam_infos, device="cuda") 
    