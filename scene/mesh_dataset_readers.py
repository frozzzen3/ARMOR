#
# Copyright (C) 2024, Gmum
# Group of Machine Learning Research. https://gmum.net/
# All rights reserved.
#
# The Gaussian-splatting software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
# The Gaussian-mesh-splatting is software based on Gaussian-splatting, used on research.
# This Games software is free for non-commercial, research and evaluation use
#

import os
import numpy as np
import trimesh
import torch

from utils.graphics_utils import MeshPointCloud
from scene.dataset_readers import (
    readColmapSceneInfo,
    readNerfSyntheticInfo,
    readCamerasFromTransforms,
    infer_mesh_image_subdir,
    getNerfppNorm,
    SceneInfo,
    storePly,
)
from utils.sh_utils import SH2RGB
from scene.budgeting import get_budgeting_policy

from pathlib import Path

softmax = torch.nn.Softmax(dim=2)


def transform_vertices_function(vertices, c=1):
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] = -vertices[:, 1]
    vertices *= c
    return vertices


def mesh_has_texture_image(mesh_scene):
    visual = getattr(mesh_scene, "visual", None)
    material = getattr(visual, "material", None)
    image = getattr(material, "image", None)
    uv = getattr(visual, "uv", None)
    return uv is not None and image is not None


def get_triangle_average_colors(mesh_scene, faces):
    visual = getattr(mesh_scene, "visual", None)

    if mesh_has_texture_image(mesh_scene):
        print("[INFO] Mesh has UV coordinates and texture image.")
        print("type(mesh_scene.visual):", type(mesh_scene.visual))
        print("mesh_scene.visual.uv.shape:", mesh_scene.visual.uv.shape)
        print("mesh_scene.visual.material.image:", mesh_scene.visual.material.image)

        texture_img = np.array(mesh_scene.visual.material.image)
        H, W = texture_img.shape[:2]
        face_uvs = mesh_scene.visual.uv[faces]

        px = (face_uvs[..., 0] * (W - 1)).astype(int)
        py = ((1 - face_uvs[..., 1]) * (H - 1)).astype(int)
        px = np.clip(px, 0, W - 1)
        py = np.clip(py, 0, H - 1)

        tri_vertex_colors = texture_img[py, px, :3]
        tri_avg_colors = tri_vertex_colors.mean(axis=1)
        print("tri_avg_colors:", tri_avg_colors.shape)
        return tri_avg_colors

    print("[INFO] Mesh has no usable UV texture; using vertex colors if available.")
    vertex_colors = getattr(visual, "vertex_colors", None)
    if vertex_colors is not None and len(vertex_colors) > int(faces.max()):
        vertex_colors = np.asarray(vertex_colors[:, :3])
        print(f"[INFO] Loaded vertex colors: {vertex_colors.shape}")
        tri_vertex_colors = vertex_colors[faces]
        return tri_vertex_colors.mean(axis=1)

    print("[WARNING] Mesh has neither texture nor vertex colors; using white mesh colors.")
    return np.full((faces.shape[0], 3), 255.0, dtype=np.float32)


def get_num_splats_per_triangle(
    triangles, # [N,3,3]
    mesh_scene,
    train_cam_infos,
    path,
    num_splats,
    policy_path: str = None,
    total_splats: int = None,
    budgeting_policy_name: str = "uniform",
    min_splats_per_tri: int = 0, # [NOTE] could be adjusted
    max_splats_per_tri: int = 8,
    textured_mesh = None,
    mesh_type: str = "sugar"
)-> np.ndarray: # [N,], number of splats on each triangle
    
    # define allocation_path only when policy_path provided
    allocation_path = Path(policy_path) if policy_path else None

    # load num_splats allocation policy from pre-computed file
    if allocation_path is not None and allocation_path.exists():
        print(f"[INFO] Loading splat allocation from: {allocation_path}")
        num_splats_per_triangle = np.load(allocation_path)
        print("[INFO] loaded distribution, max and min:", num_splats_per_triangle.max(), num_splats_per_triangle.min())
    
    # [DONE] load weights here
            
    # Use budgeting policy, computing on-the-fly
    elif total_splats is not None:
        print(f"[INFO] no pre-computed policy found")
        print(f"[INFO] Recalculating splat allocation on the fly")
        print(f"[INFO] Scene::Reader() Using budgeting policy: {budgeting_policy_name}")

        budgeting_policy = get_budgeting_policy(
            budgeting_policy_name,
            mesh=mesh_scene,
            viewpoint_camera_infos=train_cam_infos, # access camera objects from cam_infos in the allocator somehow
            dataset_path=path,
            mesh_type=mesh_type,
            p3d_mesh=textured_mesh,
        )
        num_splats_per_triangle = budgeting_policy.allocate(
            total_splats=total_splats,
        )

        # num_splats_per_triangle = budgeting_policy.allocate_bounded(
        #     triangles=triangles,
        #     total_splats=total_splats,
        #     min_per_tri=min_splats_per_tri,
        #     max_per_tri=max_splats_per_tri,
        # )

        print(f"[INFO] Scene::Reader() Requested total splats: {total_splats}")
        print(f"[INFO] Scene::Reader() Allocated total splats: {num_splats_per_triangle.sum()}")
        print(f"[INFO] Scene::Reader() Min/Max splats per triangle: {num_splats_per_triangle.min()}/{num_splats_per_triangle.max()}")
        print(f"[INFO] Scene::Reader() Mean/Std splats per triangle: {num_splats_per_triangle.mean():.2f}/{num_splats_per_triangle.std():.2f}")

        # save under {dataset_path}/policy/mesh_{mesh_type}/tri_{num_tri}/{policy_name}/
        # budget: {total_splats}.npy
        # weights: weights.npy (same for any budget using the same policy & same mesh)
        num_tri = num_splats_per_triangle.shape[0]
        allocation_save_path = Path(path) / f"policy/mesh_{mesh_type}/tri_{num_tri}/{budgeting_policy_name}/{total_splats}.npy"
        allocation_save_path.parent.mkdir(parents=True, exist_ok=True)
        assert allocation_save_path.parent.exists(), "Directory does not exist, please create it first!"
        weights_save_path = allocation_save_path.parent / f"weights.npy"
        
        np.save(allocation_save_path, num_splats_per_triangle)
        print(f"[INFO] Scene::Reader() Saving allocation policy file to: {allocation_save_path}")

        np.save(weights_save_path, budgeting_policy.weights)
        print(f"[INFO] Scene::Reader() Saving weights file to: {weights_save_path}")
        
        # [DOING] [DONE] save the weights[] 



    # Fallback: uniform distribution using num_splats
    else:
        num_splats_per_triangle = np.full(triangles.shape[0], num_splats, dtype=int)
        print(f"[WARNING] Scene::Reader() Fallback using uniform distribution: {num_splats} splats per triangle")

    num_pts = num_splats_per_triangle.sum()
    print(f"[INFO] Generating random point cloud ({num_pts})...")
    print(f"\tnumber of mesh faces:  {triangles.shape[0]}...")
    print(f"\tAverage points per triangle: {num_pts / triangles.shape[0] if triangles.shape[0] > 0 else 0}...")
        


    return num_splats_per_triangle


def readNerfSyntheticMeshInfo( # don't use num_splats
        path, white_background, eval, num_splats, extension=".png",
        # >>>> [YC] add
        texture_obj_path: str = None,
        policy_path: str = None,
        # <<<< [YC] add
        # >>>> [SAM] add budgeting policy params
        total_splats: int = None,  
        budget_per_tri: float = None, 
        budgeting_policy_name: str = "uniform",
        min_splats_per_tri: int = 0,
        max_splats_per_tri: int = 8,
        mesh_type: str = "sugar",
        textured_mesh = None,
        # <<<< [SAM] add
        skip_sampling: bool = False,
) -> SceneInfo:
    image_subdir = infer_mesh_image_subdir(texture_obj_path)
    if image_subdir is not None:
        print(f"[INFO] Using per-mesh image subdirectory: {image_subdir}")

    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path,
        "transforms_train.json",
        white_background,
        extension,
        image_subdir=image_subdir,
    )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path,
        "transforms_test.json",
        white_background,
        extension,
        image_subdir=image_subdir,
    )
    
    
    
    # not priority for now: clean all the mesh loading logic into one place.
    # [DONE] a workaround to send loaded-texture-mesh to budgeting.py
    if texture_obj_path is None:
        print(f"[INFO] DatasetReader::Reading Mesh object from {path}/mesh.obj")
        mesh_scene = trimesh.load(f'{path}/mesh.obj', force='mesh', process=False)
    else:
        print(f"[INFO] Reading Mesh object from {texture_obj_path}")
        mesh_scene = trimesh.load(texture_obj_path, force='mesh', process=False)


    # >>>> [YC] add: because the mesh is generated from torch3d, so need to rotate
    mesh_scene.apply_transform(trimesh.transformations.rotation_matrix(
        angle=-np.pi/2, direction=[1, 0, 0], point=[0, 0, 0]
    ))
    # <<<< [YC] add
    
    vertices = mesh_scene.vertices
    vertices = transform_vertices_function(
        torch.tensor(vertices),
    )
    faces = mesh_scene.faces
    triangles = vertices[torch.tensor(mesh_scene.faces).long()].float()
    
    
    # >>>> [YC] add
    has_uv = (mesh_type == "sugar") and mesh_has_texture_image(mesh_scene)
    # [YC] [NOTE] set to false support mesh from colmap; otherwise true for sugar-generated mesh
    
    print(f"[DEBUG] mesh_type: {mesh_type}, has_uv: {has_uv}")
    
    if has_uv:
        print("[INFO] Mesh has UV coordinates and texture PNG.")
    # <<<< [YC] add
    
    # [NOTE] this is weird, why merge train and test cams, even if not in --eval mode?
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply") # What is this points3d.ply? COLMAP data, generated point cloud, or just placeholder?
    print("ply_path:", ply_path)
    
    # if not os.path.exists(ply_path):
    if skip_sampling:
        # Render path (Option B): the Gaussians are loaded from a checkpoint and re-bound to
        # this (decoded) mesh by normal-aware projection, so we do NOT allocate or sample a
        # scaffold point cloud here. (Doing so is wasted work and would break under a policy
        # built for a different frame's topology.) Build a trivial mesh-geometry point cloud
        # -- one filler point per face -- whose samples are never used as a Gaussian binding.
        tri_avg_colors = get_triangle_average_colors(mesh_scene, faces)
        num_pts = triangles.shape[0]
        xyz = triangles.mean(dim=1)
        alpha = torch.full((num_pts, 3), 1.0 / 3.0)
        tri_indices = torch.arange(num_pts, dtype=torch.long)
        pcd = MeshPointCloud(
            alpha=alpha,
            points=xyz,
            colors=tri_avg_colors / 255.0,
            normals=np.zeros((num_pts, 3)),
            vertices=vertices,
            faces=faces,
            transform_vertices_function=transform_vertices_function,
            triangles=triangles.cuda(),
            triangle_indices=tri_indices,
        )
        storePly(ply_path, pcd.points, tri_avg_colors)
        print(f"[INFO] skip_sampling: built mesh-geometry-only point cloud ({num_pts} faces).")

    elif True:

        assert budget_per_tri is not None or total_splats is not None, "Either num_splats or total_splats must be provided for budgeting!"

        if total_splats is None:
            total_splats = int(budget_per_tri * triangles.shape[0])
            print(f"[INFO] total_splats not provided, computed from budget_per_tri: {total_splats} splats")
        else:
            print(f"[INFO] total_splats provided: {total_splats} splats")
        
        # >>>> [SAM] Budgeting policy integration
        num_splats_per_triangle = get_num_splats_per_triangle(
            triangles=triangles,
            mesh_scene=mesh_scene,
            train_cam_infos=train_cam_infos,
            path=path,
            num_splats=num_splats,
            policy_path=policy_path,
            total_splats=total_splats,
            budgeting_policy_name=budgeting_policy_name,
            min_splats_per_tri=min_splats_per_tri,
            max_splats_per_tri=max_splats_per_tri,
            textured_mesh=textured_mesh,
            mesh_type=mesh_type,
        )
        # <<<< [SAM] Budgeting policy integration
        num_pts = num_splats_per_triangle.sum()
        
        # Since this data set has no colmap data, we start with random points sampled on the mesh surface
        
        # ---------------------------------------------------------------------------- #
        #                 Get initial Gaussian colors from texture map                 #
        # ---------------------------------------------------------------------------- #
        tri_avg_colors = get_triangle_average_colors(mesh_scene, faces)
            
        # We create random points inside the bounds triangles
        xyz_list = []
        alpha_list = []
        color_list = []
        tri_indices_list = []
        
        # [TODO] Build point-to-triangle mapping & triangle-to-point mapping
        for i in range(triangles.shape[0]):
            n = num_splats_per_triangle[i]
            if n == 0:
                continue
                
            alpha = torch.rand(n, 3)
            alpha = alpha / alpha.sum(dim=1, keepdim=True)  # normalize to barycentric coords

            pts = (alpha[:, 0:1] * triangles[i, 0] +
                alpha[:, 1:2] * triangles[i, 1] +
                alpha[:, 2:3] * triangles[i, 2])

            color = np.repeat(tri_avg_colors[i].reshape(1, 3), n, axis=0)  # (num_pts, 3)
            # print(color.shape) # [YC] debug
            
            xyz_list.append(pts)
            alpha_list.append(alpha)
            color_list.append(color)
            tri_indices_list.append(torch.full((n,), i, dtype=torch.long))

 
        # [DEBUG] Check if xyz_list is populated
        print(f"[DEBUG] xyz_list length: {len(xyz_list)}")
        if len(xyz_list) == 0:
            print("[ERROR] xyz_list is empty! No points were generated from triangles.")
            print(f"[DEBUG] triangles shape: {triangles.shape}")
            print(f"[DEBUG] num_pts_each_triangle: {num_splats_per_triangle}")
            raise RuntimeError("Failed to generate random points inside triangles")
        
        
        xyz = torch.cat(xyz_list, dim=0)
        xyz = xyz.reshape(num_pts, 3)
        
        alpha = torch.cat(alpha_list, dim=0)
        
        # shs = np.random.random((num_pts, 3)) / 255.0
        colors = np.concatenate(color_list, axis=0)
        print(colors.shape, xyz.shape, alpha.shape) # [YC] debug
        
        points = trimesh.points.PointCloud(np.array(xyz))

        # Combine into a scene
        scene = trimesh.Scene()
        scene.add_geometry(mesh_scene)
        scene.add_geometry(points)
        
        tri_indices = torch.cat(tri_indices_list, dim=0)
        
        
        pcd = MeshPointCloud(
            alpha=alpha,
            points=xyz,
            # colors=SH2RGB(shs),
            colors=colors/255.0,
            normals=np.zeros((num_pts, 3)),
            vertices=vertices,
            faces=faces,
            transform_vertices_function=transform_vertices_function,
            triangles=triangles.cuda(),
            triangle_indices=tri_indices
        )
        # [NOTE] should save both gs-to-tri mapping and tri-to-gs triangle_indices
        
        print("Created MeshPointCloud with", pcd.points.shape[0], "points.")

        # storePly(ply_path, pcd.points, SH2RGB(shs) * 255)
        storePly(ply_path, pcd.points, colors)
        print("Stored initial point cloud to", ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


# sceneLoadTypeCallbacks = {
#     "Colmap": readColmapSceneInfo,
#     "Blender": readNerfSyntheticInfo,
#     "Blender_Mesh": readNerfSyntheticMeshInfo
# }
