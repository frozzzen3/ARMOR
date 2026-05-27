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

import os
import random
import json
import typing

from utils.system_utils import searchForMaxIteration
from games.scenes import sceneLoadTypeCallbacks
from games.mesh_splatting.scene.gaussian_mesh_model import GaussianMeshModel
 
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from pytorch3d.structures import Meshes


class Scene:

    gaussians : GaussianModel
    textured_mesh : Meshes
    # [major refactor, not now] decouple the initialization/loading ordering problem, as they shouldn't be dependent on each other
    # [DONE] use a workaround for now
    def __init__(self, 
                args : ModelParams, 
                gaussians : GaussianModel, 
                load_iteration=None, shuffle=True, resolution_scales=[1.0],
                # >>>> [YC] add
                texture_obj_path : str = None, # legacy - use textured_mesh parameter instead
                policy_path : str = None,
                textured_mesh = None,
                initialize_gaussians: bool = True
                # <<<< [YC] add
                ):
        """b
        :param path: Path to colmap scene main folder.
        """
        print("[INFO] Scene::init() policy_path:", policy_path) # [YC] debug
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.textured_mesh = textured_mesh  
        self.point_cloud = None

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print(f"Loading trained model at iteration {self.loaded_iter}")

        self.train_cameras = {}
        self.test_cameras = {}

        # ---------------------------------------------------------------------------- #
        #               Call dataset reader according to dataset type                  # 
        # ---------------------------------------------------------------------------- #
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            if args.gs_type == "gs_multi_mesh":
                scene_info = sceneLoadTypeCallbacks["Colmap_Mesh"](
                    args.source_path, args.images, args.eval, args.num_splats, args.meshes
                )
            # [YC] add gs_mesh type single colmap mesh
            # Real world scene (indoor/outdoor) uses this loader
            elif args.gs_type == "gs_mesh":
                scene_info = sceneLoadTypeCallbacks["Colmap_Single_Mesh"](
                    args.source_path, args.images, args.eval, args.num_splats[0], 
                    texture_obj_path=texture_obj_path,
                    policy_path=policy_path,
                    total_splats=args.total_splats,
                    budget_per_tri=args.budget_per_tri,
                    budgeting_policy_name=args.alloc_policy,
                    mesh_type=args.mesh_type,
                    textured_mesh = textured_mesh
                    
                )
            else:
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            if args.gs_type == "gs_mesh": #! [YC] need to be aware of gs_type
                
                print("Found transforms_train.json file, assuming Blender_Mesh dataset!")
                
                
                # Synthetic scene uses this loader
                scene_info = sceneLoadTypeCallbacks["Blender_Mesh"](
                    args.source_path, args.white_background, args.eval, args.num_splats[0],
                    # >>>> [YC] add
                    texture_obj_path=texture_obj_path,
                    policy_path=policy_path,
                    # <<<< [YC] add
                    # >>>> [Sam] add
                    total_splats=args.total_splats,
                    budget_per_tri=args.budget_per_tri,
                    budgeting_policy_name=args.alloc_policy,
                    mesh_type=args.mesh_type,
                    textured_mesh = textured_mesh
                    # <<<< [Sam] add
                )
            elif args.gs_type == "gs_flame":
                print("Found transforms_train.json file, assuming Flame Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender_FLAME"](args.source_path, args.white_background, args.eval)
            else:
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"
            
        
        self.point_cloud = scene_info.point_cloud

        # [DONE] fix the weird copying issue, budget_per_tri and total_splats behavior 
        # save a copy of allocation result into output dir
        
        num_tri = scene_info.point_cloud.triangles.shape[0] if hasattr(scene_info.point_cloud, 'triangles') else 0
        
        
        assert (args.budget_per_tri is not None) or (args.total_splats is not None), "Either num_splats or total_splats must be provided for budgeting!"
        
        if args.total_splats is None: 
            total_splats = int(args.budget_per_tri * num_tri)
        else:
            total_splats = args.total_splats
        
        print(f"[INFO] Scene:: total_splats for budgeting policy copy: {total_splats}")
        computed_policy_path = os.path.join(
            args.source_path, 
            f"policy/mesh_{args.mesh_type}/tri_{num_tri}/{args.alloc_policy}/{total_splats}.npy"
        )
        copy_dest = os.path.join(self.model_path, f"{args.alloc_policy}_{total_splats}.npy")
        print(f"[INFO] Copying computed budgeting policy from {computed_policy_path} to {copy_dest}")
        if os.path.exists(computed_policy_path):
            with open(computed_policy_path, 'rb') as src_file, open(copy_dest , 'wb') as dest_file:
                dest_file.write(src_file.read())
        else:    
            print(f"[WARNING] Didn't find computed budgeting policy file at {computed_policy_path}, skipping copy.")
        
        
        
        # ====== Load Cameras and PLY files ======
        if not self.loaded_iter:
            if args.gs_type == "gs_multi_mesh":
                for i, ply_path in enumerate(scene_info.ply_path):
                    with open(ply_path, 'rb') as src_file, open(os.path.join(self.model_path, f"input_{i}.ply") , 'wb') as dest_file:
                        dest_file.write(src_file.read())
            else:
                # print(f"[DEBUG] Scene:: Copying from ply file {scene_info.ply_path} to {os.path.join(self.model_path, f'input.ply')}")
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            json_cams = []
            json_train_cams = []
            json_test_cams = []
            camlist = []
            train_camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
                train_camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            for id, cam in enumerate(train_camlist):
                json_train_cams.append(camera_to_JSON(id, cam))
            for id, cam in enumerate(scene_info.test_cameras):
                json_test_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)
            with open(os.path.join(self.model_path, "train_cameras.json"), 'w') as file:
                json.dump(json_train_cams, file)
            with open(os.path.join(self.model_path, "test_cameras.json"), 'w') as file:
                json.dump(json_test_cams, file)
            
                
        # if shuffle:
        #     print("shuffle") # [YC] debug
        #     random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
        #     random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Scene:: Loading Training Cameras from camInfos at scale ", resolution_scale)
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            # print(self.train_cameras[resolution_scale][0].uid) # [YC] debug
            print("Scene:: Loading Test Cameras from camInfos at scale ", resolution_scale)
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        # [YC] [NOTE] Load trained GS scene (ply file) for rendering
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            print(f"[INFO] Scene:: loaded gs model from iteration {self.loaded_iter}")
            self.gaussians.point_cloud = scene_info.point_cloud
            if args.gs_type == "gs_mesh": #! [YC] need to be aware of gs_type
                self.gaussians.triangles = scene_info.point_cloud.triangles
                # >>>> [YC] add
                self.gaussians.triangle_indices = scene_info.point_cloud.triangle_indices.cuda() # [YC] add
                # <<<< [YC] add
        elif initialize_gaussians: # [YC] note: first time training
            # [YC] note: if using "gs_mesh", the create_from_pcd() 
            # will use the one defined in mesh-splat/games/scene/gaussian_model_mesh.py
            # under class GaussianMeshModel(GaussianModel)
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)




    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
