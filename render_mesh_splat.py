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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from pathlib import Path
from PIL import Image
import torchvision
import torchvision.transforms as T
# from renderer.gaussian_renderer import render
from renderer.mesh_splat_renderer import render # [YC] change to mesh_splat_renderer
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from games import gaussianModelRender

from pytorch3d.io import load_objs_as_meshes
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from train import load_textured_mesh, load_textured_mesh_for_nvdiffrast

import json
import time

def create_scene_card(dataset: ModelParams, scene, gs_type: str, occlusion: bool, 
                     mesh_type: str, iteration: int, render_time: float = None) -> dict:
    """
    Create a scene card dictionary with rendering metadata.
    
    Args:
        dataset: Model parameters
        scene: Scene object with loaded gaussians
        gs_type: Gaussian splatting type ('gs', 'gs_mesh', etc.)
        occlusion: Whether occlusion handling is enabled
        mesh_type: Type of mesh used ('sugar', 'colmap', etc.)
        iteration: Training iteration number
        render_time: Total rendering time in seconds (optional)
    
    Returns:
        Dictionary containing scene metadata
    """
    scene_card = {}
    
    # Target budget (as requested by args)
    target_budget = getattr(dataset, "total_splats", None)
    if target_budget is None and hasattr(dataset, "budget_per_tri"):
        # Try to estimate using triangle count if available
        num_tri = None
        if hasattr(scene.gaussians, "triangles"):
            try:
                num_tri = int(getattr(scene.gaussians.triangles, "shape", [None])[0])
            except Exception:
                num_tri = None
        if num_tri:
            target_budget = int(dataset.budget_per_tri * num_tri)
    scene_card["target_budget"] = target_budget
    
    # Used budget: try several common attribute names
    used_budget = None
    for attr in ("num_splats", "N", "num_gaussians", "num_points"):
        if hasattr(scene.gaussians, attr):
            used_budget = int(getattr(scene.gaussians, attr))
            break
    # Fallback to point_cloud sizes
    if used_budget is None:
        try:
            pc = getattr(scene.gaussians, "point_cloud", None)
            if pc is not None:
                if hasattr(pc, "positions"):
                    used_budget = int(len(pc.positions))
                elif hasattr(pc, "points"):
                    used_budget = int(len(pc.points))
                elif hasattr(pc, "vertices"):
                    used_budget = int(len(pc.vertices))
        except Exception:
            used_budget = None
    scene_card["used_budget"] = used_budget
    
    # Renderer / training type
    if gs_type == "gs":
        renderer_type = "pure_GS"
    elif gs_type == "gs_mesh":
        renderer_type = "DTGS" if occlusion else "TGS"
    else:
        renderer_type = gs_type
    scene_card["renderer_type"] = renderer_type
    
    # Training type: what the saved model was trained with
    training_type = getattr(dataset, "gs_type", gs_type)
    scene_card["training_type"] = training_type
    
    # Mesh type
    scene_card["mesh_type"] = mesh_type
    
    # Trained iterations / epoch
    scene_card["trained_iteration"] = scene.loaded_iter if hasattr(scene, "loaded_iter") else iteration
    
    # Budgeting policy and related
    scene_card["alloc_policy"] = getattr(dataset, "alloc_policy", None)
    scene_card["budget_per_tri"] = getattr(dataset, "budget_per_tri", None)
    
    # Number of triangles (if available)
    num_triangles = None
    try:
        if hasattr(scene.gaussians, "triangles"):
            num_triangles = int(getattr(scene.gaussians.triangles, "shape", [None])[0])
    except Exception:
        num_triangles = None
    scene_card["num_triangles"] = num_triangles
    
    # Timing information
    scene_card["render_time_seconds"] = render_time
    scene_card["training_time_seconds"] = None  # Not available in render stage
    
    # Metadata
    scene_card["timestamp"] = time.time()
    scene_card["notes"] = "training_time_seconds not available during rendering"
    
    return scene_card

def save_scene_card(scene_card: dict, output_path: str):
    """Save scene card dictionary to JSON file."""
    try:
        with open(output_path, "w") as fh:
            json.dump(scene_card, fh, indent=2)
        print(f"[INFO] Render:: wrote scene card to {output_path}")
    except Exception as e:
        print(f"[WARNING] Could not write scene card to {output_path}: {e}")

def render_set(gs_type, model_path, name, iteration, views, gaussians, pipeline, background,
                # >>>> [YC] add
                texture_obj_path : str = None,
                occlusion: bool = False,
                policy_path : str = None,
                mesh_type : str = "colmap",
                textured_mesh = None,
                precaptured_mesh_img_path : str = None,
                mesh_rasterizer_type : str = "pytorch3d"
                # <<<< [YC] add
                ):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"renders_{gs_type}")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    debug_path = os.path.join(model_path, name, "ours_{}".format(iteration), "debug") # [YC] add for debug images

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(debug_path, exist_ok=True) # [YC] add for debug images
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # >>>> [YC] add for debug images
        if idx % 10 == 0:
            debug_flag = True
        else:
            debug_flag = False
        # <<<< [YC] add for debug images
        
        # Load precaptured mesh background and depth if available
        bg = None
        bg_depth = None
        
        if precaptured_mesh_img_path:
            cached_bg_path = Path(precaptured_mesh_img_path) / mesh_rasterizer_type / "test_mesh_texture" / f"{view.image_name}.png"
            cached_bg_depth_path = Path(precaptured_mesh_img_path) / mesh_rasterizer_type / "test_mesh_depth" / f"{view.image_name}.pt"
            
            if cached_bg_path.exists():
                img = Image.open(cached_bg_path).convert("RGB")
                img = img.resize((view.image_width, view.image_height), Image.BILINEAR)
                transform = T.Compose([T.ToTensor()])
                bg = transform(img).to(torch.float32).cuda() # [0, 255] → [0.0, 1.0], shape (3, H, W)
                # >>>> [YC] add for debug images
                if debug_flag:
                    torchvision.utils.save_image(bg, os.path.join(debug_path, '{0:05d}_bg'.format(idx) + ".png"))
                # <<<< [YC] add for debug images
            if cached_bg_depth_path.exists():
                bg_depth = torch.load(cached_bg_depth_path).unsqueeze(0).to("cuda")
        
        # [DONE] add pure GS renderer back here
        if gs_type == "gs":
            # Pure GS rendering without textured mesh
            pure_bg_template = background
            pure_bg = torch.tensor(pure_bg_template, dtype=torch.float32, device="cuda").view(3, 1, 1)
            pure_bg = pure_bg.expand(3, view.image_height, view.image_width)
            pure_bg_depth = torch.full((1, view.image_height, view.image_width), 0, dtype=torch.float32, device="cuda")
            
            rendering = render(view, gaussians, pipeline, 
                            bg_color=pure_bg, bg_depth=pure_bg_depth)["render"]
            print("\033[94m [INFO] Render::GS using pure GS rasterizer\033[0m")
            
        elif gs_type == "gs_mesh":
            # [NOTE] ensure that during rendering we use the same rasterizer as in training
            if occlusion:
                rendering = render(view, gaussians, pipeline, 
                                bg_color=bg, bg_depth=bg_depth,
                                textured_mesh=textured_mesh,
                                mesh_background_color=background,
                                mesh_rasterizer_type=mesh_rasterizer_type)["render"] # [YC] using different rasterizer
                print("\033[92m [INFO] Render::DTGS using Depth+Texture+GS rasterizer for gs_mesh\033[0m")
                
            else: 
                pure_bg_depth = torch.full((1, view.image_height, view.image_width), 0, dtype=torch.float32, device="cuda")
                rendering = render(view, gaussians, pipeline, 
                                bg_color=bg, bg_depth=pure_bg_depth,
                                textured_mesh=textured_mesh,
                                mesh_background_color=background,
                                mesh_rasterizer_type=mesh_rasterizer_type)["render"] # [YC] no occlusion handling, always use pure bg and pure depth
                print("\033[96m [INFO] Render::TGS using Texture+GS rasterizer for gs_mesh\033[0m")
            
            # >>>> [YC] add for debug images
            if debug_flag:
                # save pure gaussian
                _pure_bg_template = background
                _pure_bg = torch.tensor(_pure_bg_template, dtype=torch.float32, device="cuda").view(3, 1, 1)
                _pure_bg = _pure_bg.expand(3, view.image_height, view.image_width)
                _pure_bg_depth = torch.full((1, view.image_height, view.image_width), 0, dtype=torch.float32, device="cuda")
                _rendering = render(view, gaussians, pipeline, 
                                bg_color=_pure_bg, bg_depth=_pure_bg_depth)["render"]
                torchvision.utils.save_image(_rendering, os.path.join(debug_path, '{0:05d}_pure_gs'.format(idx) + ".png"))
            # <<<< [YC] add for debug images
            
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
        # >>>> [YC] add for debug images
        if debug_flag:
            torchvision.utils.save_image(rendering, os.path.join(debug_path, '{0:05d}_rendering'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(debug_path, '{0:05d}_gt'.format(idx) + ".png"))
        # <<<< [YC] add for debug images
        
# sets are {train,test, (val)}
def render_sets(gs_type: str, dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                # >>>> [YC] add
                texture_obj_path : str = None,
                occlusion: bool = False,
                policy_path : str = None,
                precaptured_mesh_img_path : str = None,
                mesh_rasterizer_type: str = "pytorch3d"
                # <<<< [YC] add
                ):
    render_timer_start = time.time()
    
    with torch.no_grad():
        gaussians = gaussianModelRender[gs_type](dataset.sh_degree)
        if mesh_rasterizer_type == "pytorch3d":
            textured_mesh = load_textured_mesh(dataset, texture_obj_path)
        elif mesh_rasterizer_type == "nvdiffrast":
            textured_mesh = load_textured_mesh_for_nvdiffrast(dataset, texture_obj_path)
            
        # [BUG] trace from here to see how ply and policy are loaded
        scene = Scene(dataset, gaussians, 
                      load_iteration=iteration, shuffle=False,
                      policy_path=policy_path,
                      texture_obj_path=texture_obj_path,
                      textured_mesh=textured_mesh
                      )
        if hasattr(gaussians, 'update_alpha'):
            gaussians.update_alpha()
        if hasattr(gaussians, 'prepare_vertices'):
            gaussians.prepare_vertices()
        if hasattr(gaussians, 'prepare_scaling_rot'):
            gaussians.prepare_scaling_rot()

        mesh_type = dataset.mesh_type if hasattr(dataset, 'mesh_type') else "sugar"
        print(f"[INFO] Render:: Using mesh type: {mesh_type}")
        
        # if mesh_type == "colmap":
        #     bg_color = [0,0,0] 
        #     print(f"[WARNING] Render:: overriding background color to black for colmap mesh type!")
        # else:
        #     print(f"[INFO] Render:: bg:{bg_color} for mesh type: {mesh_type}")
        
        
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Add new params here
        render_kwargs = {
            'mesh_type': mesh_type,
            'texture_obj_path': texture_obj_path,
            'occlusion': occlusion,
            'policy_path': policy_path,
            'textured_mesh': scene.textured_mesh,
            'precaptured_mesh_img_path': precaptured_mesh_img_path,
            'mesh_rasterizer_type': mesh_rasterizer_type,
        }

        if not skip_train:
            render_set(gs_type, dataset.model_path, "train", scene.loaded_iter, 
                  scene.getTrainCameras(), gaussians, pipeline, background,
                  **render_kwargs)

        if not skip_test:
            render_set(gs_type, dataset.model_path, "test", scene.loaded_iter, 
                  scene.getTestCameras(), gaussians, pipeline, background,
                  **render_kwargs)

    # Create and save scene card
    render_time = time.time() - render_timer_start
    scene_card = create_scene_card(dataset, scene, gs_type, occlusion, 
                                   mesh_type, iteration, render_time)
    scene_card_path = os.path.join(dataset.model_path, "scene_card.json")
    save_scene_card(scene_card, scene_card_path)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--gs_type', type=str, default="gs_flat")
    parser.add_argument("--num_splats", nargs="+", type=int, default=[2])
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    # >>>> [YC] add
    parser.add_argument("--texture_obj_path", type=str, default=None, help="Path to the textured obj file for mesh-based datasets.")
    parser.add_argument("--occlusion", action="store_true", help="Whether to use occlusion handling during rendering.")
    parser.add_argument("--policy_path", type=str, default="", help="Path to the splat density policy npy file.")
    parser.add_argument("--precaptured_mesh_img_path", type=str, default="",
        help="path to the directory containing precaptured mesh (RGB & D) images for background. \
            should contain mesh_texture/ and mesh_depth/ sub-folders.")
    # <<<< [YC] add
    
    # >>>> [Sam] add
    parser.add_argument("--total_splats", type=int, help="Total number of splats to allocate")
    parser.add_argument("--alloc_policy", type=str, default="area", help="Allocation policy for splats (default: area)")
    parser.add_argument("--budget_per_tri", type=float, default=1.0, help="set the total number of splats to be this number * number of triangles")
    # parser.add_argument("--drop_budget", type=int, help="drop until only this number of splats remain in the scene.")
    parser.add_argument('--mesh_type', type=str, default="sugar", help="textured mesh type: sugar, colmap, or others")
    # <<<< [Sam] add
    
    parser.add_argument("--mesh_rasterizer_type", type=str, default="pytorch3d", 
                        help="which mesh rasterizer to use: pytorch3d or nvdiffrast") 
    
    
    args = get_combined_args(parser) # get args from both command line and stored file
    model.gs_type = args.gs_type
    model.num_splats = args.num_splats
    
    # >>>> [SAM] add
    model.total_splats = args.total_splats
    model.alloc_policy = args.alloc_policy
    model.budget_per_tri = args.budget_per_tri
    model.mesh_type = args.mesh_type
    # model.drop_budget = args.drop_budget
    # <<<< [SAM] add
    
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(args.gs_type, model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
                # >>>> [YC] add
                texture_obj_path=args.texture_obj_path,
                occlusion=args.occlusion,
                policy_path=args.policy_path,
                precaptured_mesh_img_path=args.precaptured_mesh_img_path,
                mesh_rasterizer_type=args.mesh_rasterizer_type
                # <<<< [YC] add
                )