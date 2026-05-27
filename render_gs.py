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
from renderer.gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from games import gaussianModelRender

from PIL import Image
import torchvision.transforms as T


def render_set(gs_type, model_path, name, iteration, views, gaussians, pipeline, 
               background, background_depth):
    
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"renders_{gs_type}")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, 
                           bg_color=background, bg_depth=background_depth)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(gs_type: str, dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = gaussianModelRender[gs_type](dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if hasattr(gaussians, 'update_alpha'):
            gaussians.update_alpha()
        if hasattr(gaussians, 'prepare_vertices'):
            gaussians.prepare_vertices()
        if hasattr(gaussians, 'prepare_scaling_rot'):
            gaussians.prepare_scaling_rot()

        # bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        # background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # # Load static image background
        # background_image_path = "/home/syjintw/Desktop/NEU/gaussian-mesh-splatting/output_render_no_light.png"
        # img = Image.open(background_image_path).convert("RGB")
        # viewpoint_camera_height = 800
        # viewpoint_camera_width = 800
        # # === 1. Use PIL to load images ===
        # img = img.resize((viewpoint_camera_height, viewpoint_camera_width), Image.BILINEAR)

        # # === 2. Change image to Tensor ===
        # transform = T.Compose([
        #     T.ToTensor(),  # [0, 255] → [0.0, 1.0], shape (3, H, W)
        # ])
        # background = transform(img).to(torch.float32).cuda()
        
        # # Random noise as background
        # viewpoint_camera_height = 800
        # viewpoint_camera_width = 800
        # background = torch.rand((3, viewpoint_camera_height, viewpoint_camera_width), dtype=torch.float32, device="cuda")
        # print(background)

        # Random noise as background
        viewpoint_camera_height = 800
        viewpoint_camera_width = 800
        background = torch.ones((3, viewpoint_camera_height, viewpoint_camera_width), dtype=torch.float32, device="cuda")
        # print(background)
        
        # ------------------------------ Pure background ----------------------------- #
        pure_bg_template = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        pure_bg = torch.tensor(pure_bg_template, dtype=torch.float32, device="cuda").view(3, 1, 1)
        pure_bg = pure_bg.expand(3, viewpoint_camera_height, viewpoint_camera_width)
        
        # --------------------- Pure depth background (all zeros) -------------------- #
        pure_bg_depth = torch.full((1, viewpoint_camera_height, viewpoint_camera_width), 0, dtype=torch.float32, device="cuda")
        
        if not skip_train:
             render_set(gs_type, dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, 
                        pure_bg, pure_bg_depth)

        if not skip_test:
             render_set(gs_type, dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, 
                        pure_bg, pure_bg_depth)

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

    args = get_combined_args(parser)
    model.gs_type = args.gs_type
    model.num_splats = args.num_splats
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(args.gs_type, model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    