import torch
import open3d as o3d
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor
)
from scene.cameras import convert_camera_from_gs_to_pytorch3d
import numpy as np
import torchvision.transforms.functional as TF

def pcd_renderer_pytorch3d(viewpoint_camera, 
                            pcd=None,
                            pcd_path=None,
                            image_height=800, image_width=800,
                            background_color=(1.0, 1.0, 1.0),
                            faces_per_pixel=1, device="cuda"):
    
    if pcd is None:
        pcd = o3d.io.read_point_cloud("/mnt/data1/syjintw/NEU/dataset/hotdog/points3d.ply")
    
    points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
    colors = torch.tensor(np.asarray(pcd.colors), dtype=torch.float32, device='cuda')
    point_cloud = Pointclouds(points=[points], features=[colors])
    
    p3d_cameras = convert_camera_from_gs_to_pytorch3d(
        [viewpoint_camera]
    )
    
    raster_settings = PointsRasterizationSettings(
        image_size=(image_height, image_width),
        radius=0.001,        # controls point size
        points_per_pixel=10 # controls density
    )
    
    rasterizer = PointsRasterizer(
        cameras=p3d_cameras[0], 
        raster_settings=raster_settings)
    
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )
    
    # ---------------------------------------------------------------------------- #
    #                                 Render Color                                 #
    # ---------------------------------------------------------------------------- # 
    with torch.no_grad():
        rgb_img = renderer(point_cloud)[0, ..., :3]
        
    bg_color = rgb_img.permute(2, 0, 1).contiguous()
    
    # # ------------------------- Save color for debugging ------------------------- #
    # bg_pil = TF.to_pil_image(bg_color.cpu())   # Convert tensor → PIL Image
    # bg_pil.save(f"./bg_color.png")
    
    return bg_color
    