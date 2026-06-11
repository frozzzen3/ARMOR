import torch
from pytorch3d.renderer import (
    AmbientLights,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    )
from pytorch3d.renderer.blending import BlendParams
from scene.cameras import convert_camera_from_gs_to_pytorch3d

def mesh_renderer_pytorch3d(viewpoint_camera, textured_mesh,
                            image_height=800, image_width=800,
                            background_color=(1.0, 1.0, 1.0),
                            faces_per_pixel=1, device="cuda"):
    
    p3d_cameras = convert_camera_from_gs_to_pytorch3d(
        [viewpoint_camera]
    )
    
    mesh_raster_settings = RasterizationSettings(
        image_size=(image_height, image_width),
        blur_radius=0.0, 
        faces_per_pixel=faces_per_pixel,
    )
    lights = AmbientLights(device=device)
    rasterizer = MeshRasterizer(
        cameras=p3d_cameras[0], 
        raster_settings=mesh_raster_settings,
    )
    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=SoftPhongShader(
            device=device, 
            cameras=p3d_cameras[0],
            lights=lights,
            blend_params=BlendParams(background_color=background_color),
        )
    )
    
    # ---------------------------------------------------------------------------- #
    #                                 Render Color                                 #
    # ---------------------------------------------------------------------------- # 
    rgb_img = renderer(textured_mesh, cameras=p3d_cameras)[0, ..., :3]
    
    bg_color = rgb_img.permute(2, 0, 1).contiguous()
    
    # ------------------------- Save color for debugging ------------------------- #
    # bg_pil = TF.to_pil_image(bg_color.cpu())   # Convert tensor → PIL Image
    # bg_pil.save("./bg_color.png")
    
    # ---------------------------------------------------------------------------- #
    #                                 Render Depth                                 #
    # ---------------------------------------------------------------------------- #
    # Get fragments from rasterizer
    fragments = rasterizer(textured_mesh, cameras=p3d_cameras)
    
    # Nearest surface depth in NDC space
    depth = fragments.zbuf[0, ..., 0]  # (H, W)
    
    # Mask out pixels that didn’t hit any face
    mask = fragments.pix_to_face[0, ..., 0] >= 0
    depth = depth.masked_fill(~mask, -1)
    
    bg_depth = depth.unsqueeze(0)
    
    # ------------------------ Save depth pt for debugging ----------------------- #
    # torch.save(depth, "./bg_depth.pt")
    
    return bg_color, bg_depth, fragments
    