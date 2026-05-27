import torch
import math
import nvdiffrast.torch as dr
import numpy as np

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def calculate_mvp_matrix(FoVx, FoVy,
                         image_width, image_height, 
                         R, T,
                         res=1.0,
                         far=100.0, near=0.1, device="cuda"):
    # Build camera→world transform
    transf_mat = np.eye(4, dtype=np.float32)
    transf_mat[:3, :3] = np.transpose(R)
    transf_mat[:3, -1] = T
    
    # Convert to world→camera
    transf_mat = np.linalg.inv(transf_mat) # world→camera
    
    # Apply your coordinate-system flip (keep as-is)
    transf_mat[:3, 1:3] *= -1
    
    # This IS the correct world→camera matrix
    transf_matrix = torch.tensor(transf_mat, device=device, dtype=torch.float32)
    view_matrix = torch.inverse(transf_matrix)  # world → camera

    fx = fov2focal(FoVx, image_width)
    fy = fov2focal(FoVy, image_height)
    focal_x = fx / res
    focal_y = fy / res
    
    # proj_matrix = torch.zeros(4, 4, device="cpu", dtype=torch.float32)
    proj_matrix = torch.zeros(4, 4, device=device, dtype=torch.float32)
    proj_matrix[0, 0] = 2 * focal_x / image_width
    proj_matrix[1, 1] = -2 * focal_y / image_height
    proj_matrix[2, 2] = -(far + near) / (far - near)
    proj_matrix[2, 3] = -2.0 * far * near / (far - near)
    proj_matrix[3, 2] = -1.0

    mvp_matrix = (proj_matrix @ view_matrix).to(device)
    
    return mvp_matrix

def mesh_renderer_nvdiffrast(viewpoint_camera, textured_mesh,
                            image_height=800, image_width=800,
                            background_color=(1.0, 1.0, 1.0),
                            device="cuda"):
    ctx = dr.RasterizeCudaContext()
    
    verts = torch.tensor(textured_mesh.vertices, dtype=torch.float32, device=device).contiguous()
    faces = torch.tensor(textured_mesh.faces, dtype=torch.int32, device=device).contiguous()
    
    # Vertex colors
    if textured_mesh.visual.kind == 'vertex' and hasattr(textured_mesh.visual, 'vertex_colors'):
        colors = torch.tensor(textured_mesh.visual.vertex_colors[:, :3], dtype=torch.float32, device=device) / 255.0
        # print("[DEBUG] Using vertex colors from mesh.")
    else:
        colors = torch.ones_like(verts) * 0.7
        # print("[DEBUG] No vertex color found; using gray.")
    
    # mvp = viewpoint_camera["mvp_matrix"].to("cuda")
    mvp = calculate_mvp_matrix(viewpoint_camera.FoVx, viewpoint_camera.FoVy,
                        image_width, image_height, 
                        viewpoint_camera.R, viewpoint_camera.T,
                        far=viewpoint_camera.zfar, near=viewpoint_camera.znear, device=device)
    
    # --------------------------------- Rendering -------------------------------- #
    clip = torch.cat([verts, torch.ones_like(verts[:, :1])], dim=1)
    clip = (clip @ mvp.T).unsqueeze(0)  # (1, V, 4)
    clip = clip.contiguous()
    
    rast_out, _ = dr.rasterize(ctx, clip, faces, resolution=[image_height, image_width])
    bg_color, _ = dr.interpolate(colors.unsqueeze(0), rast_out, faces)
    bg_color = torch.clamp(bg_color[0], 0, 1)
    
    # -------------------------- Set background to white ------------------------- #
    bg_mask = rast_out[0, ..., 3] <= 0        
    bg_color = torch.where(rast_out[0, ..., 3:] > 0, bg_color, torch.tensor(background_color).to(device))
    
    # ------------------ Change the tensor shape from HWC to CHW ----------------- #
    bg_color = bg_color.permute(2, 0, 1).contiguous() # torch.Size([2224, 2160, 3]) --> torch.Size([3, 2224, 2160])
    
    # ------------------------------ Depth extraction ------------------------------ #
    pos_clip_interp, _ = dr.interpolate(clip, rast_out, faces)  # [1,H,W,4]
    pos_clip_interp = pos_clip_interp[0]
    # z_clip = pos_clip_interp[..., 2]
    w_clip = pos_clip_interp[..., 3]
    bg_depth = w_clip.masked_fill(bg_mask, -1.0)
    bg_depth = bg_depth.unsqueeze(0)  # [1,H,W]
    
    # return bg_color, bg_depth, fragments
    return bg_color, bg_depth, None
    