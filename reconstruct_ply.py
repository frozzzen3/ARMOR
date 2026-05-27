from pathlib import Path
from games.mesh_splatting.scene.gaussian_mesh_model import GaussianMeshModel
from torch import nn
import torch
import trimesh
import numpy as np

from games.mesh_splatting.scene.dataset_readers import get_num_splats_per_triangle

from utils.general_utils import inverse_sigmoid, rot_to_quat_batch

# import pytorch3d.structures import Meshes
# from pytorch3d.renderer import TexturesVertex

C0 = 0.28209479177387814

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def transform_vertices_function(vertices, c=1):
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] = -vertices[:, 1]
    vertices *= c
    return vertices

def prepare_scaling_rot(triangles, triangle_indices, 
                        _scale,
                        _scaling, _rotation,
                        eps_s0):
    """
    approximate covariance matrix and calculate scaling/rotation tensors

    covariance matrix is [v0, v1, v2], where
    v0 is a normal vector to each face
    v1 is a vector from centroid of each face and 1st vertex
    v2 is obtained by orthogonal projection of a vector from
    centroid to 2nd vertex onto subspace spanned by v0 and v1.
    """
    def dot(v, u):
        return (v * u).sum(dim=-1, keepdim=True)
    
    def proj(v, u):
        """
        projection of vector v onto subspace spanned by u

        vector u is assumed to be already normalized
        """
        coef = dot(v, u)
        return coef * u

    triangles = triangles
    normals = torch.linalg.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        dim=1
    )
    v0 = normals / (torch.linalg.vector_norm(normals, dim=-1, keepdim=True) + eps_s0)
    means = torch.mean(triangles, dim=1)
    v1 = triangles[:, 1] - means
    v1_norm = torch.linalg.vector_norm(v1, dim=-1, keepdim=True) + eps_s0
    v1 = v1 / v1_norm
    v2_init = triangles[:, 2] - means
    v2 = v2_init - proj(v2_init, v0) - proj(v2_init, v1)  # Gram-Schmidt
    v2 = v2 / (torch.linalg.vector_norm(v2, dim=-1, keepdim=True) + eps_s0)
    s1 = v1_norm / 2.
    s2 = dot(v2_init, v2) / 2.
    s0 = eps_s0 * torch.ones_like(s1)
    scales = torch.concat((s0, s1, s2), dim=1).unsqueeze(dim=1)
    
    # scales = scales.broadcast_to((*self.alpha.shape[:2], 3))
    # self._scaling = torch.log(
    #     torch.nn.functional.relu(self._scale * scales.flatten(start_dim=0, end_dim=1)) + self.eps_s0
    # )
    # rotation = torch.stack((v0, v1, v2), dim=1).unsqueeze(dim=1)
    # rotation = rotation.broadcast_to((*self.alpha.shape[:2], 3, 3)).flatten(start_dim=0, end_dim=1)
    # rotation = rotation.transpose(-2, -1)
    # self._rotation = rot_to_quat_batch(rotation)
    
    scales = scales[triangle_indices]
    with torch.no_grad():
        s_input = _scale * scales.flatten(start_dim=0, end_dim=1)
        s_input = torch.nn.functional.relu(s_input) + eps_s0
        new_scaling = torch.log(s_input)

    _scaling = new_scaling.detach()
    rotation = torch.stack((v0, v1, v2), dim=1).unsqueeze(dim=1)
    # rotation = rotation.broadcast_to((*self.alpha.shape[:2], 3, 3)).flatten(start_dim=0, end_dim=1)
    rotation = rotation[triangle_indices]
    rotation = rotation.transpose(-2, -1)
    _rotation = rot_to_quat_batch(rotation)
        
if __name__ == "__main__":
    gs_path = "/mnt/data1/syjintw/NEU/mesh-splat/output/1110/Debug_bicycle_colmap_area_/point_cloud/iteration_10/point_cloud.ply"
    mesh_path = "/mnt/data1/syjintw/NEU/dataset/colmap/bicycle/checkpoint/mesh.ply"
    policy_path = "/mnt/data1/syjintw/NEU/mesh-splat/output/1110/Debug_bicycle_colmap_area_/area_1572865.npy"
    save_path = "/mnt/data1/syjintw/NEU/mesh-splat/output/1110/Debug_bicycle_colmap_area_/reconstructed_point_cloud/point_cloud.ply"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    gs = GaussianMeshModel(sh_degree=3)
    gs.load_ply(gs_path)
    
    print("Number of gaussians loaded from gs_path:", gs.get_xyz.shape[0])
    
    # # ---------------------------------------------------------------------------- #
    # #                             Generate the dummy gs                            #
    # # ---------------------------------------------------------------------------- #
    # # print(gs._xyz)
    # xyz_dummy = nn.Parameter(torch.zeros((gs._xyz.shape[0], gs._xyz.shape[1]), dtype=torch.float32, device="cuda"))
    # scale_dummy = nn.Parameter(torch.zeros((gs._scaling.shape[0], gs._scaling.shape[1]), dtype=torch.float32, device="cuda"))
    # rot_dummy = nn.Parameter(torch.zeros((gs._rotation.shape[0], gs._rotation.shape[1]), dtype=torch.float32, device="cuda"))
    
    # gs._xyz = xyz_dummy
    # gs._scaling = scale_dummy
    # gs._rotation = rot_dummy
    # # print(gs._xyz)
    # # print(gs._scaling)
    # # print(gs._rotation)
    
    # ---------------------------------------------------------------------------- #
    #                                  Load policy                                 #
    # ---------------------------------------------------------------------------- #
    num_splats_per_triangle = np.load(policy_path)
    num_pts = num_splats_per_triangle.sum()
    print("Number of triangle from policy:", num_splats_per_triangle.shape[0])
    print("Number of points from policy:", num_pts)
    
    # ---------------------------------------------------------------------------- #
    #                                   Load mesh                                  #
    # ---------------------------------------------------------------------------- #
    mesh_scene = trimesh.load(mesh_path, force='mesh')
    mesh_scene.apply_transform(trimesh.transformations.rotation_matrix(
        angle=-np.pi/2, direction=[1, 0, 0], point=[0, 0, 0]
    ))
    vertices = mesh_scene.vertices
    vertices = transform_vertices_function(
        torch.tensor(vertices),
    )
    faces = mesh_scene.faces
    triangles = vertices[torch.tensor(mesh_scene.faces).long()].float()
    vertex_colors = mesh_scene.visual.vertex_colors[:, :3]
    tri_vertex_colors = vertex_colors[faces]  # (n_faces, 3, 3)
    tri_avg_colors = tri_vertex_colors.mean(axis=1)
    print("Number of triangles in mesh:", triangles.shape[0])
    
    # ---------------------------------------------------------------------------- #
    #                             Generate point cloud                             #
    # ---------------------------------------------------------------------------- #
    xyz_list = []
    alpha_list = []
    color_list = []
    tri_indices_list = []
    
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
    
    xyz = torch.cat(xyz_list, dim=0)
    xyz = xyz.reshape(num_pts, 3)
    
    alpha = torch.cat(alpha_list, dim=0)
    
    colors = np.concatenate(color_list, axis=0)
    
    points = trimesh.points.PointCloud(np.array(xyz))
    
    print("Number of points in reconstructed point cloud: ", points.vertices.shape[0])
    
    # ---------------------------------------------------------------------------- #
    #                                  Generate gs                                 #
    # ---------------------------------------------------------------------------- #
    
    # fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
    # features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
    # features[:, :3, 0] = fused_color
    # features[:, 3:, 1:] = 0.0
    
    max_sh_degree = 3
    fused_color = RGB2SH(torch.tensor(np.asarray(colors)).float().cuda())
    features = torch.zeros((fused_color.shape[0], 3, (max_sh_degree + 1) ** 2)).float().cuda()
    features[:, :3, 0] = fused_color
    features[:, 3:, 1:] = 0.0
    
    opacities = inverse_sigmoid(0.1 * torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda"))

    self.vertices = nn.Parameter(
        self.point_cloud.vertices.clone().detach().requires_grad_(True).cuda().float()
    )
    self.faces = torch.tensor(self.point_cloud.faces).cuda()

    self._alpha = nn.Parameter(alpha_point_cloud.requires_grad_(True))
    self.update_alpha()
    self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
    self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
    self._scale = nn.Parameter(scale.requires_grad_(True))
    self.prepare_scaling_rot()
    self._opacity = nn.Parameter(opacities.requires_grad_(True))
    self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    # mesh_tm = trimesh.load(mesh_path, force='mesh', process=False)
    # verts = torch.tensor(mesh_tm.vertices, dtype=torch.float32)
    # faces = torch.tensor(mesh_tm.faces, dtype=torch.int64)
    # colors = torch.tensor(mesh_tm.visual.vertex_colors[:, :3], dtype=torch.float32) / 255.0
    # textured_mesh = Meshes(
    #         verts=[verts],
    #         faces=[faces],
    #         textures=TexturesVertex(verts_features=[colors])
    #         ).to("cuda")
    
    # mesh_tm.apply_transform(trimesh.transformations.rotation_matrix(
    #     angle=-np.pi/2, direction=[1, 0, 0], point=[0, 0, 0]
    # ))
    # vertices = mesh_tm.vertices
    # vertices = transform_vertices_function(
    #     torch.tensor(vertices),
    # )
    # faces = mesh_tm.faces
    # triangles = vertices[torch.tensor(mesh_tm.faces).long()].float()
    
    # min_splats_per_tri = 0
    # max_splats_per_tri = 8
    # budget_per_tri = 1.0
    # num_splats = 2
    
    # total_splats = int(budget_per_tri * triangles.shape[0])
    
    # num_splats_per_triangle = get_num_splats_per_triangle(
    #     triangles=triangles,
    #     mesh_scene=mesh_tm,
    #     train_cam_infos=None,
    #     path=None,
    #     num_splats=num_splats,
    #     policy_path=None,
    #     total_splats=total_splats,
    #     budgeting_policy_name="uniform",
    #     min_splats_per_tri=min_splats_per_tri,
    #     max_splats_per_tri=max_splats_per_tri,
    #     textured_mesh=textured_mesh   
    # )
    
    