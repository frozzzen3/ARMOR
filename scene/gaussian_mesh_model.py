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

import torch
import numpy as np

from torch import nn

from scene.gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid, rot_to_quat_batch
from utils.sh_utils import RGB2SH
from utils.graphics_utils import MeshPointCloud


class GaussianMeshModel(GaussianModel):

    def __init__(self, sh_degree: int):

        super().__init__(sh_degree)
        self.point_cloud = None
        self._alpha = torch.empty(0)
        self._uvw = torch.empty(0)
        self._scale = torch.empty(0)
        self.alpha = torch.empty(0)
        self.uvw = torch.empty(0)
        self.softmax = torch.nn.Softmax(dim=2)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.update_alpha_func = self.softmax

        self.vertices = None
        self.faces = None
        self.triangles = None
        self.eps_s0 = 1e-8
        self.hover_init_scale = 1e-2
        
        # >>>> [YC] add
        self.triangle_indices = None
        # <<<< [YC] add
        self.clear_temporal_attributes(update_geometry=False)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_uvw(self):
        return self.uvw

    @property
    def get_scaling(self):
        scaling = self._scaling
        if self.temporal_d_scaling is not None:
            scaling = scaling + self.temporal_d_scaling
        return self.scaling_activation(scaling)

    @property
    def get_features(self):
        features_dc = self._features_dc
        if self.temporal_d_features_dc is not None:
            features_dc = features_dc + self.temporal_d_features_dc
        return torch.cat((features_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self):
        opacity = self._opacity
        if self.temporal_d_opacity is not None:
            opacity = opacity + self.temporal_d_opacity
        return self.opacity_activation(opacity)

    @property
    def get_mesh_id(self):
        return self.triangle_indices

    def create_from_pcd(self, pcd: MeshPointCloud, spatial_lr_scale: float):

        self.point_cloud = pcd
        self.triangles = self.point_cloud.triangles
        self.spatial_lr_scale = spatial_lr_scale
        
        alpha_point_cloud, mesh_ids = self._initial_logical_coordinates_and_mesh_ids(pcd)
        self.triangle_indices = mesh_ids.cuda() # [YC] add

        if pcd.points.shape[0] != alpha_point_cloud.shape[0]:
            raise ValueError(
                f"Point count ({pcd.points.shape[0]}) does not match logical coordinate count "
                f"({alpha_point_cloud.shape[0]})."
            )

        print("Number of mesh facets: ", int(torch.as_tensor(self.point_cloud.faces).shape[0]))
        print("Number of Mesh-adsorbed Gaussians: ", alpha_point_cloud.shape[0])

        alpha_point_cloud = alpha_point_cloud.float().cuda()
        uvw_point_cloud = self._surface_alpha_to_raw_uvw(alpha_point_cloud)
        scale = torch.ones((pcd.points.shape[0], 1)).float().cuda()

        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        opacities = inverse_sigmoid(0.1 * torch.ones((pcd.points.shape[0], 1), dtype=torch.float, device="cuda"))
        # opacities = inverse_sigmoid(0.01 * torch.ones((pcd.points.shape[0], 1), dtype=torch.float, device="cuda"))

        self.vertices = nn.Parameter(
            torch.as_tensor(self.point_cloud.vertices).clone().detach().requires_grad_(True).cuda().float()
        )
        self.faces = torch.as_tensor(self.point_cloud.faces, dtype=torch.long, device="cuda")

        self._uvw = nn.Parameter(uvw_point_cloud.requires_grad_(True))
        self._alpha = self._uvw
        self.update_alpha()
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scale = nn.Parameter(scale.requires_grad_(True))
        self.prepare_scaling_rot()
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def _initial_logical_coordinates_and_mesh_ids(self, pcd: MeshPointCloud):
        alpha = torch.as_tensor(pcd.alpha, dtype=torch.float32)

        if pcd.triangle_indices is not None:
            mesh_ids = torch.as_tensor(pcd.triangle_indices, dtype=torch.long)
            if alpha.ndim == 3:
                alpha = alpha.reshape(-1, alpha.shape[-1])
        elif alpha.ndim == 3:
            num_faces, num_gaussians_per_face = alpha.shape[:2]
            mesh_ids = torch.arange(num_faces, dtype=torch.long).repeat_interleave(num_gaussians_per_face)
            alpha = alpha.reshape(-1, alpha.shape[-1])
        else:
            raise ValueError("MeshPointCloud must provide triangle_indices when alpha is flattened.")

        if alpha.shape[0] != mesh_ids.shape[0]:
            raise ValueError(
                f"Mismatch between logical coordinates ({alpha.shape[0]}) and MeshId entries ({mesh_ids.shape[0]})."
            )

        return alpha, mesh_ids

    def _surface_alpha_to_raw_uvw(self, alpha):
        """
        Convert existing surface barycentric coordinates to raw MaGS-style logical
        coordinates. The first two channels are decoded as w1 and w2; the third
        raw channel is a small random hover height.
        """
        eps = 1e-4
        alpha = torch.relu(alpha) + eps
        alpha = alpha / alpha.sum(dim=-1, keepdim=True)

        w1 = alpha[:, 0:1].clamp(eps, 1.0 - eps)
        w2_cond = (alpha[:, 1:2] / (1.0 - w1).clamp_min(eps)).clamp(eps, 1.0 - eps)
        hover = (torch.rand_like(w1) * self.hover_init_scale).clamp(eps, 1.0 - eps)

        return torch.cat(
            (
                inverse_sigmoid(w1),
                inverse_sigmoid(w2_cond),
                inverse_sigmoid(hover),
            ),
            dim=1,
        )

    def rebind_to_mesh(self, vertices, faces, triangle_indices=None):
        """
        Reuse the current Gaussian parameters on a new mesh with matching topology.

        This keeps the learned barycentric coordinates, SH features, opacity and scale,
        and only updates the underlying mesh geometry used to decode Gaussian centers.
        """
        if not isinstance(vertices, torch.Tensor):
            vertices = torch.tensor(vertices)
        if not isinstance(faces, torch.Tensor):
            faces = torch.tensor(faces)

        vertices = vertices.detach().to(device="cuda", dtype=torch.float32)
        faces = faces.detach().to(device="cuda", dtype=torch.long)

        with torch.no_grad():
            if self.vertices is None:
                self.vertices = nn.Parameter(vertices.clone().requires_grad_(True))
            else:
                if self.vertices.shape != vertices.shape:
                    raise ValueError(
                        f"Vertex shape mismatch during temporal rebinding: "
                        f"{self.vertices.shape} vs {vertices.shape}"
                    )
                self.vertices.copy_(vertices)

            self.faces = faces
            if triangle_indices is not None:
                triangle_indices = triangle_indices.detach().to(device="cuda", dtype=torch.long)
                if self.triangle_indices is not None and self.triangle_indices.shape != triangle_indices.shape:
                    raise ValueError(
                        f"Triangle assignment shape mismatch during temporal rebinding: "
                        f"{self.triangle_indices.shape} vs {triangle_indices.shape}"
                    )
                self.triangle_indices = triangle_indices

        self.triangles = self.vertices[self.faces]
        self.update_alpha()
        self.prepare_scaling_rot()

    def _decode_uvw(self, include_temporal=True):
        raw = self._uvw
        if include_temporal and self.temporal_d_uvw is not None:
            raw = raw + self.temporal_d_uvw
        w1 = torch.sigmoid(raw[:, 0:1])
        w2 = torch.sigmoid(raw[:, 1:2]) * (1.0 - w1)
        w3 = torch.sigmoid(raw[:, 2:3])
        return torch.cat((w1, w2, w3), dim=1)

    def _face_normals_and_hover_scales(self, triangles):
        edge_12 = triangles[:, 1] - triangles[:, 0]
        edge_13 = triangles[:, 2] - triangles[:, 0]
        normals = torch.linalg.cross(edge_12, edge_13, dim=1)
        normal_norm = torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
        unit_normals = normals / (normal_norm + self.eps_s0)

        edge_12_norm = torch.linalg.vector_norm(edge_12, dim=-1, keepdim=True)
        edge_13_norm = torch.linalg.vector_norm(edge_13, dim=-1, keepdim=True)
        hover_scales = edge_12_norm * edge_13_norm / (normal_norm + self.eps_s0)
        return unit_normals, hover_scales

    def _calc_xyz(self):
        """
        calculate the 3d Gaussian center in the coordinates xyz.

        The logical coordinates w=(w1,w2,w3) are decoded as:
            f(w, x, y, z) = w1*x + w2*y + (1 - w1 - w2)*z
            mu = f(w, v1, v2, v3) + w3*n*l

        MeshId is stored in self.triangle_indices.

        """
        triangle_idx = self.triangles[self.triangle_indices]
        surface_xyz = torch.bmm(self.alpha.unsqueeze(1), triangle_idx).squeeze(1)

        normals, hover_scales = self._face_normals_and_hover_scales(self.triangles)
        hover = self.uvw[:, 2:3] * normals[self.triangle_indices] * hover_scales[self.triangle_indices]
        self._xyz = surface_xyz + hover
        
    def prepare_scaling_rot(self):
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

        triangles = self.vertices[self.faces]
        self.triangles = triangles
        normals = torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=1
        )
        v0 = normals / (torch.linalg.vector_norm(normals, dim=-1, keepdim=True) + self.eps_s0)
        means = torch.mean(triangles, dim=1)
        v1 = triangles[:, 1] - means
        v1_norm = torch.linalg.vector_norm(v1, dim=-1, keepdim=True) + self.eps_s0
        v1 = v1 / v1_norm
        v2_init = triangles[:, 2] - means
        v2 = v2_init - proj(v2_init, v0) - proj(v2_init, v1)  # Gram-Schmidt
        v2 = v2 / (torch.linalg.vector_norm(v2, dim=-1, keepdim=True) + self.eps_s0)

        s1 = v1_norm / 2.
        s2 = dot(v2_init, v2) / 2.
        s0 = self.eps_s0 * torch.ones_like(s1)
        scales = torch.concat((s0, s1, s2), dim=1).unsqueeze(dim=1)

        scales = scales[self.triangle_indices]
        s_input = self._scale * scales.flatten(start_dim=0, end_dim=1)
        s_input = torch.nn.functional.relu(s_input) + self.eps_s0
        self._scaling = torch.log(s_input)
        rotation = torch.stack((v0, v1, v2), dim=1).unsqueeze(dim=1)
        rotation = rotation[self.triangle_indices]
        rotation = rotation.transpose(-2, -1)
        self._rotation = rot_to_quat_batch(rotation)

    def update_alpha(self):
        """
        Function to control the alpha value.

        Keep the old method name for train.py compatibility, but decode the
        learnable Mesh-adsorbed Gaussian logical coordinates w.
        """
        self.uvw = self._decode_uvw()
        self.alpha = torch.cat(
            (
                self.uvw[:, 0:1],
                self.uvw[:, 1:2],
                1.0 - self.uvw[:, 0:1] - self.uvw[:, 1:2],
            ),
            dim=1,
        )
        self.triangles = self.vertices[self.faces]
        self._calc_xyz()

    def apply_temporal_attributes(self, temporal_model, frame_time):
        if temporal_model is None:
            self.clear_temporal_attributes()
            return
        base_uvw = self._decode_uvw(include_temporal=False)
        deltas = temporal_model(self.triangle_indices, base_uvw, frame_time)
        self.temporal_d_uvw = deltas.get("d_uvw")
        self.temporal_d_scaling = deltas.get("d_scaling")
        self.temporal_d_opacity = deltas.get("d_opacity")
        self.temporal_d_features_dc = deltas.get("d_features_dc")
        self.update_alpha()

    def clear_temporal_attributes(self, update_geometry=True):
        self.temporal_d_uvw = None
        self.temporal_d_scaling = None
        self.temporal_d_opacity = None
        self.temporal_d_features_dc = None
        if update_geometry and self.vertices is not None and self.faces is not None and self.triangle_indices is not None:
            self.update_alpha()

    def training_setup(self, training_args):
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l_params = [
            {'params': [self.vertices], 'lr': training_args.vertices_lr, "name": "vertices"},
            {'params': [self._uvw], 'lr': training_args.alpha_lr, "name": "uvw"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scale], 'lr': training_args.scaling_lr, "name": "scaling"}
        ]

        self.optimizer = torch.optim.Adam(l_params, lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration) -> None:
        """ Learning rate scheduling per step """
        pass

    def save_ply(self, path):
        self.clear_temporal_attributes()
        self.update_alpha()
        self.prepare_scaling_rot()
        self._save_ply(path)

        attrs = self.__dict__
        additional_attrs = [
            '_alpha', 
            '_uvw',
            '_scale',
            'point_cloud',
            'triangles',
            'vertices',
            'faces',
            'triangle_indices'
        ]

        save_dict = {}
        for attr_name in additional_attrs:
            save_dict[attr_name] = attrs[attr_name]

        path_model = path.replace('point_cloud.ply', 'model_params.pt')
        torch.save(save_dict, path_model)

    def load_ply(self, path):
        self._load_ply(path)
        path_model = path.replace('point_cloud.ply', 'model_params.pt')
        params = torch.load(path_model)
        uvw = params.get('_uvw', params.get('_alpha'))
        scale = params['_scale']
        if 'vertices' in params:
            vertices = params['vertices']
            if not isinstance(vertices, nn.Parameter):
                vertices = nn.Parameter(vertices)
            self.vertices = nn.Parameter(vertices.detach().to(device="cuda", dtype=torch.float32).requires_grad_(True))
        if 'triangles' in params:
            self.triangles = params['triangles'].to(device="cuda", dtype=torch.float32)
        if 'faces' in params:
            self.faces = params['faces'].to(device="cuda", dtype=torch.long)
        if 'triangle_indices' in params:
            self.triangle_indices = params['triangle_indices'].to(device="cuda", dtype=torch.long)
        # point_cloud = params['point_cloud']
        self._uvw = nn.Parameter(uvw.to(device="cuda", dtype=torch.float32).requires_grad_(True))
        self._alpha = self._uvw
        self._scale = nn.Parameter(scale.to(device="cuda", dtype=torch.float32).requires_grad_(True))
        if self.vertices is not None and self.faces is not None and self.triangle_indices is not None:
            self.update_alpha()
            self.prepare_scaling_rot()
