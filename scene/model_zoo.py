#
# Copyright (C) 2024, Gmum
# Group of Machine Learning Research. https://gmum.net/
# All rights reserved.
#
# The Gaussian-splatting software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#

"""Dispatch registries for the gs_type model variants and scene loaders.

Replaces the former games/__init__.py and games/scenes/__init__.py after the
mesh/multi-mesh/FLAME/flat parameterizations were folded into scene/.
"""

from arguments import OptimizationParams
from arguments_games import (
    OptimizationParamsMesh,
    OptimizationParamsFlame,
)

from scene.gaussian_model import GaussianModel
from scene.gaussian_mesh_model import GaussianMeshModel
from scene.gaussian_multi_mesh_model import GaussianMultiMeshModel
from scene.gaussian_flame_model import GaussianFlameModel
from scene.points_gaussian_model import PointsGaussianModel
from scene.flat_gaussian_model import FlatGaussianModel

from scene.dataset_readers import (
    readColmapSceneInfo,
    readNerfSyntheticInfo,
)
from scene.mesh_dataset_readers import readNerfSyntheticMeshInfo
from scene.multi_mesh_dataset_readers import (
    readColmapMeshSceneInfo,
    readColmapSingleMeshSceneInfo,
)
from scene.flame_dataset_readers import readNerfSyntheticFlameInfo

optimizationParamTypeCallbacks = {
    "gs": OptimizationParams,
    "gs_multi_mesh": OptimizationParamsMesh,
    "gs_flat": OptimizationParams,
    "gs_mesh": OptimizationParamsMesh,
    "gs_flame": OptimizationParamsFlame
}

gaussianModel = {
    "gs": GaussianModel,
    "gs_flat": FlatGaussianModel,
    "gs_mesh": GaussianMeshModel,
    "gs_multi_mesh": GaussianMultiMeshModel,
    "gs_flame": GaussianFlameModel,
    "gs_points": PointsGaussianModel
}

gaussianModelRender = {
    "gs": GaussianModel,
    "gs_flat": FlatGaussianModel,
    "gs_mesh": GaussianMeshModel,
    "gs_multi_mesh": GaussianMultiMeshModel,
    "gs_flame": GaussianFlameModel,
    "gs_points": PointsGaussianModel
}

# Called when Scene inits
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Colmap_Mesh": readColmapMeshSceneInfo,
    "Colmap_Single_Mesh": readColmapSingleMeshSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Blender_Mesh": readNerfSyntheticMeshInfo,
    "Blender_FLAME": readNerfSyntheticFlameInfo
}
