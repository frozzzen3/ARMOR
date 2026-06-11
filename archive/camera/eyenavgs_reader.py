import csv
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

import torch
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

# -------- Camera structure --------
class CameraInfo():
    uid: int
    R: np.array
    T: np.array
    FoVy: np.array
    FoVx: np.array
    # image: np.array
    # image_path: str
    # image_name: str
    image_width: int
    image_height: int
    zfar: float
    znear: float

    def __init__(self, uid, R, T, FoVx, FoVy, image_width, image_height, zfar=100.0, znear=0.01):
        self.uid = uid
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = image_width
        self.image_height = image_height
        self.zfar = zfar
        self.znear = znear
        
        self.trans = np.array([0.0, 0.0, 0.0])
        self.scale = 1.0
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

# --------- Custom CSV camera reader ---------
def readCustomCameras(csv_file, 
                    scene_quat=None, scene_scale=1.0,
                    image_width=1920, image_height=1080):
    """
    Reads a CSV camera file with fields:
    ViewIndex,FOV1,FOV2,FOV3,FOV4,PositionX,PositionY,PositionZ,
    QuaternionX,QuaternionY,QuaternionZ,QuaternionW,...
    and converts it to a list of CameraInfo objects.
    """
    cam_infos = []
    
    # Scene-level rotation
    if scene_quat is not None:
        R_scene = R.from_quat(scene_quat).as_matrix()
    else:
        R_scene = np.eye(3)
        
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["ViewIndex"] == "1":
                continue
            
            idx = int(row["ViewIndex"])
            
            # FoV: horizontal (FOV1,FOV2), vertical (FOV3,FOV4)
            FovX = abs(float(row["FOV1"])) + abs(float(row["FOV2"]))
            FovY = abs(float(row["FOV3"])) + abs(float(row["FOV4"]))

            # Position and rotation
            T = np.array([
                float(row["PositionX"]),
                float(row["PositionY"]),
                float(row["PositionZ"]),
            ])
            quat = np.array([
                float(row["QuaternionX"]),
                float(row["QuaternionY"]),
                float(row["QuaternionZ"]),
                float(row["QuaternionW"]),
            ])
            R_cam = R.from_quat(quat).as_matrix()
            
            # Apply global scene rotation and scale
            R_world = R_scene @ R_cam
            T_world = (R_scene @ (scene_scale * T))

            cam_info = CameraInfo(
                uid=idx,
                R=R_world,
                T=T_world,
                FoVx=FovX,
                FoVy=FovY,
                image_width=image_width,
                image_height=image_height,
            )
            cam_infos.append(cam_info)
    
    print(f"Loaded {len(cam_infos)} cameras from {csv_file}")
    return cam_infos
