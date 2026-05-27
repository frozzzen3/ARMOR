import argparse
import json
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import torch
import torchvision.transforms.functional as TF
import trimesh
from PIL import Image
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    AmbientLights,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.renderer.cameras import FoVPerspectiveCameras, _get_sfm_calibration_matrix
from pytorch3d.structures import Meshes
from tqdm import tqdm

from renderer.mesh_renderer.mesh_renderer_nvdiffrast import mesh_renderer_nvdiffrast


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def resolve_mesh_sequence(mesh_path, mesh_start, mesh_end):
    if mesh_start is None and mesh_end is None:
        return [Path(mesh_path)]

    if mesh_start is None or mesh_end is None:
        raise ValueError("Both --mesh_start and --mesh_end must be provided together")

    mesh_path = Path(mesh_path)
    match = re.search(r"(\d+)$", mesh_path.stem)
    if match is None:
        raise ValueError("Mesh filename must end with digits when using --mesh_start/--mesh_end")

    width = len(match.group(1))
    prefix = mesh_path.stem[:match.start(1)]
    suffix = mesh_path.suffix
    return [
        mesh_path.with_name(f"{prefix}{index:0{width}d}{suffix}")
        for index in range(mesh_start, mesh_end + 1)
    ]


def build_mesh_subdir(mesh_path, mesh_folder_prefix):
    match = re.search(r"(\d+)$", Path(mesh_path).stem)
    if match is None:
        return Path(mesh_folder_prefix) / Path(mesh_path).stem
    return Path(f"{mesh_folder_prefix}_{match.group(1)}")


def build_output_path(out_root, rel_path, mesh_subdir=None):
    rel_path = Path(rel_path)
    if mesh_subdir is None:
        return Path(out_root) / f"{rel_path}.png"

    rel_without_suffix = rel_path.with_suffix(".png")
    parts = rel_without_suffix.parts
    if parts and parts[0] in {"train", "test"}:
        return Path(out_root) / parts[0] / mesh_subdir / Path(*parts[1:])

    return Path(out_root) / mesh_subdir / rel_without_suffix


def load_textured_mesh(mesh_type, mesh_path):
    if mesh_type == "sugar":
        if not mesh_path.lower().endswith(".obj"):
            raise ValueError("sugar mesh must be an OBJ file")
        return load_objs_as_meshes([mesh_path]).to("cuda")

    if mesh_type in {"colmap", "milo"}:
        if not mesh_path.lower().endswith(".ply"):
            raise ValueError(f"{mesh_type} mesh must be a PLY file")
        mesh_tm = trimesh.load(mesh_path, force="mesh", process=False)
        verts = torch.tensor(mesh_tm.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh_tm.faces, dtype=torch.int64)
        colors = torch.tensor(mesh_tm.visual.vertex_colors[:, :3], dtype=torch.float32) / 255.0
        return Meshes(
            verts=[verts],
            faces=[faces],
            textures=TexturesVertex(verts_features=[colors]),
        ).to("cuda")

    raise ValueError(f"Unsupported mesh_type: {mesh_type}")


def convert_camera_from_gs_to_pytorch3d(gs_cameras, device="cuda"):
    n = len(gs_cameras)
    r = torch.tensor(np.array([cam.R for cam in gs_cameras]), dtype=torch.float32, device=device)
    t = torch.tensor(np.array([cam.T for cam in gs_cameras]), dtype=torch.float32, device=device)
    fx = torch.tensor(
        np.array([fov2focal(cam.FoVx, cam.image_width) for cam in gs_cameras]),
        dtype=torch.float32,
        device=device,
    )
    fy = torch.tensor(
        np.array([fov2focal(cam.FoVy, cam.image_height) for cam in gs_cameras]),
        dtype=torch.float32,
        device=device,
    )
    image_height = torch.tensor(
        np.array([cam.image_height for cam in gs_cameras]),
        dtype=torch.int32,
        device=device,
    )
    image_width = torch.tensor(
        np.array([cam.image_width for cam in gs_cameras]),
        dtype=torch.int32,
        device=device,
    )
    cx = image_width / 2.0
    cy = image_height / 2.0

    w2c = torch.zeros(n, 4, 4, dtype=torch.float32, device=device)
    w2c[:, :3, :3] = r.transpose(-1, -2)
    w2c[:, :3, 3] = t
    w2c[:, 3, 3] = 1.0

    c2w = w2c.inverse()
    c2w[:, :3, 1:3] *= -1
    c2w = c2w[:, :3, :]

    image_size = torch.tensor([image_width[0], image_height[0]], dtype=torch.float32, device=device)[None]
    scale = image_size.min(dim=1, keepdim=True)[0] / 2.0
    c0 = image_size / 2.0
    p0 = -(torch.tensor((cx[0], cy[0]), dtype=torch.float32, device=device)[None] - c0) / scale
    focal = torch.tensor([fx[0], fy[0]], dtype=torch.float32, device=device)[None] / scale
    k = _get_sfm_calibration_matrix(1, "cpu", focal, p0, orthographic=False).to(device).expand(n, -1, -1)

    line = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device).expand(n, -1, -1)
    cam2world = torch.cat([c2w, line], dim=1)
    world2cam = cam2world.inverse()
    r_p3d, t_p3d = world2cam.split([3, 1], dim=-1)
    r_p3d = r_p3d[:, :3].transpose(1, 2) * torch.tensor([-1.0, 1.0, -1.0], dtype=torch.float32, device=device)
    t_p3d = t_p3d.squeeze(2)[:, :3] * torch.tensor([-1.0, 1.0, -1.0], dtype=torch.float32, device=device)

    return FoVPerspectiveCameras(device=device, R=r_p3d, T=t_p3d, K=k, znear=0.0001)


def mesh_renderer_pytorch3d(viewpoint_camera, textured_mesh, image_height, image_width,
                            background_color, faces_per_pixel=1, device="cuda"):
    p3d_cameras = convert_camera_from_gs_to_pytorch3d([viewpoint_camera], device=device)
    raster_settings = RasterizationSettings(
        image_size=(image_height, image_width),
        blur_radius=0.0,
        faces_per_pixel=faces_per_pixel,
    )
    lights = AmbientLights(device=device)
    rasterizer = MeshRasterizer(cameras=p3d_cameras[0], raster_settings=raster_settings)
    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=SoftPhongShader(
            device=device,
            cameras=p3d_cameras[0],
            lights=lights,
            blend_params=BlendParams(background_color=background_color),
        ),
    )

    rgb_img = renderer(textured_mesh, cameras=p3d_cameras)[0, ..., :3]
    bg_color = rgb_img.permute(2, 0, 1).contiguous()

    fragments = rasterizer(textured_mesh, cameras=p3d_cameras)
    depth = fragments.zbuf[0, ..., 0]
    mask = fragments.pix_to_face[0, ..., 0] >= 0
    bg_depth = depth.masked_fill(~mask, -1).unsqueeze(0)
    return bg_color, bg_depth, fragments


def build_viewpoint_camera(frame, dataset_root, extension, znear, zfar):
    image_rel = frame["file_path"][2:] if frame["file_path"].startswith("./") else frame["file_path"]
    image_path = Path(dataset_root) / f"{image_rel}{extension}"

    c2w = np.array(frame["transform_matrix"], dtype=np.float32)
    c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)

    with Image.open(image_path) as image:
        width, height = image.size

    fovx = float(frame["camera_angle_x"])
    fovy = 2.0 * math.atan(math.tan(fovx / 2.0) * (height / width))

    return SimpleNamespace(
        R=np.transpose(w2c[:3, :3]),
        T=w2c[:3, 3],
        FoVx=fovx,
        FoVy=fovy,
        image_width=width,
        image_height=height,
        image_name=Path(image_rel).name,
        znear=znear,
        zfar=zfar,
    )


def render_split(textured_mesh, json_path, out_root, extension, mesh_rasterizer_type,
                 mesh_background_color, device, znear, zfar, mesh_subdir=None):
    data = load_json(json_path)
    frames = data["frames"]
    camera_angle_x = data["camera_angle_x"]
    dataset_root = Path(json_path).parent

    for idx, frame in enumerate(tqdm(frames, desc=f"Rendering {Path(json_path).name}", unit="frame")):
        frame = dict(frame)
        frame["camera_angle_x"] = camera_angle_x
        viewpoint_camera = build_viewpoint_camera(frame, dataset_root, extension, znear, zfar)

        if mesh_rasterizer_type == "pytorch3d":
            bg_color, _, _ = mesh_renderer_pytorch3d(
                viewpoint_camera,
                textured_mesh,
                image_height=viewpoint_camera.image_height,
                image_width=viewpoint_camera.image_width,
                background_color=mesh_background_color,
                device=device,
            )
        elif mesh_rasterizer_type == "nvdiffrast":
            bg_color, _, _ = mesh_renderer_nvdiffrast(
                viewpoint_camera,
                textured_mesh,
                image_height=viewpoint_camera.image_height,
                image_width=viewpoint_camera.image_width,
                background_color=mesh_background_color,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported mesh_rasterizer_type: {mesh_rasterizer_type}")

        rel = frame["file_path"][2:] if frame["file_path"].startswith("./") else frame["file_path"]
        out_path = build_output_path(out_root, rel, mesh_subdir=mesh_subdir)
        ensure_dir(out_path.parent)
        TF.to_pil_image(bg_color.detach().clamp(0, 1).cpu()).save(out_path)
        print(f"[{idx + 1}/{len(frames)}] saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="Path to textured mesh")
    parser.add_argument("--mesh_type", type=str, default="sugar", choices=["sugar", "colmap", "milo"])
    parser.add_argument("--mesh_rasterizer_type", type=str, default="pytorch3d", choices=["pytorch3d", "nvdiffrast"])
    parser.add_argument("--test_json", default="transforms_test.json")
    parser.add_argument("--train_json", default="transforms_train.json")
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--extension", default=".png")
    parser.add_argument("--bg_white", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--znear", type=float, default=0.01)
    parser.add_argument("--zfar", type=float, default=100.0)
    parser.add_argument("--mesh_start", type=int, help="Start index for a mesh filename sequence")
    parser.add_argument("--mesh_end", type=int, help="End index for a mesh filename sequence")
    parser.add_argument("--mesh_folder_prefix", default="frame", help="Per-mesh output folder prefix")
    args = parser.parse_args()

    mesh_background_color = (1.0, 1.0, 1.0) if args.bg_white else (0.0, 0.0, 0.0)
    mesh_paths = resolve_mesh_sequence(args.mesh, args.mesh_start, args.mesh_end)

    for mesh_path in mesh_paths:
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        mesh_subdir = None
        if len(mesh_paths) > 1:
            mesh_subdir = build_mesh_subdir(mesh_path, args.mesh_folder_prefix)

        print(f"Rendering mesh: {mesh_path}")
        textured_mesh = load_textured_mesh(args.mesh_type, str(mesh_path))

        render_split(
            textured_mesh=textured_mesh,
            json_path=args.test_json,
            out_root=args.out_dir,
            extension=args.extension,
            mesh_rasterizer_type=args.mesh_rasterizer_type,
            mesh_background_color=mesh_background_color,
            device=args.device,
            znear=args.znear,
            zfar=args.zfar,
            mesh_subdir=mesh_subdir,
        )
        render_split(
            textured_mesh=textured_mesh,
            json_path=args.train_json,
            out_root=args.out_dir,
            extension=args.extension,
            mesh_rasterizer_type=args.mesh_rasterizer_type,
            mesh_background_color=mesh_background_color,
            device=args.device,
            znear=args.znear,
            zfar=args.zfar,
            mesh_subdir=mesh_subdir,
        )

    print("Done.")


if __name__ == "__main__":
    main()
