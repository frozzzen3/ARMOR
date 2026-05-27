import torch
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes


def ensure_mesh_has_texture(mesh, fallback_color=(1.0, 1.0, 1.0)):
    if getattr(mesh, "textures", None) is not None:
        return mesh

    verts_features = []
    for verts in mesh.verts_list():
        color = torch.tensor(fallback_color, dtype=verts.dtype, device=verts.device)
        verts_features.append(color.view(1, 3).expand(verts.shape[0], 3))

    print("[WARNING] PyTorch3D mesh has no textures; using fallback vertex colors.")
    return Meshes(
        verts=mesh.verts_list(),
        faces=mesh.faces_list(),
        textures=TexturesVertex(verts_features=verts_features),
    )
