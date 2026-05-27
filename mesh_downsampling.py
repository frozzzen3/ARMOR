"""
python mesh_downsampling.py \
--input_path  "/mnt/data1/syjintw/NEU/dataset/milo_meshes/bicycle/bicycle.ply" \
--output_path  "/mnt/data1/syjintw/NEU/dataset/milo_meshes/bicycle-dw10/bicycle-dw10.ply" \
--keep_percent 10
"""
import open3d as o3d
from pathlib import Path
from argparse import ArgumentParser

def downsample_mesh(input_path, output_path, keep_percent):
    # keep_percent = 10

    # # Input/output
    # input_path = "/mnt/data1/syjintw/NEU/dataset/milo_meshes/bicycle/bicycle.ply"
    # output_path = "/mnt/data1/syjintw/NEU/dataset/milo_meshes/bicycle_dw10/bicycle_dw10.ply"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Read the mesh
    mesh = o3d.io.read_triangle_mesh(input_path)
    print(mesh)

    # Make sure vertex colors exist
    if not mesh.has_vertex_colors():
        print("No vertex colors found! Check if colors are per-vertex in OBJ.")
    else:
        print("Vertex colors detected.")

    # Simplify
    target_triangles = int(len(mesh.triangles) * (keep_percent/100))  # keep 10% faces
    mesh_simplified = mesh.simplify_quadric_decimation(target_triangles)

    # Optional: smooth normals again
    mesh_simplified.compute_vertex_normals()
    print(mesh_simplified)

    # Save simplified mesh
    o3d.io.write_triangle_mesh(output_path, mesh_simplified)
    print(f"Saved simplified mesh → {output_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--input_path", type=str, help="Path to input mesh file")
    parser.add_argument("--output_path", type=str, help="Path to output simplified mesh file")
    parser.add_argument("--keep_percent", type=float, help="Percentage of faces to keep during downsampling")
    args = parser.parse_args()
    
    downsample_mesh(args.input_path, args.output_path, args.keep_percent)