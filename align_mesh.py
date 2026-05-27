import argparse
import copy
import os
import numpy as np
import open3d as o3d


def get_bbox(mesh, bbox_type="aabb"):
    if bbox_type == "obb":
        return mesh.get_oriented_bounding_box()
    return mesh.get_axis_aligned_bounding_box()


def get_anchor_point(bbox, anchor="center"):
    min_b = bbox.get_min_bound()
    max_b = bbox.get_max_bound()
    center = bbox.get_center()

    anchor = anchor.lower()
    if anchor == "center":
        return center
    elif anchor == "min":
        return min_b
    elif anchor == "max":
        return max_b
    elif anchor == "bottom_center":
        return np.array([center[0], center[1], min_b[2]])
    elif anchor == "top_center":
        return np.array([center[0], center[1], max_b[2]])
    else:
        raise ValueError(
            f"Unknown anchor '{anchor}'. "
            f"Choose from: center, min, max, bottom_center, top_center"
        )


def safe_extent_ratio(src_bbox, tgt_bbox):
    src_extent = np.asarray(src_bbox.get_extent(), dtype=np.float64)
    tgt_extent = np.asarray(tgt_bbox.get_extent(), dtype=np.float64)

    valid = src_extent > 1e-12
    if not np.any(valid):
        raise ValueError("Source mesh bounding box has near-zero extent in all dimensions.")

    ratios = tgt_extent[valid] / src_extent[valid]
    return float(np.mean(ratios))


def load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
    if mesh.is_empty():
        raise ValueError(f"Failed to load mesh: {path}")
    return mesh


def print_mesh_info(name, mesh):
    print(f"\n{name}")
    print(f"  Vertices : {len(mesh.vertices)}")
    print(f"  Triangles: {len(mesh.triangles)}")
    print(f"  Has UVs  : {mesh.has_triangle_uvs()}")
    print(f"  Has tex  : {mesh.has_textures()}")
    print(f"  Has vtx colors: {mesh.has_vertex_colors()}")


def main():
    parser = argparse.ArgumentParser(
        description="Scale and align a textured mesh to a target mesh while preserving texture."
    )
    parser.add_argument("--source", required=True, help="Path to textured source mesh")
    parser.add_argument("--target", required=True, help="Path to target mesh")
    parser.add_argument("--output", required=True, help="Output path for transformed source mesh")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Manual scale multiplier applied to the source mesh (default: 1.0)",
    )
    parser.add_argument(
        "--auto_scale",
        action="store_true",
        help="Automatically scale source to roughly match target bbox size",
    )
    parser.add_argument(
        "--bbox_type",
        choices=["aabb", "obb"],
        default="aabb",
        help="Bounding box type used for alignment/scaling (default: aabb)",
    )
    parser.add_argument(
        "--anchor",
        choices=["center", "min", "max", "bottom_center", "top_center"],
        default="center",
        help="Anchor point used for alignment (default: center)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize source aligned with target before saving",
    )

    args = parser.parse_args()

    source_mesh = load_mesh(args.source)
    target_mesh = load_mesh(args.target)

    print_mesh_info("Source mesh", source_mesh)
    print_mesh_info("Target mesh", target_mesh)

    transformed_mesh = copy.deepcopy(source_mesh)

    src_bbox = get_bbox(transformed_mesh, args.bbox_type)
    tgt_bbox = get_bbox(target_mesh, args.bbox_type)

    src_anchor_before = get_anchor_point(src_bbox, args.anchor)
    tgt_anchor = get_anchor_point(tgt_bbox, args.anchor)

    final_scale = args.scale
    if args.auto_scale:
        auto_ratio = safe_extent_ratio(src_bbox, tgt_bbox)
        final_scale *= auto_ratio
        print(f"\nAuto scale ratio: {auto_ratio:.6f}")

    print(f"Final scale applied: {final_scale:.6f}")

    # Scale around the chosen source anchor so the object grows/shrinks in place
    transformed_mesh.scale(final_scale, center=src_anchor_before)

    # Recompute bbox/anchor after scaling
    src_bbox_after = get_bbox(transformed_mesh, args.bbox_type)
    src_anchor_after = get_anchor_point(src_bbox_after, args.anchor)

    # Translate so chosen anchor matches target anchor
    translation = tgt_anchor - src_anchor_after
    transformed_mesh.translate(translation)

    print(f"Translation applied: {translation}")

    # Normals are useful for visualization/rendering
    transformed_mesh.compute_vertex_normals()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ok = o3d.io.write_triangle_mesh(
        args.output,
        transformed_mesh,
        write_triangle_uvs=True,
        write_vertex_normals=True,
        write_vertex_colors=True,
    )

    if not ok:
        raise RuntimeError(f"Failed to save mesh to: {args.output}")

    print(f"\nSaved transformed textured mesh to:\n  {args.output}")

    if args.visualize:
        # Paint target slightly for easier inspection
        target_vis = copy.deepcopy(target_mesh)
        if not target_vis.has_vertex_colors() and not target_vis.has_textures():
            target_vis.paint_uniform_color([0.7, 0.7, 0.7])

        o3d.visualization.draw_geometries(
            [target_vis, transformed_mesh],
            window_name="Aligned textured source + target",
            mesh_show_back_face=True,
        )


if __name__ == "__main__":
    main()