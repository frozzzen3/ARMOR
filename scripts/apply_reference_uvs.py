#!/usr/bin/env python3
"""
Transfer UV coordinates from a reference OBJ onto other OBJs with matching topology.

Default behavior:
- keep each target mesh's own vertex positions
- keep each target mesh's own .mtl and texture image if present
- copy the reference mesh's `vt` section and face UV indexing
- rewrite faces as `f v/vt v/vt v/vt`

This is intended for sequences where all frames share the same topology but only one
frame has valid UVs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


Face = Tuple[Tuple[int, Optional[int], Optional[int]], ...]


@dataclass
class ObjData:
    header_lines: List[str]
    vertex_lines: List[str]
    vt_lines: List[str]
    faces: List[Face]


def parse_face_vertex(token: str) -> Tuple[int, Optional[int], Optional[int]]:
    parts = token.split("/")
    if len(parts) == 1:
        return int(parts[0]), None, None
    if len(parts) == 2:
        vt = int(parts[1]) if parts[1] else None
        return int(parts[0]), vt, None
    if len(parts) == 3:
        vt = int(parts[1]) if parts[1] else None
        vn = int(parts[2]) if parts[2] else None
        return int(parts[0]), vt, vn
    raise ValueError(f"Unsupported face token: {token}")


def load_obj(path: Path) -> ObjData:
    header_lines: List[str] = []
    vertex_lines: List[str] = []
    vt_lines: List[str] = []
    faces: List[Face] = []

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line.startswith("v "):
                vertex_lines.append(line)
            elif line.startswith("vt "):
                vt_lines.append(line)
            elif line.startswith("f "):
                tokens = line.split()[1:]
                faces.append(tuple(parse_face_vertex(tok) for tok in tokens))
            elif line.startswith("vn "):
                # Target normals are ignored. They can be recomputed by the viewer if needed.
                continue
            elif line.startswith("mtllib ") or line.startswith("usemtl "):
                continue
            else:
                header_lines.append(line)

    return ObjData(
        header_lines=header_lines,
        vertex_lines=vertex_lines,
        vt_lines=vt_lines,
        faces=faces,
    )


def load_material_name(mtl_path: Path) -> Optional[str]:
    if not mtl_path.exists():
        return None
    with mtl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("newmtl "):
                return line.split(maxsplit=1)[1].strip()
    return None


def face_vertex_indices(face: Face) -> Tuple[int, ...]:
    return tuple(item[0] for item in face)


def face_vt_indices(face: Face) -> Tuple[int, ...]:
    vt_indices = tuple(item[1] for item in face)
    if any(vt is None for vt in vt_indices):
        raise ValueError("Reference face is missing UV indices.")
    return tuple(int(vt) for vt in vt_indices)


def validate_same_topology(reference: ObjData, target: ObjData, target_path: Path) -> None:
    if len(reference.vertex_lines) != len(target.vertex_lines):
        raise ValueError(
            f"{target_path}: vertex count mismatch "
            f"({len(target.vertex_lines)} vs {len(reference.vertex_lines)})"
        )
    if len(reference.faces) != len(target.faces):
        raise ValueError(
            f"{target_path}: face count mismatch "
            f"({len(target.faces)} vs {len(reference.faces)})"
        )

    for face_idx, (ref_face, tgt_face) in enumerate(zip(reference.faces, target.faces), start=1):
        if face_vertex_indices(ref_face) != face_vertex_indices(tgt_face):
            raise ValueError(
                f"{target_path}: topology mismatch at face {face_idx}: "
                f"{face_vertex_indices(tgt_face)} != {face_vertex_indices(ref_face)}"
            )


def build_output_obj(
    target_path: Path,
    target: ObjData,
    reference: ObjData,
    material_name: str,
    material_filename: str,
) -> str:
    out_lines: List[str] = []

    for line in target.header_lines:
        out_lines.append(line)

    out_lines.append(f"mtllib {material_filename}")
    out_lines.extend(target.vertex_lines)
    out_lines.extend(reference.vt_lines)
    out_lines.append(f"usemtl {material_name}")

    for tgt_face, ref_face in zip(target.faces, reference.faces):
        v_indices = face_vertex_indices(tgt_face)
        vt_indices = face_vt_indices(ref_face)
        face_tokens = [f"{v}/{vt}" for v, vt in zip(v_indices, vt_indices)]
        out_lines.append("f " + " ".join(face_tokens))

    return "\n".join(out_lines) + "\n"


def collect_targets(mesh_dir: Path, pattern: str, reference_path: Path) -> List[Path]:
    targets = sorted(mesh_dir.glob(pattern))
    return [path for path in targets if path.resolve() != reference_path.resolve()]


def process_target(
    reference: ObjData,
    reference_mtl_name: str,
    reference_mtl_filename: str,
    target_path: Path,
    use_reference_material: bool,
    backup_suffix: str,
    dry_run: bool,
) -> None:
    target = load_obj(target_path)
    validate_same_topology(reference, target, target_path)

    target_mtl_path = target_path.with_suffix(".mtl")
    target_mtl_name = load_material_name(target_mtl_path)

    if use_reference_material:
        material_name = reference_mtl_name
        material_filename = reference_mtl_filename
    else:
        material_name = target_mtl_name or reference_mtl_name
        material_filename = target_mtl_path.name if target_mtl_path.exists() else reference_mtl_filename

    output_text = build_output_obj(
        target_path=target_path,
        target=target,
        reference=reference,
        material_name=material_name,
        material_filename=material_filename,
    )

    if dry_run:
        print(f"[DRY RUN] Would rewrite {target_path}")
        return

    backup_path = target_path.with_suffix(target_path.suffix + backup_suffix)
    if not backup_path.exists():
        backup_path.write_text(target_path.read_text(encoding="utf-8"), encoding="utf-8")

    target_path.write_text(output_text, encoding="utf-8")
    print(f"[OK] Rewrote {target_path} using reference UVs")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=Path("data/dancer/mesh_dynamic"),
        help="Directory containing OBJ/MTL/texture files.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/dancer/mesh_dynamic/dancer_0005.obj"),
        help="Reference OBJ that already has UVs.",
    )
    parser.add_argument(
        "--pattern",
        default="dancer_*.obj",
        help="Glob pattern for target OBJ files inside --mesh-dir.",
    )
    parser.add_argument(
        "--use-reference-material",
        action="store_true",
        help="Force all targets to reference the reference OBJ's material instead of their own .mtl.",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Suffix for one-time backups before rewriting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate topology and report what would be changed without writing files.",
    )
    args = parser.parse_args(argv)

    mesh_dir = args.mesh_dir.resolve()
    reference_path = args.reference.resolve()

    if not reference_path.exists():
        raise FileNotFoundError(f"Reference OBJ not found: {reference_path}")
    if not mesh_dir.exists():
        raise FileNotFoundError(f"Mesh directory not found: {mesh_dir}")

    reference = load_obj(reference_path)
    if not reference.vt_lines:
        raise ValueError(f"Reference OBJ has no UV coordinates: {reference_path}")

    reference_mtl_path = reference_path.with_suffix(".mtl")
    reference_mtl_name = load_material_name(reference_mtl_path)
    if reference_mtl_name is None:
        raise ValueError(f"Reference MTL missing or invalid: {reference_mtl_path}")

    targets = collect_targets(mesh_dir, args.pattern, reference_path)
    if not targets:
        raise ValueError("No target OBJ files found.")

    print(f"Reference: {reference_path}")
    print(f"Targets: {len(targets)}")

    for target_path in targets:
        process_target(
            reference=reference,
            reference_mtl_name=reference_mtl_name,
            reference_mtl_filename=reference_mtl_path.name,
            target_path=target_path,
            use_reference_material=args.use_reference_material,
            backup_suffix=args.backup_suffix,
            dry_run=args.dry_run,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
