#
# Diagnostic for variable-topology compact rendering.
#
# Localizes the gap between the compact reconstruction of a frame
# (persistent base + cached binding + temporal residual) and that frame's own
# trained checkpoint, by comparing the actual Gaussian tensors. Runs purely on
# saved training artifacts (plys, model_params.pt, bindings, temporal model) -
# no cameras or dataset required.
#
# Usage (--sh_degree MUST match training; default training sh_degree is 3, so f_rest has
# 15 coeffs. Using 0 here would mismatch the ply and imply no f_rest at all):
#   PYTHONPATH=. python diagnose_compact.py \
#       --output output/dancer_test --canonical_frame 1 --frame 3 \
#       --iteration 5000 --mesh_start 1 --mesh_end 3 --sh_degree 3 \
#       --temporal_checkpoint output/dancer_test/temporal_attr_model.pth
#

import argparse
from pathlib import Path

import torch

from scene.gaussian_mesh_model import GaussianMeshModel
from scene.temporal_attribute_model import CompactTemporalAttributeModel


def frame_dir(output, frame, iteration):
    fid = f"frame_{frame:04d}"
    return Path(output) / fid / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"


def rmse(a, b):
    return float(torch.sqrt(torch.mean((a.float() - b.float()) ** 2)).item())


def stat(t):
    t = t.float()
    return f"mean|.|={t.abs().mean().item():.6f}  max|.|={t.abs().max().item():.6f}"


def load_model(ply_path, sh_degree):
    g = GaussianMeshModel(sh_degree)
    g.load_ply(str(ply_path))
    g.active_sh_degree = sh_degree
    return g


def snapshot(g):
    """Return the rendered Gaussian attributes (temporal-aware get_* properties)."""
    return {
        "xyz": g.get_xyz.detach().clone(),
        "opacity": g.get_opacity.detach().clone(),
        "scaling": g.get_scaling.detach().clone(),
        "features": g.get_features.detach().clone(),
    }


def main():
    ap = argparse.ArgumentParser(description="Diagnose variable-topology compact rendering gap")
    ap.add_argument("--output", required=True, help="training output dir (root, holds frame_XXXX/ and bindings/)")
    ap.add_argument("--canonical_frame", type=int, default=1, help="frame whose checkpoint is the persistent base")
    ap.add_argument("--frame", type=int, required=True, help="frame to diagnose (a non-canonical frame)")
    ap.add_argument("--iteration", type=int, required=True, help="iteration subdir of the checkpoints")
    ap.add_argument("--canonical_iteration", type=int, default=None, help="iteration for the base frame (defaults to --iteration)")
    ap.add_argument("--mesh_start", type=int, required=True)
    ap.add_argument("--mesh_end", type=int, required=True)
    ap.add_argument("--sh_degree", type=int, default=3, help="MUST match training (default 3); "
                    "0 means no f_rest exists")
    ap.add_argument("--binding_dir", type=str, default=None, help="defaults to <output>/bindings")
    ap.add_argument("--temporal_checkpoint", type=str, default=None, help="defaults to <output>/temporal_attr_model.pth")
    args = ap.parse_args()

    canon_it = args.canonical_iteration or args.iteration
    base_ply = frame_dir(args.output, args.canonical_frame, canon_it)
    frame_ply = frame_dir(args.output, args.frame, args.iteration)
    binding_dir = Path(args.binding_dir) if args.binding_dir else Path(args.output) / "bindings"
    binding_path = binding_dir / f"frame_{args.frame:04d}.pt"
    temporal_path = Path(args.temporal_checkpoint) if args.temporal_checkpoint else Path(args.output) / "temporal_attr_model.pth"

    for p in (base_ply, frame_ply, binding_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    frame_time = 0.0
    if args.mesh_end != args.mesh_start:
        frame_time = float(args.frame - args.mesh_start) / float(args.mesh_end - args.mesh_start)

    print("=" * 78)
    print(f"Diagnosing frame {args.frame}  (frame_time={frame_time:.4f})")
    print(f"  base  : {base_ply}")
    print(f"  frame : {frame_ply}")
    print(f"  bind  : {binding_path}")
    print(f"  temp  : {temporal_path}")
    print("=" * 78)

    # --- Reference: the frame's OWN trained checkpoint (binding + frozen base, no temporal baked in) ---
    g_ref = load_model(frame_ply, args.sh_degree)

    # --- Compact: persistent base (canonical frame) + cached binding for this frame ---
    cached = torch.load(binding_path, map_location="cuda")
    g_cmp = load_model(base_ply, args.sh_degree)
    g_cmp.apply_cached_binding(
        g_ref.vertices.detach(), g_ref.faces.detach(),
        cached["triangle_indices"], cached["uvw"],
        scale=cached.get("scale"), opacity=cached.get("opacity"),
    )

    # === 1. Did the base appearance freeze hold? (base==canonical vs frame's stored base) ===
    print("\n[1] Freeze check  (frame's stored base appearance vs canonical base; ~0 means freeze held)")
    print(f"    features_dc  RMSE = {rmse(g_ref._features_dc, g_cmp._features_dc):.6e}")
    print(f"    features_rest RMSE = {rmse(g_ref._features_rest, g_cmp._features_rest):.6e}")
    print(f"    opacity      RMSE = {rmse(g_ref._opacity, g_cmp._opacity):.6e}")
    print(f"    scale        RMSE = {rmse(g_ref._scale, g_cmp._scale):.6e}")

    # === 2. Does the cached binding match the frame's stored binding? ===
    print("\n[2] Binding check  (cached binding vs frame's stored _uvw / triangle_indices; ~0 means cache is faithful)")
    print(f"    _uvw RMSE = {rmse(g_ref._uvw, g_cmp._uvw):.6e}")
    same_tri = torch.equal(g_ref.triangle_indices.cpu(), g_cmp.triangle_indices.cpu())
    print(f"    triangle_indices identical = {same_tri}")

    # === 3. Geometry reproduction (no temporal) ===
    g_ref.clear_temporal_attributes(); g_cmp.clear_temporal_attributes()
    print("\n[3] Geometry (no temporal): get_xyz compact vs reference")
    print(f"    xyz RMSE = {rmse(g_ref.get_xyz, g_cmp.get_xyz):.6e}")

    # === 4. Is the temporal model actually producing a residual at this frame_time? ===
    temporal = CompactTemporalAttributeModel.load(str(temporal_path), device="cuda")
    for g in (g_ref, g_cmp):
        g.temporal_per_gaussian = True

    print("\n[4a] Temporal model config  (is the f_rest / geometry path even enabled?)")
    print(f"    predict_color      = {temporal.predict_color}")
    print(f"    predict_rest       = {temporal.predict_rest}    num_rest_coeffs = {temporal.num_rest_coeffs}")
    print(f"    max_d_color        = {temporal.max_d_color}    max_d_rest = {temporal.max_d_rest}")
    print(f"    deform_feature_dim = {temporal.deform_feature_dim}  (0 => geometry conditioning OFF, time-only)")
    print(f"    canonical_relative = {temporal.canonical_relative}    canonical_time = {temporal.canonical_time}")
    print(f"    output head dim    = {temporal.head.out_features}    param_count = {temporal.parameter_count}")
    if temporal.deform_feature_dim > 0 and temporal.canonical_deform_feat is not None:
        cf = temporal.canonical_deform_feat
        nonzero = float(cf.abs().sum().item())
        print(f"    canonical_deform_feat populated = {nonzero > 0}  ({stat(cf)})")
        if nonzero == 0.0:
            print("      ^ WARNING: canonical geometry reference is all-zero -- set_canonical_deform_feature "
                  "was never called; geometry conditioning is degenerate.")

    print(f"\n[4b] Temporal residual magnitudes  (G={g_cmp._uvw.shape[0]}, model num_gaussians={temporal.num_gaussians})")
    print("     t=0 should be ~0 (canonical-relative); t=N should be NONZERO if the model learned per-frame change.")
    base_t0 = snapshot_temporal_residual(g_cmp, temporal, 0.0)
    base_tN = snapshot_temporal_residual(g_cmp, temporal, frame_time)
    for k in ("d_uvw", "d_scaling", "d_opacity", "d_features_dc", "d_features_rest"):
        r0 = base_t0.get(k); rN = base_tN.get(k)
        s0 = stat(r0) if r0 is not None else "None"
        sN = stat(rN) if rN is not None else "None"
        print(f"    {k:16s}  t=0: {s0}")
        print(f"    {'':16s}  t={frame_time:.3f}: {sN}")

    # How big is the f_rest residual RELATIVE to the base it sits on top of? A residual that is
    # tiny vs the base (or pinned at the clamp) explains an invisible difference.
    if base_tN.get("d_features_rest") is not None:
        dr = base_tN["d_features_rest"].float()
        base_rest = g_cmp._features_rest.float()
        clamp_frac = float((dr.abs() >= 0.999 * temporal.max_d_rest).float().mean().item())
        ratio = dr.abs().mean().item() / (base_rest.abs().mean().item() + 1e-9)
        print(f"    f_rest residual vs base: mean|res|/mean|base| = {ratio:.4f}   "
              f"fraction at clamp(|res|>=max_d_rest) = {clamp_frac:.3f}")
        if clamp_frac > 0.2:
            print("      ^ many coeffs are saturating the clamp -> raise --temporal_max_d_rest.")

    # === 5. How much does the temporal model change the EFFECTIVE appearance at this frame? ===
    # base (no temporal) vs effective (with temporal), on the compact model. If this is ~0 the
    # render is identical to the frozen canonical appearance -> no visible per-frame change.
    g_cmp.clear_temporal_attributes()
    feat_base = g_cmp.get_features.detach().clone()
    dc_slice = slice(0, 1)
    rest_slice = slice(1, feat_base.shape[1])
    g_cmp.apply_temporal_attributes(temporal, frame_time)
    feat_eff = g_cmp.get_features.detach().clone()
    print("\n[5] Effective appearance CHANGE from temporal  (base==canonical vs base+residual at frame N)")
    print(f"    DC   change RMSE = {rmse(feat_eff[:, dc_slice], feat_base[:, dc_slice]):.6e}")
    if feat_base.shape[1] > 1:
        print(f"    rest change RMSE = {rmse(feat_eff[:, rest_slice], feat_base[:, rest_slice]):.6e}")
    else:
        print("    rest change RMSE = n/a  (sh_degree=0 -> no f_rest exists; nothing for the rest head to do)")

    # === 6. Effective appearance reproduction WITH temporal: compact vs reference ===
    g_ref.apply_temporal_attributes(temporal, frame_time)
    a_ref, a_cmp = snapshot(g_ref), snapshot(g_cmp)
    print("\n[6] Reproduction WITH temporal  (compact vs reference; ~0 means compact render == per-frame checkpoint)")
    for k in ("xyz", "opacity", "scaling", "features"):
        print(f"    {k:9s} RMSE = {rmse(a_ref[k], a_cmp[k]):.6e}")

    print("\n" + "=" * 78)
    print("Interpretation:")
    print("  [1] large  -> base did NOT freeze: per-frame base drifted from canonical (render base is wrong).")
    print("  [2] large  -> cached binding != trained binding (cache write/order bug).")
    print("  [4a] predict_rest=False / deform_feature_dim=0 -> you are NOT running the new path")
    print("       (retrain after the merge, with --temporal_predict_rest; check sh_degree>0 so f_rest exists).")
    print("  [4b] d_features_rest ~0 at t=N -> rest head learned nothing (undertrained, clamp too small,")
    print("       or f_rest simply doesn't vary frame-to-frame for this sequence).")
    print("  [5] DC & rest change ~0 -> temporal makes NO appearance change vs the canonical base, so every")
    print("       frame renders like frame 1 == the 'no obvious difference' you saw. Nonzero here = it IS working.")
    print("  [6] large but [1][2] ~0 -> reproduction path bug (how temporal/base combine at render).")
    print("  all ~0 -> compact == per-frame checkpoint; remaining visual gap is the per-frame")
    print("           checkpoint vs the live training-debug state (e.g. saved temporal != debug-time).")
    print("=" * 78)


def snapshot_temporal_residual(g, temporal, frame_time):
    """Run the temporal model and return its raw residual dict (without committing geometry)."""
    base_uvw = g._decode_uvw(include_temporal=False)
    if g.temporal_per_gaussian:
        idx = torch.arange(g._uvw.shape[0], device=g._uvw.device)
    else:
        idx = g.triangle_indices
    deform_feat = None
    if getattr(temporal, "deform_feature_dim", 0) > 0:
        # Geometry conditioning: feed this frame's local-deformation feature, exactly as the
        # training/render path does. Without it the forward pass raises.
        deform_feat = g.compute_deform_feature()
    with torch.no_grad():
        return temporal(idx, base_uvw, frame_time, deform_feat=deform_feat)


if __name__ == "__main__":
    main()
