import json
from pathlib import Path

import torch
from torch import nn


def sinusoidal_time_embedding(t, num_frequencies):
    if not torch.is_tensor(t):
        t = torch.tensor(float(t), dtype=torch.float32, device="cuda")
    t = t.to(dtype=torch.float32)
    if t.ndim == 0:
        t = t.view(1, 1)
    elif t.ndim == 1:
        t = t[:, None]

    freqs = (2.0 ** torch.arange(num_frequencies, device=t.device, dtype=t.dtype))[None, :]
    angles = t * freqs
    return torch.cat([t, torch.sin(angles), torch.cos(angles)], dim=-1)


class CompactTemporalAttributeModel(nn.Module):
    """
    Compact residual model for time-varying mesh-bound Gaussian attributes.

    The model is deliberately small: one learned latent per triangle plus a
    sinusoidal frame-time embedding predicts residuals for all Gaussians assigned
    to that triangle. Base Gaussian parameters remain stored in the normal PLY /
    model_params files; this module stores only temporal residual parameters.
    """

    def __init__(
        self,
        num_gaussians=None,
        latent_dim=8,
        hidden_dim=64,
        depth=3,
        time_frequencies=6,
        max_d_uvw=0.05,
        max_d_scaling=0.10,
        max_d_opacity=0.50,
        max_d_color=0.10,
        max_d_rest=0.05,
        predict_uvw=True,
        predict_scaling=True,
        predict_opacity=True,
        predict_color=True,
        predict_rest=False,
        num_rest_coeffs=0,
        deform_feature_dim=0,
        lr=1e-3,
        num_triangles=None,
        canonical_relative=True,
        canonical_time=0.0,
    ):
        super().__init__()
        # The latent table holds one row per indexed entity. Historically this was
        # keyed per triangle (same-topology sequences); for variable-topology
        # sequences it is keyed per persistent Gaussian. `num_triangles` is accepted
        # for backward compatibility with old checkpoints.
        num_entities = num_gaussians if num_gaussians is not None else num_triangles
        if num_entities is None:
            raise ValueError("CompactTemporalAttributeModel requires num_gaussians (or num_triangles).")
        self.num_gaussians = int(num_entities)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.time_frequencies = int(time_frequencies)
        self.max_d_uvw = float(max_d_uvw)
        self.max_d_scaling = float(max_d_scaling)
        self.max_d_opacity = float(max_d_opacity)
        self.max_d_color = float(max_d_color)
        self.max_d_rest = float(max_d_rest)
        self.predict_uvw = bool(predict_uvw)
        self.predict_scaling = bool(predict_scaling)
        self.predict_opacity = bool(predict_opacity)
        self.predict_color = bool(predict_color)
        # Heavy view-dependent SH (f_rest) residual. Frozen + shared in the base, so per-frame
        # view-dependent appearance variation -- the detail later frames otherwise lose, since
        # only the canonical frame is fit with a free f_rest -- is carried here instead.
        self.predict_rest = bool(predict_rest)
        self.num_rest_coeffs = int(num_rest_coeffs)
        # Per-Gaussian local-deformation feature (canonical face geometry vs. this frame's
        # bound face). Conditioning on the actual deformation -- MaGS-style -- instead of a raw
        # scalar time lets the residual generalize across frames rather than memorize each one.
        # 0 disables it (legacy time-only behaviour / backward-compatible checkpoints).
        self.deform_feature_dim = int(deform_feature_dim)
        self.lr = float(lr)
        # When True, residuals are measured relative to the canonical time so that the
        # residual is exactly zero at the canonical frame. This pins the base Gaussian
        # params to the canonical appearance and forces the temporal model to represent
        # only the per-frame DELTA (prevents it from absorbing a constant offset that
        # the base then mirrors, which collapses/saturates the residual).
        self.canonical_relative = bool(canonical_relative)
        self.canonical_time = float(canonical_time)

        self.triangle_latent = nn.Embedding(self.num_gaussians, self.latent_dim)
        time_dim = 1 + 2 * self.time_frequencies
        input_dim = self.latent_dim + time_dim + 3 + self.deform_feature_dim

        # Canonical-frame deformation feature per Gaussian, used as the reference in the
        # canonical-relative subtraction (so the residual is exactly zero at the canonical
        # frame). Persisted in the state_dict so render reproduces the same reference.
        if self.deform_feature_dim > 0:
            self.register_buffer(
                "canonical_deform_feat",
                torch.zeros(self.num_gaussians, self.deform_feature_dim),
            )
        else:
            self.canonical_deform_feat = None

        layers = []
        last_dim = input_dim
        for _ in range(max(1, self.depth)):
            layers.append(nn.Linear(last_dim, self.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            last_dim = self.hidden_dim
        self.backbone = nn.Sequential(*layers)

        output_dim = 0
        self._slices = {}
        if self.predict_uvw:
            self._slices["d_uvw"] = slice(output_dim, output_dim + 3)
            output_dim += 3
        if self.predict_scaling:
            self._slices["d_scaling"] = slice(output_dim, output_dim + 3)
            output_dim += 3
        if self.predict_opacity:
            self._slices["d_opacity"] = slice(output_dim, output_dim + 1)
            output_dim += 1
        if self.predict_color:
            self._slices["d_features_dc"] = slice(output_dim, output_dim + 3)
            output_dim += 3
        if self.predict_rest:
            if self.num_rest_coeffs <= 0:
                raise ValueError("predict_rest=True requires num_rest_coeffs > 0.")
            rest_dim = 3 * self.num_rest_coeffs
            self._slices["d_features_rest"] = slice(output_dim, output_dim + rest_dim)
            output_dim += rest_dim
        if output_dim == 0:
            raise ValueError("At least one temporal attribute prediction must be enabled.")

        self.head = nn.Linear(self.hidden_dim, output_dim)
        nn.init.normal_(self.triangle_latent.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, eps=1e-15)

    def config(self):
        return {
            "num_gaussians": self.num_gaussians,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "time_frequencies": self.time_frequencies,
            "max_d_uvw": self.max_d_uvw,
            "max_d_scaling": self.max_d_scaling,
            "max_d_opacity": self.max_d_opacity,
            "max_d_color": self.max_d_color,
            "max_d_rest": self.max_d_rest,
            "predict_uvw": self.predict_uvw,
            "predict_scaling": self.predict_scaling,
            "predict_opacity": self.predict_opacity,
            "predict_color": self.predict_color,
            "predict_rest": self.predict_rest,
            "num_rest_coeffs": self.num_rest_coeffs,
            "deform_feature_dim": self.deform_feature_dim,
            "lr": self.lr,
            "canonical_relative": self.canonical_relative,
            "canonical_time": self.canonical_time,
        }

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def set_canonical_deform_feature(self, feat):
        """Store the per-Gaussian canonical-frame deformation feature (the reference used by
        the canonical-relative subtraction). Captured once at the canonical frame."""
        if self.deform_feature_dim <= 0:
            return
        feat = feat.detach().to(self.canonical_deform_feat)
        if feat.shape != self.canonical_deform_feat.shape:
            raise ValueError(
                f"canonical deform feature shape {tuple(feat.shape)} != expected "
                f"{tuple(self.canonical_deform_feat.shape)}"
            )
        self.canonical_deform_feat.copy_(feat)

    def _raw_head(self, latent, uvw, frame_time, deform_feat):
        time_emb = sinusoidal_time_embedding(frame_time, self.time_frequencies)
        time_emb = time_emb.to(device=uvw.device, dtype=uvw.dtype).expand(uvw.shape[0], -1)
        parts = [latent, time_emb, uvw]
        if self.deform_feature_dim > 0:
            parts.append(deform_feat)
        return self.head(self.backbone(torch.cat(parts, dim=-1)))

    def forward(self, indices, uvw, frame_time, deform_feat=None):
        indices = indices.to(dtype=torch.long)
        uvw = uvw.detach()
        latent = self.triangle_latent(indices)

        canon_feat = None
        if self.deform_feature_dim > 0:
            if deform_feat is None:
                raise ValueError("deform_feature_dim > 0 but no deform_feat was provided.")
            deform_feat = deform_feat.detach().to(device=uvw.device, dtype=uvw.dtype)
            # The canonical reference uses the canonical-frame geometry for the same Gaussians,
            # so the residual is driven by how the local mesh deformed (and is exactly zero at
            # the canonical frame, where current geometry == canonical geometry).
            canon_feat = self.canonical_deform_feat[indices].to(device=uvw.device, dtype=uvw.dtype)

        raw = self._raw_head(latent, uvw, frame_time, deform_feat)
        # Canonical-relative: subtract the canonical prediction so the residual is exactly zero
        # at the canonical frame and only the per-frame delta survives.
        raw_canon = (
            self._raw_head(latent, uvw, self.canonical_time, canon_feat)
            if self.canonical_relative
            else None
        )

        def residual(key, max_d):
            r = torch.tanh(raw[:, self._slices[key]]) * max_d
            if raw_canon is not None:
                r = r - torch.tanh(raw_canon[:, self._slices[key]]) * max_d
            return r

        out = {
            "d_uvw": None,
            "d_scaling": None,
            "d_opacity": None,
            "d_features_dc": None,
            "d_features_rest": None,
        }
        if "d_uvw" in self._slices:
            out["d_uvw"] = residual("d_uvw", self.max_d_uvw)
        if "d_scaling" in self._slices:
            out["d_scaling"] = residual("d_scaling", self.max_d_scaling)
        if "d_opacity" in self._slices:
            out["d_opacity"] = residual("d_opacity", self.max_d_opacity)
        if "d_features_dc" in self._slices:
            out["d_features_dc"] = residual("d_features_dc", self.max_d_color)[:, None, :]
        if "d_features_rest" in self._slices:
            r = residual("d_features_rest", self.max_d_rest)
            out["d_features_rest"] = r.reshape(r.shape[0], self.num_rest_coeffs, 3)
        return out

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.config(), "state_dict": self.state_dict()}, path)
        with open(path.with_suffix(".json"), "w") as fh:
            json.dump({"config": self.config(), "parameter_count": self.parameter_count}, fh, indent=2)

    @classmethod
    def load(cls, path, device="cuda"):
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.optimizer = torch.optim.Adam(model.parameters(), lr=model.lr, eps=1e-15)
        return model


def estimate_compact_temporal_storage(gaussians, temporal_model, num_frames):
    base_numel = 0
    for attr in ("_uvw", "_scale", "_features_dc", "_features_rest", "_opacity"):
        tensor = getattr(gaussians, attr, None)
        if torch.is_tensor(tensor):
            base_numel += tensor.numel()

    base_bytes = base_numel * 4
    duplicated_bytes = base_bytes * max(1, int(num_frames))
    compact_bytes = base_bytes + temporal_model.parameter_count * 4
    saved_bytes = max(0, duplicated_bytes - compact_bytes)
    savings_ratio = saved_bytes / duplicated_bytes if duplicated_bytes > 0 else 0.0
    return {
        "num_frames": int(num_frames),
        "base_gaussian_parameter_bytes": int(base_bytes),
        "duplicated_per_frame_bytes": int(duplicated_bytes),
        "compact_temporal_bytes": int(compact_bytes),
        "temporal_model_parameter_count": int(temporal_model.parameter_count),
        "estimated_saved_bytes": int(saved_bytes),
        "estimated_savings_ratio": float(savings_ratio),
    }
