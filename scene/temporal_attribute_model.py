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
        predict_uvw=True,
        predict_scaling=True,
        predict_opacity=True,
        predict_color=True,
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
        self.predict_uvw = bool(predict_uvw)
        self.predict_scaling = bool(predict_scaling)
        self.predict_opacity = bool(predict_opacity)
        self.predict_color = bool(predict_color)
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
        input_dim = self.latent_dim + time_dim + 3

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
            "predict_uvw": self.predict_uvw,
            "predict_scaling": self.predict_scaling,
            "predict_opacity": self.predict_opacity,
            "predict_color": self.predict_color,
            "lr": self.lr,
            "canonical_relative": self.canonical_relative,
            "canonical_time": self.canonical_time,
        }

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def _raw_head(self, latent, uvw, frame_time):
        time_emb = sinusoidal_time_embedding(frame_time, self.time_frequencies)
        time_emb = time_emb.to(device=uvw.device, dtype=uvw.dtype).expand(uvw.shape[0], -1)
        return self.head(self.backbone(torch.cat([latent, time_emb, uvw], dim=-1)))

    def forward(self, indices, uvw, frame_time):
        indices = indices.to(dtype=torch.long)
        uvw = uvw.detach()
        latent = self.triangle_latent(indices)
        raw = self._raw_head(latent, uvw, frame_time)
        # Canonical-relative: subtract the canonical-time prediction so the residual is
        # exactly zero at the canonical frame and only the time-varying delta survives.
        raw_canon = self._raw_head(latent, uvw, self.canonical_time) if self.canonical_relative else None

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
        }
        if "d_uvw" in self._slices:
            out["d_uvw"] = residual("d_uvw", self.max_d_uvw)
        if "d_scaling" in self._slices:
            out["d_scaling"] = residual("d_scaling", self.max_d_scaling)
        if "d_opacity" in self._slices:
            out["d_opacity"] = residual("d_opacity", self.max_d_opacity)
        if "d_features_dc" in self._slices:
            out["d_features_dc"] = residual("d_features_dc", self.max_d_color)[:, None, :]
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
