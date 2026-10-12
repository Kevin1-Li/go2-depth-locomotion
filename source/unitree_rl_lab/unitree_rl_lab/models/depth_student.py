from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class _DepthEncoder(nn.Module):
    """Original CNN depth encoder (flatten + FC) used by the ``concat`` fusion."""

    def __init__(self, in_channels: int, feature_dim: int, in_height: int, in_width: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        out_h = in_height // 8
        out_w = in_width // 8
        self.fc = nn.Sequential(
            nn.Linear(64 * out_h * out_w, feature_dim),
            nn.ReLU(),
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(depth).flatten(1))


class _DepthEncoderSpatial(nn.Module):
    """CNN depth encoder that keeps the (B, C, 8, 8) spatial feature map.

    Identical convolutional stack to ``_DepthEncoder`` but without the flatten+FC, so the
    ``cross_attention`` fusion can tokenise the 8x8 spatial grid (64 tokens x 64 channels).
    """

    def __init__(self, in_channels: int, in_height: int, in_width: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.out_channels = 64
        self.out_h = in_height // 8
        self.out_w = in_width // 8

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.conv(depth)  # (B, 64, H/8, W/8)


class _ProprioEncoder(nn.Module):
    def __init__(self, in_dim: int, feature_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, feature_dim),
            nn.ReLU(),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.mlp(proprio)


class _CrossAttentionFusion(nn.Module):
    """Proprio-conditioned cross-attention over depth spatial tokens.

    Depth CNN spatial features ``(B, C, H, W)`` are flattened into ``num_tokens = H*W``
    tokens and linearly projected to ``embed_dim``; a learnable 2-D positional embedding
    (one vector per grid position, laid out in row-major order) is added. These projected
    tokens become both Key and Value. The proprio feature ``(B, embed_dim)`` is mapped by a
    linear layer to ``num_queries`` Query tokens ``(B, num_queries, embed_dim)``.

    A single multi-head cross-attention block (residual + LayerNorm) followed by an FFN
    (residual + LayerNorm) processes the queries; the queries are then mean-pooled to a
    single ``(B, embed_dim)`` vector. No additional transformer blocks are stacked in this
    first version.
    """

    def __init__(
        self,
        token_dim: int,
        embed_dim: int,
        num_tokens: int,
        num_queries: int,
        num_heads: int,
        ffn_dim: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_heads = num_heads

        self.token_proj = nn.Linear(token_dim, embed_dim)
        # Learnable positional embedding for the H*W grid (row-major flatten order).
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        # proprio (B, embed_dim) -> num_queries query tokens (B, num_queries, embed_dim).
        self.query_proj = nn.Linear(embed_dim, num_queries * embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        spatial_feat: torch.Tensor,
        proprio_feat: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b = spatial_feat.shape[0]
        # (B, C, H, W) -> (B, num_tokens, C) row-major
        tokens = spatial_feat.flatten(2).transpose(1, 2)
        tokens = self.token_proj(tokens)  # (B, num_tokens, embed_dim)
        tokens = tokens + self.pos_embed  # (B, num_tokens, embed_dim)

        q = self.query_proj(proprio_feat).view(b, self.num_queries, self.embed_dim)  # (B, Q, E)
        k = v = tokens

        attn_out, attn_weights = self.cross_attn(
            q, k, v, average_attn_weights=False
        )  # (B, Q, E), (B, heads, Q, num_tokens)

        x = self.norm1(q + attn_out)
        x = self.norm2(x + self.ffn(x))
        pooled = x.mean(dim=1)  # (B, E)

        return pooled, (attn_weights if return_attention else None)


class _DepthPatcher(nn.Module):
    """ViT-style depth patch tokeniser for the ``patch_ca`` / ``lstm_patch_ca`` fusions.

    Splits a single-channel depth image ``(B, 1, H, W)`` into ``patch_size x patch_size``
    non-overlapping patches and flattens each patch into one token. Tokens are laid out in
    row-major order: token index ``r * num_patches_w + c`` covers image rows
    ``[r*P, (r+1)*P)`` and columns ``[c*P, (c+1)*P)``.
    """

    def __init__(self, in_height: int, in_width: int, patch_size: int = 4, in_channels: int = 1):
        super().__init__()
        if in_height % patch_size or in_width % patch_size:
            raise ValueError(
                f"depth {in_height}x{in_width} not divisible by patch_size={patch_size}."
            )
        self.patch_size = patch_size
        self.num_patches_h = in_height // patch_size
        self.num_patches_w = in_width // patch_size
        self.num_tokens = self.num_patches_h * self.num_patches_w
        self.token_dim = in_channels * patch_size * patch_size

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, num_patches_h, P, num_patches_w, P) -> (B, tokens, C*P*P)
        b, c, h, w = depth.shape
        p = self.patch_size
        x = depth.reshape(b, c, self.num_patches_h, p, self.num_patches_w, p)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, nph, npw, C, P, P)
        return x.reshape(b, self.num_tokens, c * p * p)


class _PatchCrossAttention(nn.Module):
    """Single-query cross-attention: current-frame proprio is the Query, depth patch tokens are K/V.

    ``patch_tokens (B, num_tokens, token_dim)`` are linearly projected to ``embed_dim`` and
    summed with a learnable positional embedding (row-major grid). The current-frame proprio
    ``x_t (B, proprio_dim)`` is projected to a single query token ``(B, 1, embed_dim)``.
    A single multi-head cross-attention block (residual + LayerNorm) with an optional FFN
    (residual + LayerNorm) outputs one context vector per query; the single query is squeezed
    to ``(B, embed_dim)``.
    """

    def __init__(
        self,
        token_dim: int,
        proprio_dim: int,
        embed_dim: int,
        num_tokens: int,
        num_heads: int,
        ffn_dim: int,
        use_ffn: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_ffn = use_ffn

        self.token_proj = nn.Linear(token_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.query_proj = nn.Linear(proprio_dim, embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        x_t: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        tokens = self.token_proj(patch_tokens) + self.pos_embed  # (B, N, E)
        q = self.query_proj(x_t).unsqueeze(1)  # (B, 1, E)
        k = v = tokens

        attn_out, attn_weights = self.cross_attn(
            q, k, v, average_attn_weights=False
        )  # (B, 1, E), (B, heads, 1, N)

        x = self.norm1(q + attn_out)
        if self.use_ffn:
            x = self.norm2(x + self.ffn(x))
        c_t = x.squeeze(1)  # (B, E)

        return c_t, (attn_weights if return_attention else None)


class _ProprioHistoryLSTM(nn.Module):
    """LSTM over an ordered proprio history window ``(B, history_len, proprio_dim)``.

    Returns the output at the final timestep of the window, ``(B, hidden_dim)``.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(history)  # (B, T, hidden)
        return out[:, -1, :]  # (B, hidden)


class _FusionMLP(nn.Module):
    """Fusion MLP mapping the concatenated latent inputs to the Student latent ``z_dim``."""

    def __init__(self, in_dim: int, z_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def _mlp_head(in_dim: int, action_dim: int, head_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for out_dim in head_dims:
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())
        in_dim = out_dim
    layers.append(nn.Linear(in_dim, action_dim))
    return nn.Sequential(*layers)


class DepthStudentPolicy(nn.Module):
    """Deployable Student policy with four fusion modes sharing one external API.

    ``forward`` / ``mean_action`` / ``update_distribution`` / ``act`` / ``std`` / ``log_std``
    all take ``(depth, proprio)``:

    * ``concat`` (original)::
          depth (N,1,H,W) -> _DepthEncoder -> feat_d (256)
          proprio (N,45)   -> _ProprioEncoder -> feat_p (128)
          concat (384) -> MLP -> 12-D action mean

    * ``cross_attention`` (legacy, 4-query)::
          depth -> CNN -> 64 spatial tokens -> proj 128 -> K,V
          proprio -> MLP -> 4 proprio-conditioned Query -> CA -> pool -> concat proprio -> MLP -> 12-D

    * ``patch_ca`` (new, no LSTM)::
          depth (N,1,32,48) -> 4x4 patches -> 96 tokens x 16 -> proj 128 + pos-embed -> K,V
          proprio x_t (N,45) -> Query proj -> single Query (N,1,128) -> CA -> c_t (N,128)
          z_s = FusionMLP(c_t) (N,128)
          [z_s, x_t] (N,173) -> Actor MLP -> 12-D

    * ``lstm_patch_ca`` (target)::
          proprio history (N,21,45) -> LSTM -> h_t (N,128)
          current frame x_t = history[:, -1] -> Query proj -> single Query -> CA over patches -> c_t (N,128)
          z_s = FusionMLP([h_t; c_t]) (N,128)
          [z_s, x_t] (N,173) -> Actor MLP -> 12-D
    """

    def __init__(
        self,
        depth_channels: int,
        depth_height: int,
        depth_width: int,
        proprio_dim: int,
        action_dim: int,
        depth_feature_dim: int = 256,
        proprio_feature_dim: int = 128,
        head_dims: list[int] | None = None,
        init_std: float = 0.5,
        fusion_type: str = "concat",
        attn_embed_dim: int = 128,
        attn_num_queries: int = 4,
        attn_num_heads: int = 4,
        attn_ffn_dim: int = 256,
        patch_size: int = 4,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 1,
        ca_embed_dim: int = 128,
        ca_num_heads: int = 4,
        ca_ffn_dim: int = 256,
        ca_use_ffn: bool = True,
        history_len: int = 21,
        z_dim: int = 128,
    ):
        super().__init__()
        valid = ("concat", "cross_attention", "patch_ca", "lstm_patch_ca")
        if fusion_type not in valid:
            raise ValueError(f"Unknown fusion_type={fusion_type!r}; expected one of {valid}.")

        head_dims = head_dims if head_dims is not None else [256, 128]

        # Store construction config for checkpoint metadata / reload.
        self.depth_channels = depth_channels
        self.depth_height = depth_height
        self.depth_width = depth_width
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.depth_feature_dim = depth_feature_dim
        self.proprio_feature_dim = proprio_feature_dim
        self.head_dims = list(head_dims)
        self.init_std = init_std
        self.fusion_type = fusion_type
        self.attn_embed_dim = attn_embed_dim
        self.attn_num_queries = attn_num_queries
        self.attn_num_heads = attn_num_heads
        self.attn_ffn_dim = attn_ffn_dim
        self.patch_size = patch_size
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.ca_embed_dim = ca_embed_dim
        self.ca_num_heads = ca_num_heads
        self.ca_ffn_dim = ca_ffn_dim
        self.ca_use_ffn = ca_use_ffn
        self.history_len = history_len
        self.z_dim = z_dim

        self.proprio_encoder = _ProprioEncoder(proprio_dim, proprio_feature_dim)

        if fusion_type == "concat":
            self.depth_encoder = _DepthEncoder(depth_channels, depth_feature_dim, depth_height, depth_width)
            self.fusion = None
            fusion_out = depth_feature_dim + proprio_feature_dim
            self.head = _mlp_head(fusion_out, action_dim, self.head_dims)
        elif fusion_type == "cross_attention":
            self.depth_encoder = _DepthEncoderSpatial(depth_channels, depth_height, depth_width)
            num_tokens = (depth_height // 8) * (depth_width // 8)
            self.fusion = _CrossAttentionFusion(
                token_dim=self.depth_encoder.out_channels,
                embed_dim=attn_embed_dim,
                num_tokens=num_tokens,
                num_queries=attn_num_queries,
                num_heads=attn_num_heads,
                ffn_dim=attn_ffn_dim,
            )
            fusion_out = attn_embed_dim + proprio_feature_dim
            self.head = _mlp_head(fusion_out, action_dim, self.head_dims)
        else:  # patch_ca / lstm_patch_ca
            self.depth_patcher = _DepthPatcher(depth_height, depth_width, patch_size, depth_channels)
            num_tokens = self.depth_patcher.num_tokens
            self.fusion = _PatchCrossAttention(
                token_dim=self.depth_patcher.token_dim,
                proprio_dim=proprio_dim,
                embed_dim=ca_embed_dim,
                num_tokens=num_tokens,
                num_heads=ca_num_heads,
                ffn_dim=ca_ffn_dim,
                use_ffn=ca_use_ffn,
            )
            self.proprio_lstm = None
            if fusion_type == "lstm_patch_ca":
                self.proprio_lstm = _ProprioHistoryLSTM(proprio_dim, lstm_hidden_dim, lstm_num_layers)
                fusion_in = lstm_hidden_dim + ca_embed_dim
            else:
                fusion_in = ca_embed_dim
            self.fusion_mlp = _FusionMLP(fusion_in, z_dim)
            # Actor input = [z_s (z_dim), x_t (proprio_dim)].
            self.head = _mlp_head(z_dim + proprio_dim, action_dim, self.head_dims)

        self.log_std = nn.Parameter(torch.log(torch.full((action_dim,), init_std)))
        self.distribution: Normal | None = None

    def _forward_patch_impl(self, depth, proprio, return_attention):
        """Shared core for ``patch_ca`` / ``lstm_patch_ca`` returning (z_s, a_s, attn)."""
        if self.fusion_type == "lstm_patch_ca":
            x_t = proprio[:, -1, :]  # (B, 45)
            h_t = self.proprio_lstm(proprio)  # (B, lstm_hidden)
        else:
            x_t = proprio
            h_t = None
        tokens = self.depth_patcher(depth)  # (B, N, token_dim)
        c_t, attn = self.fusion(tokens, x_t, return_attention=return_attention)  # (B, E)
        if h_t is not None:
            z_s = self.fusion_mlp(torch.cat([h_t, c_t], dim=-1))
        else:
            z_s = self.fusion_mlp(c_t)
        a_s = self.head(torch.cat([z_s, x_t], dim=-1))
        return z_s, a_s, attn

    def _forward_patch(self, depth, proprio, return_attention):
        z_s, a_s, attn = self._forward_patch_impl(depth, proprio, return_attention)
        if return_attention:
            return a_s, attn
        return a_s

    def forward_with_latent(self, depth, proprio) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(z_s, a_s)`` for the patch/lstm fusions (needed for dual-loss distillation)."""
        if self.fusion_type not in ("patch_ca", "lstm_patch_ca"):
            raise NotImplementedError(
                f"forward_with_latent is only defined for patch_ca/lstm_patch_ca, got {self.fusion_type!r}."
            )
        z_s, a_s, _ = self._forward_patch_impl(depth, proprio, return_attention=False)
        return z_s, a_s

    def forward(
        self,
        depth: torch.Tensor,
        proprio: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.fusion_type in ("patch_ca", "lstm_patch_ca"):
            return self._forward_patch(depth, proprio, return_attention)

        proprio_feat = self.proprio_encoder(proprio)
        if self.fusion_type == "concat":
            depth_feat = self.depth_encoder(depth)
            fused = torch.cat([depth_feat, proprio_feat], dim=-1)
            return self.head(fused)

        spatial = self.depth_encoder(depth)  # (B, C, H, W)
        pooled, attn = self.fusion(spatial, proprio_feat, return_attention=return_attention)
        fused = torch.cat([pooled, proprio_feat], dim=-1)
        out = self.head(fused)
        if return_attention:
            return out, attn
        return out

    def mean_action(self, depth: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.forward(depth, proprio)

    def update_distribution(self, depth: torch.Tensor, proprio: torch.Tensor) -> None:
        mean = self.forward(depth, proprio)
        std = self.log_std.exp().expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, depth: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        self.update_distribution(depth, proprio)
        return self.distribution.sample()

    @property
    def std(self) -> torch.Tensor:
        return self.log_std.exp()

    def to_config(self) -> dict:
        """Serialisable construction config (all keyword args of ``__init__``)."""
        cfg = {
            "depth_channels": self.depth_channels,
            "depth_height": self.depth_height,
            "depth_width": self.depth_width,
            "proprio_dim": self.proprio_dim,
            "action_dim": self.action_dim,
            "depth_feature_dim": self.depth_feature_dim,
            "proprio_feature_dim": self.proprio_feature_dim,
            "head_dims": list(self.head_dims),
            "init_std": self.init_std,
            "fusion_type": self.fusion_type,
        }
        if self.fusion_type == "cross_attention":
            cfg.update(
                attn_embed_dim=self.attn_embed_dim,
                attn_num_queries=self.attn_num_queries,
                attn_num_heads=self.attn_num_heads,
                attn_ffn_dim=self.attn_ffn_dim,
            )
        elif self.fusion_type in ("patch_ca", "lstm_patch_ca"):
            cfg.update(
                patch_size=self.patch_size,
                ca_embed_dim=self.ca_embed_dim,
                ca_num_heads=self.ca_num_heads,
                ca_ffn_dim=self.ca_ffn_dim,
                ca_use_ffn=self.ca_use_ffn,
                z_dim=self.z_dim,
            )
            if self.fusion_type == "lstm_patch_ca":
                cfg.update(
                    lstm_hidden_dim=self.lstm_hidden_dim,
                    lstm_num_layers=self.lstm_num_layers,
                    history_len=self.history_len,
                )
        return cfg


# Legacy v1 Student geometry (no metadata): these are the fixed defaults for the
# original ``DepthStudentPolicy`` produced by ``train_distill.py`` before metadata was
# added to checkpoints.
LEGACY_STUDENT_DEFAULTS = {
    "depth_channels": 1,
    "depth_height": 64,
    "depth_width": 64,
    "proprio_dim": 45,
    "action_dim": 12,
}


def ckpt_model_config(ckpt: dict) -> dict:
    """Extract a ``DepthStudentPolicy`` construction config from a checkpoint dict.

    Checkpoints written by the new ``train_distill.py`` carry a ``model_config`` key. Older
    (v1) Student checkpoints have no metadata; they are mapped to the fixed legacy geometry
    and the ``concat`` fusion so they keep loading deterministically.
    """
    cfg = dict(ckpt.get("model_config") or {})
    if not cfg:
        cfg = dict(LEGACY_STUDENT_DEFAULTS, fusion_type="concat")
    cfg.setdefault("fusion_type", "concat")
    return cfg


_FUSION_EXTRA_KEYS = {
    "cross_attention": {"attn_embed_dim", "attn_num_queries", "attn_num_heads", "attn_ffn_dim"},
    "patch_ca": {"patch_size", "ca_embed_dim", "ca_num_heads", "ca_ffn_dim", "ca_use_ffn", "z_dim"},
    "lstm_patch_ca": {
        "patch_size",
        "ca_embed_dim",
        "ca_num_heads",
        "ca_ffn_dim",
        "ca_use_ffn",
        "z_dim",
        "lstm_hidden_dim",
        "lstm_num_layers",
        "history_len",
    },
}


def build_student_from_config(config: dict) -> DepthStudentPolicy:
    """Construct a ``DepthStudentPolicy`` from a ``ckpt_model_config`` dict."""
    cfg = dict(config)
    fusion_type = cfg.pop("fusion_type", "concat")
    accepted = {
        "depth_channels", "depth_height", "depth_width", "proprio_dim", "action_dim",
        "depth_feature_dim", "proprio_feature_dim", "head_dims", "init_std",
    } | _FUSION_EXTRA_KEYS.get(fusion_type, set())
    cfg = {k: v for k, v in cfg.items() if k in accepted}
    return DepthStudentPolicy(fusion_type=fusion_type, **cfg)


def load_student_policy(
    ckpt_path: str,
    device: str,
    fusion_type: str | None = None,
    strict: bool = True,
) -> DepthStudentPolicy:
    """Load a Student policy from a checkpoint, honouring (and validating) metadata.

    * If ``fusion_type`` is given and differs from the checkpoint's recorded type, a clear
      ``ValueError`` is raised (no silent parameter drop).
    * v1 checkpoints (no metadata) load as ``concat``.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt_model_config(ckpt)
    actual = config.get("fusion_type", "concat")
    if fusion_type is not None and fusion_type != actual:
        raise ValueError(
            f"Checkpoint fusion_type={actual!r} does not match requested {fusion_type!r}; "
            f"refusing to load {ckpt_path}."
        )
    if "student_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint has no 'student_state_dict'; keys: {list(ckpt.keys())}")
    policy = build_student_from_config(config)
    policy.load_state_dict(ckpt["student_state_dict"], strict=strict)
    return policy.to(device)
