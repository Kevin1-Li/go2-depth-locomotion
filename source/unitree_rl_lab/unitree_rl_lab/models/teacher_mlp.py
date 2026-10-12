from __future__ import annotations

import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    return {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "lrelu": nn.LeakyReLU,
    }[name]()


class TeacherMLPPolicy(nn.Module):
    """Reconstructed privileged Teacher policy (the frozen behaviour-cloning labeler).

    The converged Teacher checkpoint (``model_29999.pt``) is stored in the rsl-rl 5.x
    layout: an ``actor_state_dict`` whose keys are ``mlp.{i}.weight/bias`` (an MLP with
    activation layers between the linear layers) plus a ``distribution.std_param``. This
    class rebuilds the exact same MLP geometry and loads only the ``mlp.*`` weights, so the
    deterministic mean action it produces is bit-identical to the original Teacher.

    The input is the privileged 232-D observation (45-D proprio + 187-D height scan); the
    output is the 12-D joint-position action mean.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], action_dim: int, activation: str = "elu"):
        super().__init__()
        self.hidden_dims = list(hidden_dims)
        self.latent_dim = int(hidden_dims[-1])
        dims = [input_dim] + list(hidden_dims) + [action_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(_get_activation(activation))
        self.mlp = nn.Sequential(*layers)
        # Index of the final linear layer (the action head). Everything before it — up to and
        # including the activation that follows the penultimate linear — is the latent trunk.
        # ``forward_with_latent`` splits here so that the latent equals the pre-head feature.
        self._latent_split = len(self.mlp) - 1

    def load_actor_state_dict(self, actor_state_dict: dict[str, torch.Tensor]) -> None:
        mlp_state = {k.removeprefix("mlp."): v for k, v in actor_state_dict.items() if k.startswith("mlp.")}
        self.mlp.load_state_dict(mlp_state, strict=True)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)

    def forward_with_latent(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint latent + action forward.

        Returns ``(z_t, a_t)`` where ``z_t`` is the pre-action-head latent (the output of the
        activation following the penultimate linear layer, ``latent_dim`` features) and
        ``a_t`` is the deterministic action mean. ``a_t`` is bit-identical to ``forward(obs)``.
        """
        z_t = self.mlp[: self._latent_split](obs)
        a_t = self.mlp[self._latent_split :](z_t)
        return z_t, a_t

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)
