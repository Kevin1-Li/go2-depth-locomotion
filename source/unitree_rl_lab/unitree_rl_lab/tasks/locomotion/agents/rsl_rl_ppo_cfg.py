# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class GaussianDistributionCfg:
    """Configuration for the Gaussian output distribution (rsl-rl >= 5.0.0)."""

    class_name: str = "GaussianDistribution"
    init_std: float = 1.0
    std_type: str = "scalar"
    learn_std: bool = True


@configclass
class MlpModelCfg:
    """Configuration for an MLP model (rsl-rl >= 5.0.0 separate actor/critic API)."""

    class_name: str = "MLPModel"
    hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"
    obs_normalization: bool = False
    distribution_cfg: object = None


@configclass
class PpoAlgorithmCfg:
    """Configuration for the PPO algorithm (rsl-rl >= 5.0.0 API)."""

    class_name: str = "PPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    optimizer: str = "adam"
    normalize_advantage_per_mini_batch: bool = False
    rnd_cfg: object = None
    symmetry_cfg: object = None


@configclass
class BasePPORunnerCfg:
    """On-policy PPO runner configuration (rsl-rl >= 5.0.0 API).

    This replaces the deprecated ``RslRlOnPolicyRunnerCfg`` + ``RslRlPpoActorCriticCfg``
    layout used by Isaac Lab 2.3.x (rsl-rl-lib 3.0.1). The installed rsl-rl-lib is 5.4.2,
    which expects separate ``actor`` / ``critic`` model configs, ``obs_groups``, and a
    ``distribution_cfg`` inside the stochastic actor model.
    """

    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 50000
    save_interval: int = 100
    experiment_name: str = ""
    run_name: str = ""
    logger: str = "tensorboard"
    neptune_project: str = "isaaclab"
    wandb_project: str = "isaaclab"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    clip_actions: float | None = None
    check_for_nan: bool = True
    obs_groups: dict[str, list[str]] = {"actor": ["policy"], "critic": ["critic"]}
    actor: MlpModelCfg = MlpModelCfg(distribution_cfg=GaussianDistributionCfg())
    critic: MlpModelCfg = MlpModelCfg(distribution_cfg=None)
    algorithm: PpoAlgorithmCfg = PpoAlgorithmCfg()
