from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def depth_image(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, far_clip: float = 10.0) -> torch.Tensor:
    """Normalised depth image from a :class:`RayCasterCamera` depth sensor.

    Clips the raw ``distance_to_image_plane`` image to :math:`[0, far_clip]` and divides by
    ``far_clip`` so values live in :math:`[0, 1]`. Returns the image channels-first as
    ``(num_envs, 1, H, W)``, the convention expected by the student's convolutional depth
    encoder.
    """
    sensor: RayCasterCamera = env.scene.sensors[sensor_cfg.name]
    depth = sensor.data.output["distance_to_image_plane"]
    depth = depth.clamp(min=0.0, max=far_clip) / far_clip
    depth = depth.permute(0, 3, 1, 2).contiguous()
    return depth.clone()
