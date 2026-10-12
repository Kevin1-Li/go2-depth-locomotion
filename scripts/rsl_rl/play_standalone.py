# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Standalone replay for the blind-walk / Teacher / Student Go2 policies.
#
# This script is SELF-CONTAINED: it does NOT import ``rsl_rl``. The installed rsl-rl is
# the 2.3.4 SRU fork, which has no 5.x ``MLPModel`` / ``Distillation`` classes, so the
# official ``scripts/rsl_rl/play.py`` (which reconstructs the policy through the rsl-rl
# runner) cannot load the rsl-rl 5.x ``actor_state_dict`` checkpoints produced for these
# three tasks. Instead we rebuild the exact network geometry from the unitree models:
#
#   * blind   -> TeacherMLPPolicy(45)     : generic MLP 512-256-128, loaded from
#                actor_state_dict ``mlp.*`` keys of ``model_49999.pt``.
#   * teacher -> TeacherMLPPolicy(232)    : same MLP, privileged 45-D proprio + 187-D
#                height_scan input, loaded from ``model_29999.pt``.
#   * student -> DepthStudentPolicy       : CNN depth encoder + proprio branch, loaded
#                from ``student_state_dict`` of ``student_model_2999.pt``.
#
# All three use deterministic mean actions (no sampling noise), which is exactly the
# behaviour of the reference scripts (train_distill.py gate, official play.py policy()).
#
# Usage (GUI replay, num_envs defaults to 8):
#   ~/IsaacLab/_isaac_sim/python.sh scripts/rsl_rl/play_standalone.py --model blind
#   ~/IsaacLab/_isaac_sim/python.sh scripts/rsl_rl/play_standalone.py --model teacher
#   ~/IsaacLab/_isaac_sim/python.sh scripts/rsl_rl/play_standalone.py --model student
#
# Keyboard control (single robot, real-time, WASD from the terminal):
#   ~/IsaacLab/_isaac_sim/python.sh scripts/rsl_rl/play_standalone.py --model student --keyboard
#
# Headless smoke (run a bounded number of steps and print tracking evidence):
#   ~/IsaacLab/_isaac_sim/python.sh scripts/rsl_rl/play_standalone.py --model blind   --headless --num_envs 4 --max_steps 200
#   ~/IsaacLab/_isaac_sim/python.sh scripts/rsl_rl/play_standalone.py --model teacher --headless --num_envs 4 --max_steps 200
#   ~/IsaacLab/_isaac_sim/python.sh scripts/rsl_rl/play_standalone.py --model student --headless --num_envs 4 --max_steps 200

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Standalone replay of Go2 blind/teacher/student policies.")
parser.add_argument("--model", type=str, required=True, choices=["blind", "teacher", "student"],
                    help="Which policy to replay.")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Explicit checkpoint path. Defaults to each model's latest.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments (default 8 for GUI).")
parser.add_argument("--max_steps", type=int, default=None,
                    help="Stop after this many control steps (default: run until the window is closed; "
                         "headless defaults to 200).")
parser.add_argument("--real_time", action="store_true", default=False,
                    help="Sleep each step to run at approximately real-time speed.")
parser.add_argument("--forward", action="store_true", default=False,
                    help="Demo mode: lock velocity commands to forward-only (vx 0.5-1.0 m/s, no lateral/yaw/standing) "
                         "so all robots walk forward for cleaner comparison videos.")
parser.add_argument("--fusion_type", type=str, default=None,
                    choices=["concat", "cross_attention", "patch_ca", "lstm_patch_ca"],
                    help="Optional fusion_type to validate against the checkpoint metadata. "
                         "Omitted => auto-detect from checkpoint metadata (v1 => concat).")
parser.add_argument("--keyboard", action="store_true", default=False,
                    help="Manual control mode: forces num_envs=1 + real_time, disables command "
                         "resampling/heading/standing. HOLD-to-drive / RELEASE-to-stop from the "
                         "terminal: W/S forward-back, A/D turn, Q/E strafe, SPACE brake. Command "
                         "ramps to cruise while held and decays to zero on release (hold detection "
                         "via terminal key auto-repeat). Keep the TERMINAL focused for key input.")
parser.add_argument("--speed_scale", type=float, default=1.0,
                    help="Keyboard mode only: scale cruise targets and command clamps. "
                         "Values >1.0 exceed the trained command range (vx cap 1.0 m/s) — "
                         "the policy goes out-of-distribution and tracking degrades.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import select  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.models import DepthStudentPolicy, TeacherMLPPolicy, load_student_policy  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Model registry: task id, deployment observation group, network geometry,
# and default (latest) checkpoint path.
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "blind": {
        "task": "Unitree-Go2-Velocity-BlindEval",
        "obs_group": "policy",
        "input_dim": 45,
        "default_ckpt": "logs/rsl_rl/unitree_go2_velocity/2026-09-22_18-57-19/model_49999.pt",
    },
    "teacher": {
        "task": "Unitree-Go2-Velocity-Teacher",
        "obs_group": "policy",
        "input_dim": 232,
        "default_ckpt": "logs/rsl_rl/unitree_go2_teacher/2026-10-10_15-00-10/model_19999.pt",
    },
    "student": {
        "task": "Unitree-Go2-Depth-Student",
        "obs_group": "student",
        "input_dim": 45,  # proprio branch dimension (depth is a separate 1x32x48 tensor)
        "default_ckpt": "logs/rsl_rl/unitree_go2_student_lstm_ca/2026-10-11_20-45-12_lstm_patch_ca_long_s42/student_model_9999.pt",
    },
}

# Proprio observation term order for the student "student" group (matches train_distill.py).
STUDENT_PROPRIO_KEYS = [
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
]


def build_policy(model: str, device: str) -> torch.nn.Module:
    """Construct the (uninitialised) network for the requested model."""
    if model in ("blind", "teacher"):
        return TeacherMLPPolicy(
            input_dim=MODEL_REGISTRY[model]["input_dim"],
            hidden_dims=[512, 256, 128],
            action_dim=12,
            activation="elu",
        )
    # student
    return DepthStudentPolicy(depth_channels=1, depth_height=32, depth_width=48, proprio_dim=45, action_dim=12)


def load_policy(model: str, ckpt_path: str, device: str) -> torch.nn.Module:
    """Load weights into the policy from a checkpoint and move it to ``device``.

    The Student policy is reconstructed from checkpoint metadata (fusion type + structure
    hyperparams), so a ``cross_attention`` checkpoint auto-builds the cross-attention net
    and a legacy v1 (metadata-less) checkpoint builds the ``concat`` net.
    """
    if model == "student":
        policy = load_student_policy(ckpt_path, device, fusion_type=args_cli.fusion_type)
        policy.eval()
        return policy

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    policy = build_policy(model, device)
    if "actor_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint has no 'actor_state_dict'; keys: {list(ckpt.keys())}")
    policy.load_actor_state_dict(ckpt["actor_state_dict"])
    policy.to(device).eval()
    return policy


def policy_act(policy: torch.nn.Module, model: str, obs: dict) -> torch.Tensor:
    """Deterministic mean action from the deployment observation group."""
    if model == "student":
        student = obs["student"]
        proprio_hist = torch.cat([student[k] for k in STUDENT_PROPRIO_KEYS], dim=-1)
        depth = student["depth"]
        if policy.fusion_type == "lstm_patch_ca":
            proprio = proprio_hist  # (num_envs, 21, 45)
        else:
            proprio = proprio_hist[:, -1, :]  # (num_envs, 45)
        return policy(depth, proprio)
    return policy(obs[MODEL_REGISTRY[model]["obs_group"]])


class KeyboardCommander:
    """Terminal-keyboard velocity driver with hold-to-drive / release-to-stop semantics.

    Terminals deliver no key-release events, so "held" is inferred from the OS key
    auto-repeat: a key counts as held while its characters keep arriving within
    REPEAT_TIMEOUT. While held, the command ramps toward the axis cruise target at
    ACCEL; once released it decays back to zero at DECEL (so the robot receives no
    velocity command after you let go). SPACE brakes to zero immediately. If stdin is
    not a tty (headless smoke tests), the commander degrades to a no-op holding [0,0,0].
    """

    KEYMAP = {
        "w": (0, +1),  # vx forward
        "s": (0, -1),  # vx backward
        "q": (1, +1),  # vy strafe left
        "e": (1, -1),  # vy strafe right
        "a": (2, +1),  # yaw left (CCW positive)
        "d": (2, -1),  # yaw right
    }
    CRUISE = (1.0, 0.4, 1.0)  # |target| per axis while held: vx m/s, vy m/s, yaw rad/s
    ACCEL = (2.5, 1.2, 3.5)  # ramp-up rate while held (units/s^2)
    DECEL = (3.0, 1.5, 4.0)  # decay-to-zero rate after release (units/s^2)
    REPEAT_TIMEOUT = 0.12  # s; must exceed the terminal auto-repeat interval (~25-30Hz)
    HELP = (
        "=== KEYBOARD CONTROL (keep THIS TERMINAL focused) ===\n"
        "  HOLD to drive, RELEASE to stop (hold detection via key auto-repeat):\n"
        "  W / S : forward / backward   (cruise +/-1.0 m/s)\n"
        "  A / D : turn left / right    (cruise +/-1.0 rad/s)\n"
        "  Q / E : strafe left / right  (cruise +/-0.4 m/s)\n"
        "  SPACE : brake to zero immediately\n"
        "  Close the GUI window to quit.\n"
        "  NOTE: needs terminal key auto-repeat ON (the OS default).\n"
        "======================================================"
    )

    def __init__(self, ranges, speed_scale: float = 1.0):
        self.ranges = [tuple(r) for r in ranges]  # [(lo, hi)] for vx, vy, yaw
        self.cruise = tuple(c * speed_scale for c in self.CRUISE)
        self.cmd = [0.0, 0.0, 0.0]
        self.enabled = False
        self._old_attr = None
        self._last_key = {}
        self._last_poll = time.monotonic()
        self._last_print = 0.0
        try:
            if sys.stdin is not None and sys.stdin.isatty():
                import termios
                import tty

                self._termios = termios
                self._tty = tty
                self._fd = sys.stdin.fileno()
                self._old_attr = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
                self.enabled = True
        except Exception as exc:  # noqa: BLE001 - never let keyboard setup kill the replay
            print(f"[WARN] keyboard input unavailable ({exc}); commands stay [0, 0, 0].", flush=True)
            self.enabled = False
        print(self.HELP if self.enabled else "[WARN] stdin is not a tty: keyboard disabled.", flush=True)

    def poll(self):
        """Drain pending keys, advance the command by wall-clock dt toward targets."""
        if not self.enabled:
            return
        now = time.monotonic()
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if not ch:
                break
            ch = ch.lower()
            if ch == " ":
                self.cmd = [0.0, 0.0, 0.0]
                self._last_key.clear()
                self._announce(now, tag="BRAKE")
            elif ch in self.KEYMAP:
                self._last_key[ch] = now

        dt = min(now - self._last_poll, 0.1)
        self._last_poll = now
        changed = False
        for axis in range(3):
            sign, newest = 0, -1.0
            for key, (ax, sgn) in self.KEYMAP.items():
                t = self._last_key.get(key)
                if ax == axis and t is not None and now - t < self.REPEAT_TIMEOUT and t > newest:
                    newest, sign = t, sgn
            if sign != 0:
                target, rate = sign * self.cruise[axis], self.ACCEL[axis]
            else:
                target, rate = 0.0, self.DECEL[axis]
            delta = target - self.cmd[axis]
            max_step = rate * dt
            new = self.cmd[axis] + max(-max_step, min(max_step, delta))
            lo, hi = self.ranges[axis]
            new = min(max(new, lo), hi)
            if abs(new - self.cmd[axis]) > 1e-9:
                self.cmd[axis] = new
                changed = True
        if changed and (now - self._last_print > 0.25
                        or all(abs(c) < 1e-9 for c in self.cmd)):
            self._announce(now)

    def _announce(self, now, tag=None):
        self._last_print = now
        prefix = f"[cmd {tag}]" if tag else "[cmd]"
        print(f"{prefix} vx={self.cmd[0]:+.2f} m/s  vy={self.cmd[1]:+.2f} m/s  "
              f"yaw={self.cmd[2]:+.2f} rad/s", flush=True)

    def restore(self):
        if self.enabled and self._old_attr is not None:
            try:
                self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._old_attr)
            except Exception:  # noqa: BLE001
                pass
            self.enabled = False


def main():
    model = args_cli.model
    meta = MODEL_REGISTRY[model]
    ckpt_path = args_cli.checkpoint or meta["default_ckpt"]

    if args_cli.keyboard:
        if args_cli.forward:
            raise SystemExit("[ERROR] --keyboard and --forward are mutually exclusive.")
        args_cli.num_envs = 1
        args_cli.real_time = True
        print("[INFO] --keyboard: forcing num_envs=1, real_time=True.", flush=True)

    env_cfg = parse_env_cfg(
        meta["task"],
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.seed = 42
    kb_ranges = None
    if args_cli.keyboard:
        # Freeze the command term: no resampling, no heading override, no standing envs,
        # so the manual vel_command_b writes below are the single source of truth.
        cmd_cfg = env_cfg.commands.base_velocity
        cmd_cfg.rel_heading_envs = 0.0
        cmd_cfg.rel_standing_envs = 0.0
        cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
        # Kill the 20s training-episode truncation: no mid-drive respawn to the spawn point.
        env_cfg.episode_length_s = 3600.0
        lim = getattr(cmd_cfg, "limit_ranges", None) or cmd_cfg.ranges
        scale = args_cli.speed_scale
        kb_ranges = [(lo * scale, hi * scale) for lo, hi in
                     (lim.lin_vel_x, lim.lin_vel_y, lim.ang_vel_z)]
        print(f"[INFO] keyboard clamps (speed_scale={scale}): "
              f"vx={kb_ranges[0]} vy={kb_ranges[1]} yaw={kb_ranges[2]}; "
              f"episode truncation disabled (3600s)", flush=True)
    if args_cli.forward:
        cmd_cfg = env_cfg.commands.base_velocity
        cmd_cfg.ranges.lin_vel_x = (0.5, 1.0)
        cmd_cfg.ranges.lin_vel_y = (0.0, 0.0)
        cmd_cfg.ranges.ang_vel_z = (0.0, 0.0)
        cmd_cfg.limit_ranges.lin_vel_x = (0.5, 1.0)
        cmd_cfg.limit_ranges.lin_vel_y = (0.0, 0.0)
        cmd_cfg.limit_ranges.ang_vel_z = (0.0, 0.0)
        cmd_cfg.rel_standing_envs = 0.0
        print("[INFO] --forward demo mode: commands locked to vx in [0.5, 1.0] m/s, no lateral/yaw/standing.",
              flush=True)
    env = gym.make(meta["task"], cfg=env_cfg)

    device = env.unwrapped.device
    policy = load_policy(model, ckpt_path, device)
    n_params = sum(p.numel() for p in policy.parameters())
    ft = getattr(policy, "fusion_type", None)
    print(f"[INFO] model={model} task={meta['task']} ckpt={ckpt_path} fusion_type={ft}", flush=True)
    print(f"[INFO] num_envs={env.unwrapped.num_envs} device={device} params={n_params}", flush=True)

    reward_manager = env.unwrapped.reward_manager
    track_idx = reward_manager._term_names.index("track_lin_vel_xy")

    if args_cli.max_steps is None and args_cli.headless:
        max_steps = 200
    else:
        max_steps = args_cli.max_steps

    step = 0
    track_acc = 0.0
    reward_acc = 0.0
    obs, _ = env.reset()
    print("[INFO] starting replay loop", flush=True)

    commander = KeyboardCommander(kb_ranges, speed_scale=args_cli.speed_scale) if args_cli.keyboard else None
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity") if commander is not None else None

    try:
        with torch.inference_mode():
            while simulation_app.is_running():
                if commander is not None:
                    commander.poll()
                    cmd_term.vel_command_b[:] = torch.as_tensor(commander.cmd, dtype=torch.float32,
                                                                device=device)

                actions = policy_act(policy, model, obs)
                obs, reward, term, trunc, extras = env.step(actions)

                track_acc += reward_manager._step_reward[:, track_idx].mean().item()
                reward_acc += reward.mean().item()
                step += 1

                if step % 100 == 0:
                    extra = ""
                    if commander is not None:
                        robot = env.unwrapped.scene["robot"]
                        extra = (f" cmd=[{commander.cmd[0]:+.2f}, {commander.cmd[1]:+.2f}, "
                                 f"{commander.cmd[2]:+.2f}]"
                                 f" | actual vx={robot.data.root_lin_vel_b[0, 0].item():+.2f}"
                                 f" vy={robot.data.root_lin_vel_b[0, 1].item():+.2f}"
                                 f" yaw_rate={robot.data.root_ang_vel_b[0, 2].item():+.2f}"
                                 f" heading={robot.data.heading_w[0].item():+.2f}rad")
                    print(
                        f"[step {step}] mean_total_reward={reward_acc / step:6.2f} "
                        f"mean_track_lin_vel_xy={track_acc / step:5.3f}{extra}",
                        flush=True,
                    )

                if max_steps is not None and step >= max_steps:
                    break

                if args_cli.real_time:
                    time.sleep(env.unwrapped.step_dt)
    finally:
        if commander is not None:
            commander.restore()

    print(f"[DONE] steps={step} mean_total_reward={reward_acc / max(step, 1):6.2f} "
          f"mean_track_lin_vel_xy={track_acc / max(step, 1):5.3f}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    simulation_app.close()
